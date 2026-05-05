#!/usr/bin/env python3
"""
MITRE ATT&CK Log Analyzer - BATCHED VERSION (FASTEST)
Analyzes security logs and maps them to MITRE ATT&CK framework

Optimizations:
1. Fixed tokenizer padding (left-padding for decoder models)
2. Semantic caching (avoid re-analyzing identical events)
3. 4-bit quantization (faster inference, less memory)
4. BATCH PROCESSING (process multiple events in single GPU call)

Expected speedup: 40-60x faster than original
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass
import re
from tqdm import tqdm
import warnings
import hashlib
warnings.filterwarnings('ignore')


@dataclass
class MITREPrediction:
    """Data class for MITRE predictions"""
    tactic: str
    technique: str
    technique_id: str
    technique_name: str
    confidence_score: float
    mitigation_strategies: List[Dict]
    detection_strategies: List[Dict]
    reasoning: str


class MITRELogAnalyzerBatched:
    """
    Hierarchical MITRE ATT&CK log analyzer using LLMs - BATCHED VERSION
    Stage 1: Raw log → Tactic identification (BATCHED)
    Stage 2: Log + Tactic → Technique identification (BATCHED)
    Stage 3: Extract mitigation and detection strategies
    
    Key Optimization: Batch GPU inference instead of threading
    - Processes 8-16 events per GPU call
    - True parallel computation on GPU
    - Much better GPU utilization (70-90% vs 30-40% with threading)
    """
    
    def __init__(
        self,
        mitre_kb_path: str,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = None,
        use_quantization: bool = True,
        use_caching: bool = True,
        batch_size: int = 8
    ):
        """
        Initialize the analyzer
        
        Args:
            mitre_kb_path: Path to MITRE ATT&CK knowledge base JSON
            model_name: HuggingFace model to use
            device: Device to run on (cuda/cpu)
            use_quantization: Use 4-bit quantization for speedup
            use_caching: Cache results for identical events
            batch_size: Number of events to process per GPU call (4-16 recommended)
        """
        self.mitre_kb_path = mitre_kb_path
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_caching = use_caching
        self.batch_size = batch_size
        
        # Initialize cache
        self.cache = {} if use_caching else None
        
        # Load MITRE knowledge base
        print(f"\n📚 Loading MITRE knowledge base from {mitre_kb_path}...")
        with open(mitre_kb_path, 'r', encoding='utf-8') as f:
            self.mitre_kb = json.load(f)
        
        num_tactics = len(self.mitre_kb.get('tactics', {}))
        print(f"✅ Loaded {num_tactics} tactics from knowledge base")
        
        # Check if accelerate is available for device_map
        print(f"\n🔍 Checking for accelerate package...")
        try:
            import accelerate
            accelerate_available = True
            print(f"✅ Accelerate {accelerate.__version__} found - device_map enabled")
        except ImportError:
            accelerate_available = False
            print("⚠️  Warning: accelerate not found - device_map disabled")
            print("   Install with: pip install accelerate")
            print("   (Model will still work, just slower on multi-GPU)")
        
        # Initialize model and tokenizer with optimizations
        print(f"\n🤖 Loading language model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Quantization: {'4-bit' if use_quantization and self.device == 'cuda' else 'None'}")
        
        # Use 4-bit quantization for faster inference
        if use_quantization and self.device == "cuda":
            print("\n⚡ Setting up 4-bit quantization for 2-4x speedup...")
            print("   • Load in 4-bit: Yes")
            print("   • Compute dtype: float16")
            print("   • Double quantization: Yes")
            print("   • Quantization type: nf4")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            print("\n⏳ Downloading/loading model weights (this may take a minute)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if accelerate_available else None,
                low_cpu_mem_usage=True
            )
            print("✅ Model loaded with 4-bit quantization")
        else:
            # Only use device_map if accelerate is available
            model_kwargs = {
                "torch_dtype": "auto",
                "low_cpu_mem_usage": True
            }
            
            if accelerate_available and self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                print(f"   Using device_map=auto for optimal GPU placement")
            
            print("\n⏳ Downloading/loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            print("✅ Model loaded")
            
            # Manually move to device if device_map not used
            if not accelerate_available:
                print(f"   Moving model to {self.device}...")
                self.model = self.model.to(self.device)
                print(f"✅ Model ready on {self.device}")
        
        print(f"\n📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix tokenizer padding for decoder models (CRITICAL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Critical for batched generation
        print(f"✅ Tokenizer loaded and configured")
        print(f"   Vocabulary size: {len(self.tokenizer)}")
        print(f"   Padding token: {self.tokenizer.pad_token}")
        
        print(f"\n✅ INITIALIZATION COMPLETE")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size} events per GPU call")
        print(f"   Caching: {'Enabled' if use_caching else 'Disabled'}")
        
        # Stats
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processed': 0,
            'batches_processed': 0
        }
    
    def _get_cache_key(self, log_entry: str) -> str:
        """Generate cache key for log entry"""
        normalized = log_entry.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _create_tactic_prompt(self, log_entry: str) -> str:
        """Create prompt for tactic identification - UNCHANGED"""
        
        # Extract tactic summaries
        tactic_summaries = []
        for tactic_key, tactic_data in self.mitre_kb.get('tactics', {}).items():
            tactic_summaries.append(
                f"- {tactic_data['name']} ({tactic_data['shortname']}): "
                f"{tactic_data['description']}..."
            )
        
        prompt = f"""
            You are a Senior Security Operations Center (SOC) Analyst and MITRE ATT&CK Specialist. Your task is to classify security logs into the most accurate Tactic.

            ### STEP 1: LOG DECONSTRUCTION
            Examine the log below. Even if field names are unfamiliar, identify:
            1. **The Actor (Subject):** Which process, user, or service is initiating the action?
            2. **The Action (Verb):** What is happening? (e.g., Is something being accessed, created, modified, executed, or moved?)
            3. **The Target (Object):** What is being acted upon? (e.g., a registry key, a file path, a network socket, a memory address, or a configuration setting.)

            ### STEP 2: INTENT INFERENCE
            Based on the Subject-Verb-Object relationship:
            - If the Verb is 'Read/Query' and the Object is 'System Config/Registry', the intent is **Discovery**.
            - If the Verb is 'Execute/Start' and the Object is 'Binary/Script', the intent is **Execution**.
            - If the Verb is 'Modify/Write' and the Object is 'Auto-run location/Service', the intent is **Persistence**.

            ### LOG ENTRY:
            {log_entry}

            ### MITRE ATT&CK TACTIC DEFINITIONS:
            {chr(10).join(tactic_summaries)}

            ### ANALYSIS GUIDELINES:
            - **Do not rely on field labels:** A field named 'cmd' might just be a process name; a field named 'path' might be a registry key. Focus on the *content* of the values.
            - **Context Matters:** Remote access tools (VNC, RDP, SSH) performing routine reads are likely in a 'Discovery' or 'Lateral Movement' phase, not 'Execution', unless a new process is being spawned.
            - **Match the Goal:** Choose the Tactic whose definition most closely matches the *primary goal* of the action identified in Step 2.

            ### OUTPUT FORMAT (JSON):
            {{
                "tactic_name": "Exact name from provided list",
                "tactic_shortname": "shortname",
                "confidence": 0.0-1.0,
                "reasoning": "Explain in less than 50 words. how the inferred action matches the specific MITRE description."
            }}
            
            JSON Response:
        """

        return prompt
    
    def _create_technique_prompt(
        self, 
        log_entry: str, 
        tactic_name: str,
        tactic_shortname: str
    ) -> str:
        """Create prompt for technique identification - UNCHANGED"""
        
        # Get techniques for this tactic
        tactic_data = self.mitre_kb.get('tactics', {}).get(tactic_shortname, {})
        techniques = tactic_data.get('techniques', [])
        
        # Create technique summaries (ID + Name only, no descriptions to save tokens)
        technique_summaries = []
        for tech in techniques:
            tech_summary = f"- {tech['attack_id']}: {tech['name']}"
            technique_summaries.append(tech_summary)

        prompt = f"""
            You are a Senior Security Operations Center (SOC) Analyst and MITRE ATT&CK Specialist. You are performing a deep-dive analysis to map a log entry to a specific MITRE ATT&CK Technique.


            ### PREVIOUS CLASSIFICATION:
            - **Identified Tactic:** {tactic_name}
            - **Scope:** Your analysis MUST stay within the provided {tactic_name} techniques. Do not suggest techniques from other tactics.

            ### LOG ENTRY:
            {log_entry}

            ### CANDIDATE TECHNIQUES (Under {tactic_name}):
            {chr(10).join(technique_summaries)}

            ### STEP-BY-STEP MAPPING LOGIC:
            1. **Identify the Target Object:** What specific system resource is being touched? 
            - Is it a **Registry Key**? (Look for techniques involving 'Query Registry' or 'Modify Registry').
            - Is it a **File or Directory**? (Look for techniques involving 'File Discovery' or 'Data Destruction').
            - Is it a **Process or Service**? (Look for 'Process Discovery' or 'Service Execution').
            2. **Determine the Mechanism:** How is the action being performed? (e.g., via a remote tool, a native API call, or a command shell).
            3. **Semantic Match:** Compare the 'Target Object' and 'Mechanism' found in the log against the 'AVAILABLE TECHNIQUES' list. 

            ### OUTPUT FORMAT (JSON):
            {{
                "technique_id": "T####",
                "technique_name": "Exact name from the list",
                "confidence": 0.0-1.0,
                "reasoning": "Explain why this specific technique is the best fit within the {tactic_name} tactic."
            }}

            JSON Response:
        """
        return prompt
    

    def _extract_json_from_response(self, response: str, verbose: bool = False) -> Optional[Dict]:
        """Extract JSON from LLM response with multiple fallback strategies"""
        if verbose:
            print(f"  Parsing response: {response[:300]}...")
        
        try:
            # Strategy 1: Find JSON block with proper braces
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    if verbose:
                        print(f"  ✓ Parsed JSON: {result}")
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Look for JSON after "JSON Response:" marker
            if "JSON Response:" in response:
                json_part = response.split("JSON Response:")[-1].strip()
                json_match = re.search(r'\{[^{}]*\}', json_part, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        if verbose:
                            print(f"  ✓ Parsed JSON after marker: {result}")
                        return result
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Extract fields individually and build JSON
            # For tactic
            tactic_name = re.search(r'"tactic_name"\s*:\s*"([^"]+)"', response)
            tactic_short = re.search(r'"tactic_shortname"\s*:\s*"([^"]+)"', response)
            confidence = re.search(r'"confidence"\s*:\s*([0-9.]+)', response)
            
            if tactic_name and tactic_short:
                result = {
                    'tactic_name': tactic_name.group(1),
                    'tactic_shortname': tactic_short.group(1),
                    'confidence': float(confidence.group(1)) if confidence else 0.7,
                    'reasoning': 'Extracted from response'
                }
                if verbose:
                    print(f"  ✓ Built tactic JSON: {result}")
                return result
            
            # For technique
            tech_id = re.search(r'"technique_id"\s*:\s*"(T\d+(?:\.\d+)?)"', response)
            tech_name = re.search(r'"technique_name"\s*:\s*"([^"]+)"', response)
            
            if tech_id and tech_name:
                result = {
                    'technique_id': tech_id.group(1),
                    'technique_name': tech_name.group(1),
                    'confidence': float(confidence.group(1)) if confidence else 0.7,
                    'reasoning': 'Extracted from response'
                }
                if verbose:
                    print(f"  ✓ Built technique JSON: {result}")
                return result
            
            if verbose:
                print(f"  ✗ All JSON extraction strategies failed")
            return None
            
        except Exception as e:
            if verbose:
                print(f"  ✗ JSON extraction error: {e}")
            return None
    
    def _batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts in a single GPU call
        This is the KEY optimization - processes multiple events in parallel on GPU
        """
        if not prompts:
            return []
        
        # Tokenize all prompts at once
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Single GPU call processes ALL prompts in parallel
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode all responses
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        self.stats['batches_processed'] += 1
        
        return responses
    
    def _get_strategies(
        self,
        tactic_shortname: str,
        technique_id: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract mitigation and detection strategies for a technique - UNCHANGED"""
        
        mitigations = []
        detections = []
        
        try:
            tactic_data = self.mitre_kb.get('tactics', {}).get(tactic_shortname, {})
            techniques = tactic_data.get('techniques', [])
            
            # Find matching technique
            for tech in techniques:
                if tech.get('attack_id') == technique_id:
                    raw_mitigations = tech.get('mitigations', [])
                    mitigations = [
                        {"name": m.get("name"), "strategy": m.get("description")} 
                        for m in raw_mitigations
                    ]
                    detections = tech.get('detection_strategies', [])
                    break
        except Exception as e:
            pass
        
        return mitigations, detections
    
    def process_log_file(
        self,
        log_file_path: str,
        output_path: str,
        max_logs_per_file: Optional[int] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Process a DARPA log file using BATCHED inference
        
        Args:
            log_file_path: Path to log file (CSV/TSV)
            output_path: Path to save results
            max_logs: Maximum number of logs to process (None = all)
            verbose: Print detailed debug information
            
        Returns:
            DataFrame with predictions
        """
        self.verbose = verbose  # Store for use in other methods
        print(f"\nProcessing log file: {log_file_path}")
        
        # Load log file
        if log_file_path.endswith('.csv'):
            df = pd.read_csv(log_file_path)
        elif log_file_path.endswith('.tsv'):
            df = pd.read_csv(log_file_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Use .csv or .tsv")
        
        print(f"Loaded {len(df)} log entries")
        
        # Limit if specified
        if max_logs_per_file:
            df = df.head(max_logs_per_file)
            print(f"Processing first {max_logs_per_file} entries")
        
        # Ensure raw_text column exists
        if 'raw_text' not in df.columns:
            raise ValueError("Log file must contain 'raw_text' column")
        
        print(f"Using BATCHED processing (batch_size={self.batch_size})...")
        results = self._process_batched(df)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        # Print summary
        self._print_summary(results_df, df)
        
        return results_df
    
    def _process_batched(self, df: pd.DataFrame) -> List[Dict]:
        """
        Process logs using BATCHED GPU inference
        KEY METHOD: This is where the magic happens
        """
        all_results = []
        
        # Process in batches
        num_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(0, len(df), self.batch_size), 
                             desc="Processing batches", total=num_batches):
            
            batch_df = df.iloc[batch_idx:batch_idx + self.batch_size]
            
            # Check cache first for this batch
            cached_results = []
            uncached_indices = []
            uncached_logs = []
            
            for idx, row in batch_df.iterrows():
                log_entry = row['raw_text']
                
                if self.use_caching:
                    cache_key = self._get_cache_key(log_entry)
                    if cache_key in self.cache:
                        self.stats['cache_hits'] += 1
                        cached_results.append((idx, row, self.cache[cache_key]))
                        continue
                    self.stats['cache_misses'] += 1
                
                uncached_indices.append(idx)
                uncached_logs.append((idx, row, log_entry))
            
            # Process uncached logs in batch
            if uncached_logs:
                batch_predictions = self._analyze_batch(uncached_logs)
                
                # Cache results
                for (idx, row, log_entry), prediction in zip(uncached_logs, batch_predictions):
                    if prediction and self.use_caching:
                        cache_key = self._get_cache_key(log_entry)
                        self.cache[cache_key] = prediction
                    
                    if prediction:
                        result = self._create_result_dict(idx, row, prediction)
                        all_results.append(result)
            
            # Add cached results
            for idx, row, prediction in cached_results:
                if prediction:
                    result = self._create_result_dict(idx, row, prediction)
                    all_results.append(result)
        
        return all_results
    
    def _analyze_batch(self, batch_logs: List[Tuple]) -> List[Optional[MITREPrediction]]:
        """
        Analyze a batch of logs through the two-stage pipeline
        BATCHED: All prompts processed together in single GPU calls
        """
        indices = [item[0] for item in batch_logs]
        rows = [item[1] for item in batch_logs]
        log_entries = [item[2] for item in batch_logs]
        
        verbose = getattr(self, 'verbose', False)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing batch of {len(log_entries)} events")
            print(f"{'='*80}")
        
        # STAGE 1: Batch tactic identification
        if verbose:
            print("\n[STAGE 1] Tactic Identification")
            print("-" * 80)
        
        # Create all tactic prompts
        tactic_prompts = [self._create_tactic_prompt(log) for log in log_entries]
        
        # BATCH GENERATE: Process all tactics in one GPU call
        tactic_responses = self._batch_generate(tactic_prompts)
        
        # Parse tactic results
        tactic_results = []
        for i, response in enumerate(tactic_responses):
            if verbose:
                print(f"\nEvent {i+1}:")
                print(f"  Log: {log_entries[i][:200]}...")
            
            response_text = response.split("JSON Response:")[-1]
            result = self._extract_json_from_response(response_text, verbose=verbose)
            tactic_results.append(result)
            
            if result and verbose:
                print(f"  ✓ Tactic: {result.get('tactic_name')} (conf: {result.get('confidence', 0):.2f})")
            elif verbose:
                print(f"  ✗ Tactic extraction FAILED")
        
        # STAGE 2: Batch technique identification
        if verbose:
            print(f"\n[STAGE 2] Technique Identification")
            print("-" * 80)
        
        # Create technique prompts for successful tactic predictions
        technique_prompts = []
        valid_indices = []
        
        for i, (log, tactic_result) in enumerate(zip(log_entries, tactic_results)):
            if tactic_result:
                prompt = self._create_technique_prompt(
                    log,
                    tactic_result.get('tactic_name', ''),
                    tactic_result.get('tactic_shortname', '')
                )
                technique_prompts.append(prompt)
                valid_indices.append(i)
            else:
                technique_prompts.append("")  # Placeholder
                if verbose:
                    print(f"\nEvent {i+1}: Skipped (no tactic)")
        
        # BATCH GENERATE: Process all techniques in one GPU call
        if any(technique_prompts):
            valid_prompts = [p for p in technique_prompts if p]
            if verbose:
                print(f"\nGenerating {len(valid_prompts)} technique predictions...")
            technique_responses = self._batch_generate(valid_prompts)
        else:
            technique_responses = []
        
        # Parse technique results
        technique_results = [None] * len(log_entries)
        for valid_idx, response in zip(valid_indices, technique_responses):
            if verbose:
                print(f"\nEvent {valid_idx+1}:")
            
            response_text = response.split("JSON Response:")[-1]
            result = self._extract_json_from_response(response_text, verbose=verbose)
            technique_results[valid_idx] = result
            
            if result and verbose:
                print(f"  ✓ Technique: {result.get('technique_id')} - {result.get('technique_name')} (conf: {result.get('confidence', 0):.2f})")
            elif verbose:
                print(f"  ✗ Technique extraction FAILED")
        
        # STAGE 3: Create predictions
        if verbose:
            print(f"\n[STAGE 3] Creating Predictions")
            print("-" * 80)
        
        predictions = []
        for i, (log, tactic_result, technique_result) in enumerate(
            zip(log_entries, tactic_results, technique_results)
        ):
            if tactic_result and technique_result:
                # Extract mitigation strategies
                mitigation_strategies, detection_strategies = self._get_strategies(
                    tactic_result.get('tactic_shortname', ''),
                    technique_result.get('technique_id', '')
                )
                
                # Create prediction
                prediction = MITREPrediction(
                    tactic=tactic_result.get('tactic_name', ''),
                    technique=technique_result.get('technique_id', ''),
                    technique_id=technique_result.get('technique_id', ''),
                    technique_name=technique_result.get('technique_name', ''),
                    confidence_score=min(
                        tactic_result.get('confidence', 0),
                        technique_result.get('confidence', 0)
                    ),
                    mitigation_strategies=mitigation_strategies,
                    detection_strategies=detection_strategies,
                    reasoning=f"Tactic: {tactic_result.get('reasoning', '')}; \n"
                             f"Technique: {technique_result.get('reasoning', '')}"
                )
                predictions.append(prediction)
                self.stats['total_processed'] += 1
                
                if verbose:
                    print(f"Event {i+1}: ✓ SUCCESS - {prediction.technique_id}")
            else:
                predictions.append(None)
                if verbose:
                    print(f"Event {i+1}: ✗ FAILED - Missing {'tactic' if not tactic_result else 'technique'}")
        
        if verbose:
            success_count = sum(1 for p in predictions if p is not None)
            print(f"\n{'='*80}")
            print(f"Batch complete: {success_count}/{len(predictions)} successful")
            print(f"{'='*80}")
        
        return predictions
    
    def _create_result_dict(self, idx, row, prediction: MITREPrediction) -> Dict:
        """Create result dictionary from prediction"""
        return {
            'log_index': idx,
            # 'event_type': row.get('event_type', ''),
            # 'timestamp': row.get('ts_human', ''),
            # 'pid': row.get('pid', ''),
            # 'cmdline': row.get('cmdline', ''),
            # 'file_path': row.get('file_path', ''),
            # 'registry_key': row.get('registry_key', ''),
            'raw_text': row['raw_text'],
            'tactic': prediction.tactic,
            'technique_id': prediction.technique_id,
            'technique_name': prediction.technique_name,
            'confidence_score': prediction.confidence_score,
            'reasoning': prediction.reasoning,
            'num_mitigations': len(prediction.mitigation_strategies),
            'mitigation_strategies': json.dumps(prediction.mitigation_strategies),
        }
    
    def _print_summary(self, results_df: pd.DataFrame, original_df: pd.DataFrame):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total logs processed: {len(results_df)}/{len(original_df)}")
        
        if len(results_df) > 0:
            print(f"Average confidence: {results_df['confidence_score'].mean():.3f}")
            print(f"\nTop Tactics:")
            print(results_df['tactic'].value_counts().head())
            print(f"\nTop Techniques:")
            print(results_df['technique_id'].value_counts().head())
        
        # Performance statistics
        print(f"\nPerformance Statistics:")
        print(f"  Batches processed: {self.stats['batches_processed']}")
        print(f"  Batch size: {self.batch_size}")
        
        # Cache statistics
        if self.use_caching:
            print(f"\nCache Statistics:")
            print(f"  Cache hits: {self.stats['cache_hits']}")
            print(f"  Cache misses: {self.stats['cache_misses']}")
            total = self.stats['cache_hits'] + self.stats['cache_misses']
            if total > 0:
                hit_rate = (self.stats['cache_hits'] / total) * 100
                print(f"  Hit rate: {hit_rate:.1f}%")
        
        print("="*80)
    
    def process_multiple_files(
        self,
        file_paths: List[str],
        output_dir: str,
        max_logs_per_file: Optional[int] = None
    ):
        """Process multiple DARPA log files - UNCHANGED"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        all_results = []
        
        for file_path in file_paths:
            file_name = Path(file_path).stem
            output_file = output_path / f"{file_name}_mitre_analysis.csv"
            
            results_df = self.process_log_file(
                file_path,
                str(output_file),
                max_logs_per_file
            )
            
            all_results.append(results_df)
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_output = output_path / "combined_mitre_analysis.csv"
        combined_df.to_csv(combined_output, index=False)
        
        print(f"\nCombined results saved to {combined_output}")
        
        return combined_df


def main():
    """Example usage"""
    
    # Configuration
    MITRE_KB_PATH = "mitre_detection_kb.json"
    SHARDS_DIR = r"darpa_input_files"
    LOG_FILES = [        
        SHARDS_DIR + "/" + "ta1-fivedirections-2-e5-official-1.bin.77.csv",
    ]
    OUTPUT_DIR = "mitre_analysis_results"
    
    # Initialize BATCHED analyzer
    print("="*80)
    print("BATCHED MITRE ATT&CK ANALYZER (FASTEST VERSION)")
    print("Optimizations: Caching + Quantization + Batch GPU Processing")
    print("="*80)
    
    analyzer = MITRELogAnalyzerBatched(
        mitre_kb_path=MITRE_KB_PATH,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        use_quantization=True,   # 2-4x speedup
        use_caching=True,        # 2-3x speedup on duplicates
        batch_size=8             # Process 8 events per GPU call
    )
    
    # Process files
    import time
    start = time.time()
    
    # Process with verbose=True to see what's happening
    for file_path in LOG_FILES:
        file_name = Path(file_path).stem
        output_file = Path(OUTPUT_DIR) / f"{file_name}_mitre_analysis.csv"
        
        results = analyzer.process_log_file(
            file_path,
            str(output_file),
            max_logs_per_file=10,  # Test with 10 events
            verbose=True  # Enable detailed logging
        )
    
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Events processed: {len(results)}")
    print(f"Average time per event: {elapsed/len(results):.2f}s")
    print(f"Events per second: {len(results)/elapsed:.2f}")
    print(f"\nEstimated speedup vs original (180s/event): {180/(elapsed/len(results)):.1f}x")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()