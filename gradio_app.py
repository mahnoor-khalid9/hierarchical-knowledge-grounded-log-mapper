#!/usr/bin/env python3
"""
HKLM - Hierarchical Knowledge-grounded Log Mapper
Gradio Web Interface with LIVE LOGGING

Features:
  - Upload CSV file OR paste logs directly into a text field
  - Threaded GPU inference with live log streaming
  - Deployable on HuggingFace Spaces, Google Colab, or local GPU
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import json
import time
import threading
from datetime import datetime
import sys
import io
from mitre_analyzer_batched import MITRELogAnalyzerBatched


class LiveLogger:
    """Thread-safe logger that captures print statements and streams them to Gradio"""

    def __init__(self):
        self.logs = []
        self.original_stdout = sys.stdout
        self._lock = threading.Lock()

    def write(self, text):
        if text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            with self._lock:
                self.logs.append(f"[{timestamp}] {text}")
        self.original_stdout.write(text)
        self.original_stdout.flush()

    def flush(self):
        self.original_stdout.flush()

    def isatty(self):
        return False

    def get_logs(self):
        with self._lock:
            return "\n".join(self.logs)

    def clear(self):
        with self._lock:
            self.logs = []


class GradioMITREAnalyzer:
    """Wrapper for MITRE analyzer with Gradio UI and live logging"""

    def __init__(self):
        self.analyzer = None
        self.results_df = None
        self.logger = None

    def _build_dataframe_from_text(self, raw_text):
        """Convert pasted log text into a DataFrame with raw_text column.
        
        Supports:
          - One log event per line (most common)
          - Pasted CSV with headers (auto-detected)
        """
        lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
        if not lines:
            return None

        # Check if first line looks like a CSV header containing 'raw_text'
        if "raw_text" in lines[0].lower() and "," in lines[0]:
            try:
                df = pd.read_csv(io.StringIO(raw_text))
                if "raw_text" in df.columns:
                    return df
            except Exception:
                pass

        # Otherwise treat each line as a separate log event
        return pd.DataFrame({"raw_text": lines})

    def analyze(
        self,
        file_path,
        text_input,
        model_name,
        batch_size,
        max_logs,
        use_quantization,
        use_caching,
        verbose,
        progress=gr.Progress()
    ):
        """Analyze logs from either file upload or text input with live streaming"""

        # ── Determine input source ──
        df = None
        input_source = None

        if text_input and text_input.strip():
            df = self._build_dataframe_from_text(text_input)
            input_source = "text"
            if df is None or len(df) == 0:
                yield None, "⚠️ Could not parse any log events from text input!", "", ""
                return
        elif file_path is not None:
            try:
                df = pd.read_csv(file_path)
                input_source = "file"
            except Exception as e:
                yield None, f"⚠️ Error reading CSV: {e}", "", ""
                return
        else:
            yield None, "⚠️ Please upload a CSV file or paste logs in the text field!", "", ""
            return

        # ── Validate ──
        if "raw_text" not in df.columns:
            # If only one column, assume it's the log text
            if len(df.columns) == 1:
                df.columns = ["raw_text"]
            else:
                yield None, "⚠️ CSV must contain a 'raw_text' column!", "", ""
                return

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            self.logger = LiveLogger()
            sys.stdout = self.logger
            sys.stderr = self.logger

            yield None, "🚀 Starting analysis...", "", self.logger.get_logs()

            # ── MODEL LOADING (threaded) ──
            progress(0.0, desc="🔧 Initializing model...")
            print("\n" + "=" * 80)
            print("🤖 HKLM — MITRE ATT&CK LOG MAPPER")
            print("=" * 80)
            print(f"Model: {model_name}")
            print(f"Batch size: {batch_size}")
            print(f"Quantization: {'Enabled (4-bit)' if use_quantization else 'Disabled'}")
            print(f"Caching: {'Enabled' if use_caching else 'Disabled'}")
            print(f"Input source: {'Text field' if input_source == 'text' else 'CSV file'}")
            print("=" * 80)

            yield None, "🔧 Loading model...", "", self.logger.get_logs()

            model_container = {"done": False, "error": None}

            def _load_model():
                try:
                    self.analyzer = MITRELogAnalyzerBatched(
                        mitre_kb_path="mitre_detection_kb.json",
                        model_name=model_name,
                        use_quantization=use_quantization,
                        use_caching=use_caching,
                        batch_size=batch_size,
                    )
                except Exception as e:
                    model_container["error"] = e
                finally:
                    model_container["done"] = True

            thread = threading.Thread(target=_load_model, daemon=True)
            thread.start()

            while not model_container["done"]:
                time.sleep(0.5)
                yield None, "🔧 Loading model...", "", self.logger.get_logs()

            thread.join()

            if model_container["error"]:
                raise model_container["error"]

            if verbose:
                self.analyzer.verbose = True
                print("🔍 Verbose mode enabled")

            print("✅ Model loaded successfully!\n")
            yield None, "✅ Model ready!", "", self.logger.get_logs()

            # ── DATA LOADING ──
            progress(0.15, desc="📂 Loading log data...")
            total_logs = len(df)
            print("\n" + "=" * 80)
            print("📂 LOADING LOG DATA")
            print("=" * 80)
            print(f"Input source: {'Pasted text' if input_source == 'text' else 'CSV file'}")
            print(f"Total events: {total_logs:,}")

            if max_logs and max_logs > 0:
                df = df.head(int(max_logs))
                print(f"Processing first {len(df):,} events (max_logs setting)")

            print(f"✅ Ready to process {len(df):,} events")
            print("=" * 80)

            yield None, f"📊 Loaded {len(df):,} events", "", self.logger.get_logs()

            # ── BATCH PROCESSING (threaded) ──
            progress(0.2, desc="🔄 Starting batch processing...")
            batch_size_int = int(batch_size)
            num_events = len(df)
            num_batches = (num_events + batch_size_int - 1) // batch_size_int

            print("\n" + "=" * 80)
            print("🔄 BATCH PROCESSING STARTED (GPU-Accelerated)")
            print("=" * 80)
            print(f"Total events: {num_events:,}")
            print(f"Batch size: {batch_size_int}")
            print(f"Number of batches: {num_batches}")
            print("=" * 80)

            all_results = []
            start_time = time.time()

            for batch_num in range(num_batches):
                batch_start = batch_num * batch_size_int
                batch_end = min(batch_start + batch_size_int, num_events)
                batch_df = df.iloc[batch_start:batch_end]

                batch_progress = 0.2 + (0.7 * (batch_num / num_batches))
                progress(batch_progress, desc=f"⚡ Batch {batch_num + 1}/{num_batches}")

                print(f"\n{'─' * 80}")
                print(f"📦 BATCH {batch_num + 1}/{num_batches}")
                print(f"{'─' * 80}")
                print(f"Events: {batch_start + 1} to {batch_end} ({len(batch_df)} events)")

                if all_results:
                    current_df = pd.DataFrame(all_results)
                    stats = self._format_statistics(current_df, len(all_results), num_events, time.time() - start_time)
                    yield current_df, f"⚡ Batch {batch_num + 1}/{num_batches}...", stats, self.logger.get_logs()
                else:
                    yield None, f"⚡ Batch {batch_num + 1}/{num_batches}...", "", self.logger.get_logs()

                # Run batch in background thread for live log streaming
                batch_container = {"results": None, "done": False, "error": None}

                def _run_batch(data=batch_df):
                    try:
                        batch_container["results"] = self.analyzer._process_batched(data)
                    except Exception as e:
                        batch_container["error"] = e
                    finally:
                        batch_container["done"] = True

                batch_thread = threading.Thread(target=_run_batch, daemon=True)
                batch_thread.start()

                while not batch_container["done"]:
                    time.sleep(0.5)
                    elapsed = time.time() - start_time
                    if all_results:
                        current_df = pd.DataFrame(all_results)
                        stats = self._format_statistics(current_df, len(all_results), num_events, elapsed)
                        yield current_df, f"⚡ Batch {batch_num + 1}/{num_batches}...", stats, self.logger.get_logs()
                    else:
                        yield None, f"⚡ Batch {batch_num + 1}/{num_batches}...", "", self.logger.get_logs()

                batch_thread.join()

                if batch_container["error"]:
                    raise batch_container["error"]

                all_results.extend(batch_container["results"])

                elapsed = time.time() - start_time
                rate = len(all_results) / elapsed if elapsed > 0 else 0
                print(f"\n📊 Batch {batch_num + 1} done — {len(all_results):,}/{num_events:,} processed ({rate:.1f} events/sec)")

                current_df = pd.DataFrame(all_results)
                stats = self._format_statistics(current_df, len(all_results), num_events, elapsed)
                yield current_df, f"✅ Batch {batch_num + 1}/{num_batches}", stats, self.logger.get_logs()

            # ── FINALIZATION ──
            progress(0.95, desc="💾 Saving results...")
            final_df = pd.DataFrame(all_results)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("gradio_results")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"hklm_results_{timestamp}.csv"
            final_df.to_csv(output_path, index=False)

            total_time = time.time() - start_time
            final_stats = self._format_statistics(final_df, len(final_df), num_events, total_time)

            print("\n" + "=" * 80)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 80)
            print(f"Total events: {len(final_df):,}")
            print(f"Time: {total_time:.1f}s ({len(final_df)/total_time:.1f} events/sec)")
            print(f"Results saved: {output_path}")
            print("=" * 80)

            yield final_df, f"✅ Complete! {len(final_df):,} events in {total_time:.1f}s", final_stats, self.logger.get_logs()

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            print(traceback.format_exc())
            yield None, f"❌ Error: {e}", "", self.logger.get_logs()

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _format_statistics(self, df, processed, total, elapsed):
        stats = []
        stats.append("=" * 50)
        stats.append("📊 STATISTICS")
        stats.append("=" * 50)
        pct = (processed / total * 100) if total > 0 else 0
        stats.append(f"Progress: {processed:,} / {total:,} ({pct:.1f}%)")
        rate = processed / elapsed if elapsed > 0 else 0
        stats.append(f"Rate: {rate:.1f} events/sec")
        stats.append(f"Time: {elapsed:.1f}s")

        for col in ["tactic", "predicted_tactic"]:
            if col in df.columns:
                stats.append(f"\n🎯 Tactic Distribution:")
                for tactic, count in df[col].value_counts().head(5).items():
                    stats.append(f"  • {tactic}: {count} ({count/len(df)*100:.1f}%)")
                break

        for col in ["confidence_score", "confidence"]:
            if col in df.columns:
                stats.append(f"\n📈 Avg Confidence: {df[col].mean():.1%}")
                break

        stats.append("=" * 50)
        return "\n".join(stats)


# ─────────────────────────────────────────────────
# SAMPLE LOGS for the text input placeholder
# ─────────────────────────────────────────────────


# ── Sample logs ──
SAMPLE_LOGS = r"""type=EVENT_CONNECT | pid=8428 | cmd=ssh admin@128.55.12.56
type=EVENT_READ | pid=3488 | cmd=scp -r C:\Users\admin\Documents admin@128.55.12.106:./files/ | path=\REGISTRY\MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\SideBySide\
type=EVENT_READ | pid=7980 | cmd="C:\Program Files\OpenSSH-Win64\sshd.exe" | path=\REGISTRY\MACHINE\SYSTEM\ControlSet001\Services\Tcpip\Parameters\Winsock\
type=EVENT_CONNECT | pid=9448 | cmd="C:\Program Files\OpenSSH-Win64\ssh.exe" "-x" "-oForwardAgent=no" "-oPermitLocalCommand=no" "-oClearAllForwardings=yes" 
type=EVENT_READ | pid=4364 | cmd="C:\Program Files\TightVNC\tvnserver.exe" -desktopserver -logdir "C:\WINDOWS\system32\config\systemprofile\AppData\Roami | path=\REGISTRY\MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\SideBySide\
type=EVENT_SENDTO | bytes=68
type=EVENT_MODIFY_FILE_ATTRIBUTES | pid=7104 | cmd="C:\Program Files\OpenSSH-Win64\ssh.exe" "-x" "-oForwardAgent=no" "-oPermitLocalCommand=no" "-oClearAllForwardings=yes"  | path=\REGISTRY\MACHINE\SYSTEM\ControlSet001\Services\WinSock2\Parameters\NameSpace_Catalog5\
type=EVENT_READ | path=\REGISTRY\MACHINE\SYSTEM\ControlSet001\Services\WinSock2\Parameters\NameSpace_Catalog5\Catalog_Entries64\000000000001\
type=EVENT_READ | path=\REGISTRY\MACHINE\SAM\SAM\Domains\Account\Users\Names\admin\
type=EVENT_MODIFY_FILE_ATTRIBUTES | pid=3784 | cmd=scp  -r C:\Users\admin\Documents admin@128.55.12.51:./test/ | path=\REGISTRY\MACHINE\SYSTEM\ControlSet001\Services\WinSock2\Parameters\Protocol_Catalog9\
type=EVENT_READ | pid=3784 | cmd=scp  -r C:\Users\admin\Documents admin@128.55.12.51:./test/ | path=\REGISTRY\MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones\Eastern Standard Time\Dynamic DST\
type=EVENT_READ | pid=3784 | cmd=scp  -r C:\Users\admin\Documents admin@128.55.12.51:./test/ | path=\REGISTRY\MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\GRE_Initialize\
type=EVENT_READ | pid=8668 | cmd=C:\WINDOWS\system32\cmd.exe | path=\REGISTRY\USER\S-1-5-21-231540947-922634896-4161786520-1004\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders\
type=EVENT_READ | pid=8736 | cmd=C:\WINDOWS\system32\cmd.exe | path=\REGISTRY\USER\S-1-5-21-231540947-922634896-4161786520-1004\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders\
type=EVENT_READ | pid=8460 | cmd=C:\WINDOWS\system32\cmd.exe | path=\REGISTRY\USER\S-1-5-21-231540947-922634896-4161786520-1004\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders\
""".strip()


def create_interface():
    """Create the Gradio interface"""

    analyzer = GradioMITREAnalyzer()

    with gr.Blocks(
        title="HKLM — MITRE ATT&CK Log Mapper",
        theme=gr.themes.Soft(),
    ) as interface:

        gr.Markdown(
            "# 🛡️ HKLM — Hierarchical Knowledge-grounded Log Mapper\n"
            "Map any raw system log to MITRE ATT&CK tactics and techniques using open-source LLMs"
        )

        with gr.Row():
            # ── LEFT: Input & Settings ──
            with gr.Column(scale=1):

                gr.Markdown("### 📂 Log Input")
                with gr.Tabs() as input_tabs:
                    with gr.Tab("📝 Paste Logs"):
                        text_input = gr.Textbox(
                            label="Paste log events (one per line)",
                            lines=8,
                            max_lines=20,
                            value=SAMPLE_LOGS,
                        )
                        load_sample_btn = gr.Button("📋 Load Sample Logs", size="sm")
                        
                   
                    with gr.Tab("📁 Upload CSV", id="file_tab"):
                        file_input = gr.File(label="Upload CSV with 'raw_text' column", file_types=[".csv", ".tsv"])

                gr.Markdown("### ⚙️ Settings")
                model_choice = gr.Dropdown(
                    choices=[
                        "microsoft/Phi-3.5-mini-instruct",
                        "Qwen/Qwen2.5-3B-Instruct",
                        "Qwen/Qwen2.5-1.5B-Instruct",
                        "mistralai/Mistral-7B-Instruct-v0.2",
                        "Qwen/Qwen2.5-7B-Instruct",
                    ],
                    value="mistralai/Mistral-7B-Instruct-v0.2",
                    label="🤖 Model",
                    info="Phi-3.5: Fast (3.8B) | Mistral: Balanced (7B) | Qwen 7B: Best accuracy",
                )
                batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size",
                                      info="Events per GPU call (higher = faster, more VRAM)")
                max_logs = gr.Number(value=None, label="Max Events",
                                    info="Leave empty to process all events")
                use_quantization = gr.Checkbox(value=True, label="4-bit Quantization",
                                              info="2–4x faster, 4x less VRAM")
                use_caching = gr.Checkbox(value=True, label="Semantic Caching",
                                         info="Skip duplicate events via MD5 hash")
                verbose = gr.Checkbox(value=True, label="Verbose Logging",
                                     info="Show per-event model outputs")

            # ── RIGHT: Outputs ──
            with gr.Column(scale=2):
                # Results header with RUN button
                with gr.Row(variant="panel"):
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Results")
                    with gr.Column(scale=0, min_width=120):
                        analyze_btn = gr.Button(
                            "🚀 RUN",
                            variant="primary",
                            size="lg"
                        )

                # gr.Markdown("### 📊 Results")
                status_box = gr.Textbox(label="Status", value="Ready — paste logs or upload a CSV to begin",
                                        max_lines=1)

                with gr.Tabs():
                    with gr.Tab("📝 Live Logs"):
                        logs_box = gr.Textbox(
                            label="Processing Logs (Live Stream)",
                            lines=25, max_lines=50, autoscroll=True,
                        )
                    with gr.Tab("📋 Results Table"):
                        results_table = gr.Dataframe(
                            label="Analysis Results",
                            headers=["raw_text", "tactic", "technique_id", "technique_name", "confidence_score"],
                            max_height=400,
                        )
                    with gr.Tab("📊 Statistics"):
                        stats_box = gr.Textbox(label="Statistics", lines=20, max_lines=30)

        # ── Wire up buttons ──
        load_sample_btn.click(fn=lambda: SAMPLE_LOGS, outputs=[text_input])

        analyze_btn.click(
            fn=analyzer.analyze,
            inputs=[
                file_input, text_input, model_choice, batch_size,
                max_logs, use_quantization, use_caching, verbose,
            ],
            outputs=[results_table, status_box, stats_box, logs_box],
        )

        gr.Markdown("""
        ### 📖 How to Use
        1. **Paste logs** directly into the text field (one event per line) **or upload a CSV** with a `raw_text` column
        2. Select a model and adjust settings
        3. Click **Run Analysis** — watch live progress in the Live Logs tab
        4. View results in the Results Table tab

        ### 🔑 Key Concepts
        - **Log-source agnostic:** works with any raw text log — Windows events, syslog, firewall, cloud, etc.
        - **Post-analysis framework:** processes collected logs in batch, not real-time
        - **Knowledge-constrained:** LLM picks from ATT&CK KB options — doesn't hallucinate from memory

        ### 🚀 Deployment
        This app runs on any machine with a CUDA GPU. Deploy via:
        - **Local:** `python gradio_app.py`
        - **Google Colab:** run in notebook with `interface.launch(share=True)` for public URL
        - **HuggingFace Spaces:** push to a Space with GPU runtime
        """)

    return interface


if __name__ == "__main__":
    print("🚀 Starting HKLM — MITRE ATT&CK Log Mapper...")
    print("=" * 80)

    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",   # accessible from network (needed for Colab/HF)
        server_port=8000,
        share=False,             # set True for public URL in Colab
        show_error=True,
        inbrowser=True,
    )
