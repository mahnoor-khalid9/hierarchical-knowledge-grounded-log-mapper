# 🛡️ HKLM — Hierarchical Knowledge-grounded Log Mapper

**A detection framework that maps raw system logs to MITRE ATT&CK tactics and techniques using open-source LLMs.**

HKLM uses a hierarchical, knowledge-constrained approach — the LLM reasons over bounded options from the ATT&CK knowledge base instead of recalling from training data, reducing hallucination and improving classification quality.

> **CIS 544: Cyber Defense and Operations** — Spring 2026  
> Minal Ali • Fnu Mahnoor  
> University of Massachusetts Dartmouth  

---

## 🎯 What HKLM Does

Given any raw log event — from any source, any format — HKLM produces:

| Output | Description |
|--------|-------------|
| **Tactic** | Which of the 14 ATT&CK tactics this event maps to (e.g., Initial Access, Execution) |
| **Technique** | The specific T-number and name (e.g., T1133 — External Remote Services) |
| **Confidence** | Score from 0.0 to 1.0 for both tactic and technique |
| **Reasoning** | Natural language explanation of why the model chose this classification |
| **Mitigations** | Defensive strategies retrieved from the ATT&CK knowledge base |

**Example:**

```
Input:  type=EVENT_CONNECT | pid=8428 | cmd=ssh admin@128.55.12.56

Output:
  Tactic:      Initial Access (confidence: 0.20)
  Technique:   T1133 — External Remote Services (confidence: 0.80)
  Reasoning:   "SSH connection attempt from external IP aligns with External Remote Services"
  Mitigations: Network Segmentation, MFA, Disable Remote Services, Limit Network Access
```

---

## 🏗️ Architecture

HKLM uses a **three-stage hierarchical pipeline**:

**Stage 1 — Tactic Classification (LLM)**
Raw log + all 14 tactic descriptions sent to LLM. Model selects best-matching tactic from bounded list (not from memory).

**Stage 2 — Technique Classification (LLM)**
Raw log + predicted tactic + only techniques under that tactic (~10–30 options) sent to LLM. Hierarchical narrowing reduces decision space.

**Stage 3 — Strategy Retrieval (Deterministic)**
Given predicted (tactic, technique), retrieve mitigation and detection strategies directly from MITRE ATT&CK knowledge base. No generation, zero hallucination.

### Key Design: Constrained Reasoning

LLM acts as a **reasoner**, not a knowledge store. Official definitions provided; model reasons over bounded options. This eliminates hallucinated technique IDs.

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| GPU VRAM | 8GB | 12GB |
| System RAM | 8GB | 16GB |
| Storage | 5GB free | 10GB free |
| OS | Windows, Linux, macOS | Ubuntu 20.04+ |
| Python | 3.9+ | 3.11+ |
| GPU | NVIDIA | RTX 4000Ada, RTX 3080+ |

**Note:** CPU-only mode is not practical (100+ seconds per log). GPU is required.

---

## ⚡ Quick Start

### Step 1: Clone Repository

```bash
git clone https://github.com/minaali/hierarchical-knowledge-grounded-log-mapper.git
cd hierarchical-knowledge-grounded-log-mapper
```

### Step 2: Create Virtual Environment

**Using venv (default):**
```bash
python3.11 -m venv logs-mapper
```

**Using conda (alternative):**
```bash
conda create -n logs-mapper python=3.11
```

### Step 3: Activate Virtual Environment

**venv on Windows:**
```bash
logs-mapper\Scripts\activate.bat
```

**venv on Linux/macOS:**
```bash
source logs-mapper/bin/activate
```

**conda (all platforms):**
```bash
conda activate logs-mapper
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch with CUDA support and all dependencies. First installation takes 10-20 minutes (downloading model files).

### Step 5: Verify GPU Access

```bash
python3.11 -c "import torch; print(torch.cuda.is_available())"
```

Should output: `True`

If `False`: Update NVIDIA drivers or reinstall PyTorch.

### Step 6: Run the Application

```bash
python3.11 gradio_app.py
```

Opens at `http://localhost:7860`

---

## 📖 How to Use

### Web Interface (Running Locally)

**1. Input Your Logs**

Two ways to input logs:

- **Paste Logs tab:** Paste raw log events directly (one per line)
- **Upload CSV tab:** Upload CSV file with logs

Example log formats:
```
type=EVENT_READ | pid=8668 | path=\REGISTRY\MACHINE\SAM
type=EVENT_CONNECT | pid=8428 | cmd=ssh admin@128.55.12.56
type=EVENT_MODIFY | pid=7980 | cmd=scp -r C:\Users\admin\Documents
```

**2. Configure Settings (Optional)**

Left panel shows:
- **Model:** Choose between Mistral 7B, Qwen 7B, Phi-3.5
- **Batch Size:** Higher = faster but more VRAM (4 recommended for 12GB)
- **Max Events:** Leave empty to process all events
- **Checkboxes:**
  - ✅ 4-bit Quantization (saves VRAM, faster)
  - ✅ Semantic Caching (skips duplicates, faster)
  - ✅ Verbose Logging (shows per-event details)

**3. Click the RUN Button**

Click the blue **RUN** button to start analysis.

**4. Monitor Analysis**

Three tabs show results:

- **Live Logs tab:** Real-time analysis progress
  - Shows each log being processed
  - Stage 1: Tactic identification
  - Stage 2: Technique identification
  - Stage 3: Mitigation retrieval
  - Updates every 0.5 seconds

- **Results Table tab:** Structured results
  - Log index, raw text, tactic, technique, confidence scores
  - Downloadable as CSV
  - Search and filter capabilities

- **Statistics tab:** Analysis summary
  - Chart of detected tactics
  - Chart of detected techniques
  - Distribution graphs
  - Overall statistics

**Example Output:**

```
Log 5: type=EVENT_READ | pid=4364 | cmd="C:\Program Files\TightVNC\tvnserver.exe"
Stage 1: Tactic = Persistence (confidence: 0.89)
Stage 2: Technique = T1547 (Boot or Logon Autostart Execution) (confidence: 0.92)
Stage 3: Mitigations = Disable autostart features, Restrict registry access...
```

### Command Line (Batch Processing)

```python
from mitre_analyzer_batched import MITRELogAnalyzerBatched

analyzer = MITRELogAnalyzerBatched(
    mitre_kb_path="mitre_detection_kb.json",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    use_quantization=True,
    batch_size=4
)

results = analyzer.analyze("your_logs.csv", max_logs=1000)
results.to_csv("hklm_results.csv", index=False)
```

---

## 📂 Project Structure

```
├── gradio_app.py              # Web interface
├── mitre_analyzer_batched.py  # Core analysis engine
├── mitre_detection_kb.json    # MITRE knowledge base
├── requirements.txt           # Dependencies (GPU enabled)
└── README.md
```

---

## 🤖 Supported Models

All models run locally. Change in `gradio_app.py`:

```python
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Change this line
```

| Model | VRAM (4-bit) | Speed | JSON Output |
|-------|---|---|---|
| **Mistral 7B** | **3.5GB** | **🔄 Moderate** | **✅ Reliable** |
| Qwen 2.5 7B | 3.5GB | 🔄 Moderate | ✅ Reliable |
| Phi-3.5 3.8B | 2GB | ⚡ Fast | ✅ Good |

⚠️ **Note:** 1.5B and 3B models cannot reliably produce JSON output. Not supported.

**Default:** Mistral 7B (best reasoning + JSON compliance)

---

## ⚡ Performance Optimization

All optimizations enabled by default:

| Feature | Benefit |
|---------|---------|
| **4-bit Quantization** | 8x compression, 1% accuracy loss, 2-4x faster |
| **Batched Inference** | Process 4 logs simultaneously, 2.5-3x throughput |
| **Semantic Caching** | Skip 30-40% duplicate logs (MD5 hash lookup) |
| **Hierarchical Narrowing** | Tactic (14) → Technique (10-30), better accuracy |

**Performance on RTX 4000Ada 12GB:**
- Batch Size: 4
- Speed: 6-8 logs/second
- First run: 2-3 minutes (model download)
- Subsequent runs: 5-10 seconds startup

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **GPU not detected** | Update NVIDIA drivers. Run `nvidia-smi`. Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| **Virtual env not activating (Windows)** | Use: `logs-mapper\Scripts\activate.bat` |
| **Virtual env not activating (Linux/Mac)** | Use: `source logs-mapper/bin/activate` |
| **Out of memory** | Reduce batch_size in gradio_app.py: `batch_size = 2` |
| **Port 7860 in use** | Use different port: `python3.11 gradio_app.py --port 7861` |
| **Module not found errors** | Reinstall dependencies: `pip install -r requirements-gpu.txt` |
| **Slow processing (1-2 logs/sec)** | Check GPU usage: `nvidia-smi`. Verify quantization enabled. |

---

## 📊 Dataset

Demonstrated on **DARPA Transparent Computing Engagement 5 (TC E5)** — Five Directions provenance logs.

- **265,189 events** per log file
- Mix of scripted benign activity and real APT attack scenarios
- Windows OS-level provenance: file I/O, registry access, process creation, network connections
- Publicly available from DARPA

> **Note:** HKLM is log-source agnostic. Works with any raw log format — Windows Event Logs, syslog, firewall logs, cloud audit logs, NIDS alerts.

---

## 🔬 Related Work

| Approach | Input | Limitation |
|----------|-------|-----------|
| **TRAM** (MITRE CTID, 2024) | CTI prose | Human text, not raw logs. 50 techniques only. Requires labeled data. |
| **RHINO** (2025) | NIDS alerts | Pre-filtered alerts, not OS-level events. Cloud API dependent. |
| **OntoLogX** (2025) | Raw logs | Full KG construction. Tactic-level only, no technique mapping. |

**Gap:** No prior work applies open-source LLMs to raw OS-level provenance logs for hierarchical ATT&CK classification (tactic → technique) using knowledge-base-constrained zero-shot prompting.

---

## 📄 License

This project is developed for academic purposes as part of CIS 544-01: Cyber Defense and Operations at UMass Dartmouth.

---

## 🔗 References

- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [DARPA Transparent Computing](https://www.darpa.mil/program/transparent-computing)
- [DARPA TC E5 Data](https://github.com/darpa-i2o/Transparent-Computing)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Qwen 2.5](https://huggingface.co/Qwen)

---

## ❓ FAQ

**Q: Can I use Windows?**
A: Yes, fully supported.

**Q: Do I need 16GB system RAM?**
A: No, 8GB minimum works fine.

**Q: Can I run without GPU?**
A: Technically yes, but impractical (100+ seconds per log).

**Q: How do I stop the app?**
A: Press Ctrl+C in terminal.

**Q: Is my data private?**
A: Yes, all processing is local. No cloud transmission.

**Q: Can I use AMD/Intel GPUs?**
A: NVIDIA only. AMD/Intel not supported.

**Q: How long is first run?**
A: 2-3 minutes (model download). Subsequent runs: 5-10 seconds.

**Q: Can I change the model?**
A: Yes, edit `model_name` in `gradio_app.py`.