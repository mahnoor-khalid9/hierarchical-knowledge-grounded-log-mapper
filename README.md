# 🛡️ HKLM — Hierarchical Knowledge-grounded Log Mapper

**A detection framework that maps raw system logs to MITRE ATT&CK tactics and techniques using open-source LLMs.**

HKLM uses a hierarchical, knowledge-constrained approach — the LLM reasons over bounded options from the ATT&CK knowledge base instead of recalling from training data, reducing hallucination and improving classification quality.

> **CIS 544-01: Cyber Defense and Operations** — Spring 2026
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

![HKLM Architecture](images\architecture_diagram.png)

### Three-Stage Pipeline

**Stage 1 — Tactic Classification (LLM)**
The raw log event + all 14 tactic descriptions from the KB are sent to the LLM. The model picks the best-matching tactic from this bounded list — not from memory.

**Stage 2 — Technique Classification (LLM)**
The same log + predicted tactic + only the techniques under that tactic (~10–30 options, not 200+) are sent to the LLM. Hierarchical narrowing reduces the decision space.

**Stage 3 — Strategy Retrieval (No LLM)**
Deterministic KB lookup. Given the predicted (tactic, technique), retrieve mitigation and detection strategies directly from the MITRE ATT&CK knowledge base. No generation, no hallucination.

### Key Design Decision: Constrained Reasoning

We don't ask the LLM to recall ATT&CK from its training data. We give it the official definitions and ask it to **reason** over bounded options. The LLM acts as a reasoner, not a knowledge store.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (8+ GB VRAM recommended)
- ~4 GB disk space for model weights

### Installation

```bash
git clone https://github.com/mahnoor-khalid9/hierarchical-knowledge-grounded-log-mapper.git
cd hierarchical-knowledge-grounded-log-mapper
pip install -r requirements.txt
```

### Run the Gradio Web UI

```bash
python gradio_app.py
```

Opens at `http://localhost:7860`. Two input modes:

- **Paste Logs** — paste any raw log events directly (one per line)
- **Upload CSV** — upload a CSV file with a `raw_text` column

### Run from Command Line

```python
from mitre_analyzer_batched import MITRELogAnalyzerBatched

analyzer = MITRELogAnalyzerBatched(
    mitre_kb_path="mitre_detection_kb.json",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    use_quantization=True,
    batch_size=4
)

results = analyzer.analyze("your_logs.csv", max_logs=100)
results.to_csv("hklm_results.csv", index=False)
```

---

## 📂 Project Structure

```
├── gradio_app.py              # Gradio web interface with live log streaming
├── mitre_analyzer_batched.py  # Core HKLM pipeline (3-stage classification)
├── mitre_detection_kb.json    # MITRE ATT&CK knowledge base (tactics, techniques, mitigations)
├── requirements.txt           # Python dependencies
├── app.py                     # HuggingFace Spaces deployment version
├── MITRE_ATT_CK_FrameWork.png # Architecture diagram
└── README.md
```

---

## 🤖 Supported Models

All models run locally — no cloud API, no data leaves your machine.

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| Qwen 2.5 1.5B Instruct | 1.5B | ⚡ Fastest | Large datasets, quick demos |
| Qwen 2.5 3B Instruct | 3B | ⚡ Fast | Good balance for bulk processing |
| Phi-3.5 Mini Instruct | 3.8B | ⚡ Fast | Strong JSON compliance |
| **Mistral 7B Instruct v0.2** | 7B | 🔄 Moderate | **Default — best reasoning quality** |
| Qwen 2.5 7B Instruct | 7B | 🔄 Moderate | Highest accuracy |

---

## ⚡ Optimizations

| Optimization | What It Does | Speedup |
|-------------|-------------|---------|
| **4-bit Quantization** | Compresses model weights (14 GB → 3.5 GB), reduces memory 4x | 2–4x faster |
| **Batched GPU Inference** | Processes N events per GPU call simultaneously | 2.5–3x throughput |
| **Semantic Caching** | MD5 hash lookup skips duplicate log events | Skips 30–60% of events |
| **Hierarchical Narrowing** | Tactic (14 options) → Technique (~10–30 options) | Better accuracy, shorter prompts |

---

## 📊 Dataset

Demonstrated on **DARPA Transparent Computing Engagement 5 (TC E5)** — Five Directions provenance logs.

- **265,189 events** per log file
- Mix of scripted benign activity and real APT attack scenarios
- Windows OS-level provenance: file I/O, registry access, process creation, network connections
- Publicly available from DARPA

> **Note:** HKLM is log-source agnostic. DARPA TC E5 is our demo dataset, but the framework works with any raw text log — Windows Event Logs, syslog, firewall logs, cloud audit logs, NIDS alerts, or anything else.

---

## 🔬 Related Work

| Approach | Input | Limitation vs. HKLM |
|----------|-------|---------------------|
| **TRAM** (MITRE CTID, 2024) | CTI report sentences | Human-written prose, not raw logs. Only 50 techniques. Requires labeled data. |
| **RHINO** (2025) | NIDS alert logs | Suricata/Snort alerts (pre-filtered), not OS-level events. Uses cloud API models. |
| **OntoLogX** (2025) | Raw logs → KG | Builds full knowledge graph first. Tactic-level only, no technique classification. |

**Gap:** No prior work applies open-source LLMs directly to raw OS-level provenance logs for hierarchical ATT&CK classification (tactic → technique) using knowledge-base-constrained reasoning with zero-shot prompting.

---

## ⚠️ Scope & Limitations

**This IS:**
- A detection framework for mapping any log to MITRE ATT&CK
- A post-analysis (forensic/batch) tool — processes collected logs, not real-time streams
- A research proof-of-concept demonstrating constrained LLM reasoning for cybersecurity

**This is NOT:**
- A production SIEM or real-time detection system
- A comparative benchmarking study between models
- A fine-tuned model — uses zero-shot prompting with off-the-shelf LLMs
- Claiming accuracy metrics — DARPA TC E5 has no per-event ATT&CK ground truth labels

---

## 🖥️ Deployment

### Local (recommended for full performance)
```bash
python gradio_app.py
```

### Google Colab (free GPU, shareable link)
```python
!pip install gradio transformers bitsandbytes accelerate pandas
# Upload: gradio_app.py, mitre_analyzer_batched.py, mitre_detection_kb.json

from gradio_app import create_interface
interface = create_interface()
interface.launch(share=True)  # Generates public URL
```

### HuggingFace Spaces
Upload `app.py`, `mitre_analyzer_batched.py`, `mitre_detection_kb.json`, `requirements.txt`, and `README.md` to a new Space with Gradio SDK.

---

## 📖 References

- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [DARPA Transparent Computing](https://www.darpa.mil/program/transparent-computing)
- [DARPA TC E5 Data Release](https://github.com/darpa-i2o/Transparent-Computing)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Qwen 2.5](https://huggingface.co/Qwen)
- [Phi-3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)

---

## 📄 License

This project is developed for academic purposes as part of CIS 544-01: Cyber Defense and Operations at UMass Dartmouth.