# The Relay: System Under Pressure

An empirical research assistant answering technical questions over three arXiv papers (*Attention Is All You Need*, *Moshi*, and *Dynamic Chunking*) under severe rate limits, calibrated against real Groq measurements.

---

## Architecture Overview

- **5-Step Pipeline**: Understand $\to$ Retrieve $\to$ Decide $\to$ Draft $\to$ Verify.
- **Admission Control**: Bounded queue with client deadlines and load shedding.
- **Rate & Token Limiter**: Proactively tracks RPM, TPM, RPD, TPD, and in-flight concurrency.
- **Degradation Ladder**:
  1. *Normal*: Full context ($k$ chunks), complete verify loop.
  2. *Near Limit ($>75\%$ TPM/RPM)*: Trim chunks, cap tokens to 256.
  3. *Limit Hit (429)*: Exponential backoff with jitter respecting `retry-after`.
  4. *Daily Budget Gone*: Deterministic synthesis delivering cited passages directly with zero LLM calls.
- **Calibrated Stand-in Server**: FastAPI mock implementing SSE token streaming, concurrency degradation, and 429 headers.
- **Results Engine**: Single authority writing `results.csv` and computing Tables 1, 2a, 2b, and 3.

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `httpx`, `fastapi`, `uvicorn`, `rank_bm25`, `tiktoken`, `pypdf`, `pymupdf`)*

### 2. Fetch Papers & Extract Text
```bash
python scripts/fetch_papers.py
```

### 3. Probe Real Groq Service (Optional / Live Verification)
```bash
export GROQ_API_KEY="your_api_key_here"
python scripts/probe_groq.py --n 10 --rate 15
```

### 4. Start Stand-in Model Server
```bash
python -m standin.server
```

### 5. Run Benchmark Load Shapes
```bash
python scripts/run_load.py --scenario all
```

### 6. Compute Required Results Tables
```bash
python scripts/analyze.py
```

All analysis and write-up details are available in [WRITEUP.md](WRITEUP.md).
