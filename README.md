# ClariGen: Ambiguity Detection and Clarification System

## Project Description
Ambiguous user queries pose a significant challenge to Retrieval-Augmented Generation (RAG) systems, often resulting in unclear or irrelevant responses. ClariGen is an end-to-end Large Language Model (LLM)-driven pipeline for interactive ambiguity resolution in Question-Answering (QA) systems.

The system employs a dual-model "fast-slow" architecture: a lightweight instruction-tuned model performs low-latency binary ambiguity detection, while a larger model conducts in-depth ambiguity reasoning using a modified CLAMBER taxonomy and generates targeted clarifying questions. By resolving ambiguity *before* retrieval, ClariGen ensures that RAG systems retrieve grounded, relevant information.

## Approach
ClariGen leverages a tiered LLM approach to balance computational efficiency with reasoning depth:

1.  **Fast Path (Detection)**: A lightweight 8B model (Llama-3.1-8B) acts as a high-throughput gatekeeper. It performs binary ambiguity detection to filter out clear queries in milliseconds, minimizing latency for the majority of traffic.
2.  **Reasoning Path (Clarification)**: Queries flagged as ambiguous are escalated to a 70B model (Llama-3.3-70B). This model handles complex ambiguity diagnosis, generates clarifying questions via Chain-of-Thought (CoT) prompting, and reformulates queries based on user feedback.

## Demo
See [assets/clarification-module-demo.mp4](assets/clarification-module-demo.mp4) for a demo of ClariGen in action.

## Results
The system has been evaluated on standard benchmarks and real-world case studies:

*   **Ambiguity Detection**: Zero-shot prompting on Llama-3.1-8B proved highly effective, achieving an F1 score of **73.14%** on the ClariQ benchmark. Experiments showed that zero-shot approaches outperformed few-shot prompting, which tended to introduce classification bias.
*   **Clarification Generation**: The **Ambiguity-Type Chain-of-Thought (AT-CoT)** strategy achieved the highest semantic similarity to human-generated questions, with a BERTScore F1 of **0.9187** on ClariQ and **0.9093** on QULAC.
*   **Case Studies**: In a domain-specific evaluation on fall prevention inquiries for older adults, ClariGen successfully resolved lexical and semantic ambiguities (e.g., clarifying "exercises" vs "home exercises"), leading to significantly more targeted and practical retrieval results compared to baseline RAG responses.

## Quick Start

Get ClariGen up and running in three steps:

### 1. Set Up Environment
```bash
# Clone and enter the repo
cp .env.example .env
# Edit .env and set your model URLs and API keys
```

### 2. Start Model Servers
ClariGen requires two model servers (port 8368 and 8369 by default).
```bash
cd llm_hosting
./serve_models.sh  # Requires vLLM and GPUs
```
*See [LLM Hosting](llm_hosting/README.md) for remote setup or alternative hosting.*

### 3. Launch Applications
Run the backend and frontend in separate terminals:
```bash
# Terminal A: Start API
uvicorn apps.api.main:app --port 8370

# Terminal B: Start UI
streamlit run apps/frontend/app.py --server.port 8501
```

---

## Project Structure

- [**Core Library**](core/README.md): The heart of ClariGen (Detection & Orchestration).
- [**API Backend**](apps/api/README.md): FastAPI service providing programmatic access.
- [**Frontend**](apps/frontend/README.md): Streamlit-based interactive playground.
- [**Model Hosting**](llm_hosting/README.md): Scripts for serving Llama-3 models via vLLM.
- [**Deployment**](deployment/README.md): Docker and SSH tunnel configurations for production-like setups.
- [**Evaluation**](evaluation/README.md): Tools to measure system performance on benchmark datasets.

## Configuration

Detailed environment settings can be found in [ENVIRONMENT.md](docs/ENVIRONMENT.md).

**Key Settings:**
- `CLARIFICATION_STRATEGY`: `at_standard` (default), `at_cot` (advanced), or `vanilla`.
- `MAX_CLARIFICATION_ATTEMPTS`: Limits the iteration count.

## License

Copyright © 2026, Wireless System Research Group (WiSeR), McMaster University.
Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
