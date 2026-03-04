# gomlx-pgvect-rag

A Multimodal Retrieval-Augmented Generation (RAG) system using **T5Gemma 2**, **GoMLX** (XLA-accelerated), and **PostgreSQL** with **pgvector**, packaged as an **MCP (Model Context Protocol) Server**.

## Features
- **Multimodal Retrieval:** Search across text and images in a unified vector space.
- **GoMLX Engine:** High-performance model inference using XLA (CPU/GPU/TPU).
- **pgvector Integration:** Scalable vector similarity search using HNSW indexing.
- **MCP Server:** Exposes tools for ingestion and search to any MCP-compliant LLM client.

## Prerequisites
1. **Go 1.25+**
2. **PostgreSQL** with the `pgvector` extension installed.
3. **libxla**: Required for GoMLX. Follow the [GoMLX installation guide](https://github.com/gomlx/gomlx) to set up the XLA backend for your platform (ARM64/AARCH64).

## Setup

### 1. Database
Ensure PostgreSQL is running and the `vector` extension is enabled:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
Set your database connection string:
```bash
export DATABASE_URL="postgres://postgres@localhost:5432/postgres?sslmode=disable"
```

### 2. Download Model Weights
Download the T5Gemma 2-270M Safetensors from Hugging Face:
```bash
pip install huggingface_hub
huggingface-cli login
python3 download_weights.py
```

### 3. Build & Run
To start the MCP server, use the following command (ensuring `CGO_ENABLED=1` for the tokenizer bindings):

```bash
CGO_ENABLED=1 go run cmd/mcp-server/main.go -weights ./models/t5gemma-2-270m
```

## MCP Tools
The server exposes the following tools to your LLM client:
- `search_multimodal`: Search for assets using text or image paths.
- `ingest_asset`: Ingest a local file (image/text) and generate its embedding.

## Project Structure
- `cmd/mcp-server/`: MCP Server entry point and tool definitions.
- `internal/db/`: PostgreSQL and pgvector logic.
- `internal/embedder/`: GoMLX image preprocessing and model graph.
- `internal/gomlx_utils/`: Safetensors loading and backend initialization.
- `pkg/utils/`: Tokenizer and other shared utilities.
