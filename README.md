# gomlx-pgvect-rag

A high-performance Multimodal Retrieval-Augmented Generation (RAG) system using **T5Gemma 2** (based on the **Gemma 3** architecture), **GoMLX** (XLA-accelerated), and **PostgreSQL** with **pgvector**. This project is packaged as a **Model Context Protocol (MCP) Server**.

## 🚀 Features
- **Multimodal Embedding:** Encodes both text and images into a shared 640-dimension vector space.
- **State-of-the-Art Architecture:** Leverages SigLIP vision encoding and Gemma 3 transformer blocks.
- **GoMLX Engine:** Pure Go implementation of the model graph, accelerated by XLA (CPU/GPU/TPU).
- **Scalable Vector Search:** High-performance similarity retrieval using pgvector HNSW indexing.
- **MCP Native:** Seamlessly integrates as a tool for AI clients like **Goose**, **Claude Desktop**, or **Gemini**.

## 📋 Prerequisites
1. **Go 1.25+**
2. **PostgreSQL 16+** with the `pgvector` extension.
3. **libxla**: Required for GoMLX. Ensure `LD_LIBRARY_PATH` includes your XLA installation.

## ⚙️ Setup

### 1. Database Configuration
Ensure PostgreSQL is running and initialize the schema:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filesys (
    path TEXT PRIMARY KEY,
    metadata JSONB,
    content BYTEA,
    tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    embedding vector(640)
);

CREATE INDEX ON filesys USING hnsw (embedding vector_cosine_ops);
```
Create a `.env` file in the root directory:
```bash
DATABASE_URL="postgres://user:pass@localhost:5432/dbname?sslmode=disable"
MODEL_WEIGHTS_DIR="./models/t5gemma-2-270m"
```

### 2. Download Model Weights
```bash
pip install huggingface_hub
huggingface-cli login
python3 download_weights.py
```

### 3. Build the Server
```bash
CGO_ENABLED=1 go build -o mcp-server cmd/mcp-server/main.go
```

## 🛠️ Usage with MCP Clients

### Goose Configuration
Add the following to your `goose` extension configuration:
```yaml
extensions:
  gomlx-pgvect-rag:
    cmd: /path/to/gomlx-pgvect-rag/mcp-server
    args: ["-weights", "/path/to/gomlx-pgvect-rag/models/t5gemma-2-270m"]
    env:
      DATABASE_URL: "postgres://..."
```

### Available Tools
- `search_multimodal`: Search for assets using a text query or a local image path.
- `ingest_asset`: Ingest a file (image/text) to generate its embedding and store it in the database.

## 📂 Project Structure
- `cmd/mcp-server/`: Main entry point and MCP tool handlers.
- `internal/embedder/`: GoMLX implementation of SigLIP and Gemma 3 encoder blocks.
- `internal/db/`: PostgreSQL and pgvector persistence logic.
- `internal/gomlx_utils/`: Safetensors loading and XLA backend management.
- `pkg/utils/`: Pure Go tokenization using `sugarme/tokenizer`.

## 📝 Troubleshooting
- **Authentication:** If you get SASL errors, verify your `DATABASE_URL` matches your PostgreSQL `pg_hba.conf` settings (use `peer` for local Linux users).
- **Logging:** The server redirects all status messages (e.g., "Loading weights") to `stderr` to avoid corrupting the MCP `stdout` stream.
