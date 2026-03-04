# SPECIFICATION.md - Multimodal RAG with T5Gemma 2

This document details the architecture for the T5Gemma 2 RAG system.

## 1. Technical Stack
- **ML Framework:** GoMLX (XLA-accelerated model inference).
- **Models:** T5Gemma 2 (270M, 1B, or 4B variants).
  - Vision Encoder: SigLIP (896x896 input, 256 tokens).
  - Text Encoder: T5.
  - Generative Decoder: Gemma 2.
- **Database:** PostgreSQL + `pgvector`.
- **Drivers:** `pgx/v5`, `pgvector-go`.

## 2. Multimodal Embedding Strategy
- **Image Preprocessing:**
  - Resize to **896x896** using Lanczos resampling.
  - Normalize to `[-1, 1]` within the GoMLX graph: `(input/255.0 - 0.5) / 0.5`.
- **Tokenization:**
  - Image: 256 visual tokens.
  - Text: Standard Gemma tokenizer.
- **Encoding:** Concatenate visual and text tokens -> T5Gemma 2 Encoder.
- **Pooling:** Perform **Mean Pooling** across the sequence dimension of the last hidden state.
- **Output:** Vector of dimension **1152** (for the 270M model).

## 3. Database Schema (PostgreSQL)
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filesys (
    path TEXT PRIMARY KEY,
    metadata JSONB,
    content BYTEA,
    tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1152) -- For T5Gemma 2-270M
);

-- Index metadata
CREATE INDEX idx_filesys_metadata ON filesys USING GIN (metadata);

-- Index embeddings with HNSW for Cosine Similarity
CREATE INDEX idx_filesys_embedding ON filesys 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);
```

## 4. MCP Server Architecture
The system will be packaged as an **MCP Server**, exposing the following tools to MCP-compliant clients:

use the official go mcp sdk : `github.com/modelcontextprotocol/go-sdk/mcp`

### 4.1. Tools
- **`search_multimodal`**:
  - **Inputs:** `query_text` (string, optional), `query_image_path` (string, optional).
  - **Description:** Encodes the input using T5Gemma 2, performs a vector search in pgvector, and returns the top-K relevant contexts (metadata + content summary).
- **`ingest_asset`**:
  - **Inputs:** `path` (string), `metadata` (JSON string).
  - **Description:** Reads a local file, generates a T5Gemma 2 embedding, and performs an UPSERT into the PostgreSQL database.
- **`get_asset_details`**:
  - **Inputs:** `path` (string).
  - **Description:** Retrieves the full metadata and content BLOB for a specific asset.

## 5. RAG Engine Lifecycle
1.  **Ingestion (via MCP Tool):**
    - Load file (Text/Image) -> Resize & Normalize (if image).
    - GoMLX Inference (Encoder) -> Mean Pool -> Result Vector.
    - **UPSERT** into PostgreSQL `filesys` table.
2.  **Retrieval (via MCP Tool):**
    - Query (Text/Image) -> Embed using same GoMLX Encoder.
    - PG Vector Search: `ORDER BY embedding <=> $1 LIMIT K`.
    - Return relevant metadata to the LLM.
3.  **Generation (Client-Side):**
    - The MCP Client (LLM) uses the retrieved context to generate a final response.

## 6. Deployment Considerations
- **Protocol:** JSON-RPC over Standard I/O (stdio).
- **GoMLX Runtime:** The MCP server process must have access to `libxla` and the model weights.
- **Distance Metric:** Always use **Cosine Distance** (`<=>`).
- **RAM Usage:** 1152-dim HNSW index for 1M rows requires ~5-7GB RAM.
- **Precision:** Use `halfvec` (FP16) for larger variants (1B/4B) to stay within PG limits (4000 dims).
