# IMPLEMENTATION_PLAN.md - Multimodal RAG with T5Gemma 2

This plan outlines the steps for building the T5Gemma 2 multimodal RAG system.

## Implementation Checklist

### Phase 1: Environment & Project Setup
- [ ] Install Go (1.23+), PostgreSQL, and `pgvector` extension.
- [ ] Install `libxla` and configure `LD_LIBRARY_PATH`.
- [ ] Initialize Go project (`go mod init`).
- [ ] Add dependencies:
  - `go get github.com/gomlx/gomlx@latest`
  - `go get github.com/jackc/pgx/v5`
  - `go get github.com/pgvector/pgvector-go`
  - `go get github.com/disintegration/imaging`
  - `go get github.com/modelcontextprotocol/go-sdk/mcp`

### Phase 2: Database Schema (PostgreSQL)
- [ ] Create `filesys` table with `path` (PK), `metadata` (JSONB), `content` (BYTEA), `embedding` (vector(1152)).
- [ ] Create GIN index for metadata.
- [ ] Create HNSW index for embeddings with `vector_cosine_ops`.
- [ ] Implement `internal/db/upsert.go` with `ON CONFLICT` support for the `filesys` table.

### Phase 3: Image Preprocessing (GoMLX)
- [ ] Implement Go logic to decode JPG/PNG and resize to **896x896** using Lanczos resampling.
- [ ] Implement GoMLX graph for SigLIP normalization: `(float32(pixels)/255.0 - 0.5) / 0.5`.
- [ ] Verify 4D tensor output shape: `[batch, 896, 896, 3]`.

### Phase 4: Model Integration (T5Gemma 2 Encoder)
- [ ] Convert T5Gemma 2-270M weights for GoMLX.
- [ ] Implement T5Gemma encoder logic in GoMLX.
- [ ] Implement **Mean Pooling** graph: `graph.ReduceMean(encoderHiddenStates, 1)`.
- [ ] Verify 1152-dimension output vector.

### Phase 5: MCP Server Implementation
- [ ] Choose a Go MCP library (e.g., `github.com/mark3labs/mcp-go`).
- [ ] Define the MCP Server with its tools (`search_multimodal`, `ingest_asset`, `get_asset_details`).
- [ ] Implement tool handlers that call the underlying RAG logic.
- [ ] Set up the standard input/output (stdio) loop for MCP communication.
- [ ] Test the server using the MCP Inspector or a compliant client.

### Phase 6: Interface & Verification
- [ ] Implement `cmd/mcp-server/main.go` to initialize GoMLX and start the MCP loop.
- [ ] Test the full flow: LLM calls `ingest_asset` -> GoMLX embeds -> PG stores.
- [ ] Test search: LLM calls `search_multimodal` -> PG finds context -> LLM receives JSON.
- [ ] Benchmark MCP tool-call latency.

## Step-by-Step Details

### DB Creation SQL
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE filesys (
    path TEXT PRIMARY KEY,
    metadata JSONB,
    content BYTEA,
    tmstamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1152)
);
CREATE INDEX ON filesys USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
```

### Normalization Logic
```go
// Normalization Maps [0, 255] to [-1, 1]
images = DivScalar(ConvertDType(rawImages, Float32), 255.0)
normalized = Div(Sub(images, Const(g, 0.5)), Const(g, 0.5))
```

### PG Search Accuracy Tuning
```sql
SET hnsw.ef_search = 100;
```
