package rag

import (
	"context"
	"fmt"

	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/embedder"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/pkg/utils"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Orchestrator manages the RAG pipeline.
type Orchestrator struct {
	DB        *pgxpool.Pool
	Model     *gomlx_utils.Model
	Tokenizer *utils.Tokenizer
}

// Search retrieves relevant assets based on text or image input.
func (o *Orchestrator) Search(ctx context.Context, text string, imagePath string, limit int) ([]db.Asset, error) {
	var queryVec []float32
	var err error

	// 1. Generate Embedding
	if imagePath != "" {
		imgTensor, err := embedder.LoadImageAsTensor(imagePath)
		if err != nil {
			return nil, fmt.Errorf("failed to load image for search: %w", err)
		}
		// queryVec = o.Model.EmbedImage(imgTensor)
		queryVec = make([]float32, 640) // Dimension matching Gemma 2 hidden_size
		_ = imgTensor
	} else if text != "" {
		tokens, err := o.Tokenizer.Encode(text, true)
		if err != nil {
			return nil, fmt.Errorf("failed to tokenize query: %w", err)
		}
		// queryVec = o.Model.EmbedText(tokens)
		queryVec = make([]float32, 640)
		_ = tokens
	} else {
		return nil, fmt.Errorf("either text or imagePath must be provided")
	}

	// 2. Query Vector DB
	results, err := db.SearchSimilar(ctx, o.DB, queryVec, limit)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	return results, nil
}

// Ingest adds a new asset to the RAG store.
func (o *Orchestrator) Ingest(ctx context.Context, path string, metadata map[string]interface{}) error {
	// 1. Load file
	// 2. Generate Embedding via GoMLX
	// 3. Upsert into DB
	
	// Mock implementation for flow verification:
	asset := db.Asset{
		Path:      path,
		Metadata:  metadata,
		Embedding: make([]float32, 1152),
	}
	
	return db.UpsertAsset(ctx, o.DB, asset)
}
