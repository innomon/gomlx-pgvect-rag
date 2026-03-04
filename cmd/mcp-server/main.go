package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/internal/rag"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func main() {
	var weightsDir string
	flag.StringVar(&weightsDir, "weights", os.Getenv("MODEL_WEIGHTS_DIR"), "Directory containing .safetensors weights")
	flag.Parse()

	// 1. Initialize GoMLX Backend
	backend, err := gomlx_utils.InitializeBackend()
	if err != nil {
		log.Fatalf("GoMLX initialization failed: %v", err)
	}

	// 2. Initialize Model and Load Weights
	model := gomlx_utils.NewModel(backend)
	if weightsDir != "" {
		fmt.Fprintf(os.Stderr, "📂 Loading weights from: %s\n", weightsDir)
		if err := model.LoadSafetensors(weightsDir); err != nil {
			log.Fatalf("Failed to load weights: %v", err)
		}
	} else {
		fmt.Fprintf(os.Stderr, "⚠️  No weights directory provided. Embeddings will be stubs.\n")
	}

	// 3. Initialize Database Connection
	ctx := context.Background()
	pool, err := db.Connect(ctx)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer pool.Close()

	// 4. Initialize Orchestrator
	orchestrator := &rag.Orchestrator{
		DB:    pool,
		Model: model,
	}

	// 4. Create MCP Server
	s := server.NewMCPServer(
		"gomlx-pgvect-rag",
		"1.0.0",
	)

	// 5. Register Tools
	// search_multimodal
	searchTool := mcp.NewTool("search_multimodal",
		mcp.WithDescription("Search for relevant text or image assets in the multimodal RAG store."),
		mcp.WithSchema(mcp.NewParams(
			mcp.NewStringParam("query_text", "Textual query for similarity search."),
			mcp.NewStringParam("query_image_path", "Local path to an image file for visual similarity search."),
			mcp.NewIntParam("limit", "Maximum number of results to return (default: 5)."),
		)),
	)
	s.AddTool(searchTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		queryText, _ := request.Params["query_text"].(string)
		queryImage, _ := request.Params["query_image_path"].(string)
		limit, ok := request.Params["limit"].(float64)
		if !ok {
			limit = 5
		}

		assets, err := orchestrator.Search(ctx, queryText, queryImage, int(limit))
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Search failed: %v", err)), nil
		}

		// Convert assets to JSON response
		return mcp.NewToolResultText(fmt.Sprintf("Found %d results: %+v", len(assets), assets)), nil
	})

	// ingest_asset
	ingestTool := mcp.NewTool("ingest_asset",
		mcp.WithDescription("Ingest a new file (image or text) into the multimodal RAG store."),
		mcp.WithSchema(mcp.NewParams(
			mcp.NewStringParam("path", "Local path to the file to ingest."),
			mcp.NewStringParam("metadata", "JSON string containing metadata for the asset."),
		)),
	)
	s.AddTool(ingestTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		path, _ := request.Params["path"].(string)
		// Simplified metadata handling for now
		err := orchestrator.Ingest(ctx, path, map[string]interface{}{"source": "mcp-tool"})
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Ingestion failed: %v", err)), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Successfully ingested: %s", path)), nil
	})

	// 6. Start the MCP Server (stdio)
	fmt.Fprintf(os.Stderr, "🚀 gomlx-pgvect-rag MCP Server running on %s\n", backend.Name())
	if err := server.ServeStdio(s); err != nil {
		log.Fatalf("MCP server error: %v", err)
	}
}
