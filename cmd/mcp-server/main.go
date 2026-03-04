package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/innomon/gomlx-pgvect-rag/internal/db"
	"github.com/innomon/gomlx-pgvect-rag/internal/embedder"
	"github.com/innomon/gomlx-pgvect-rag/internal/gomlx_utils"
	"github.com/innomon/gomlx-pgvect-rag/internal/rag"
	"github.com/innomon/gomlx-pgvect-rag/pkg/utils"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	// Register XLA backend
	_ "github.com/gomlx/gomlx/backends/xla"
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
		// Compile the graph once weights are loaded
		fmt.Fprintf(os.Stderr, "🛠️  Compiling GoMLX graph...\n")
		model.CompileEmbed(embedder.EmbedMultimodalGraph)
	} else {
		fmt.Fprintf(os.Stderr, "⚠️  No weights directory provided. Embeddings will be stubs.\n")
	}

	// 3. Initialize Tokenizer
	var tk *utils.Tokenizer
	if weightsDir != "" {
		tokenizerPath := filepath.Join(weightsDir, "tokenizer.json")
		tk, err = utils.NewTokenizer(tokenizerPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "⚠️  Failed to load tokenizer from %s: %v\n", tokenizerPath, err)
		}
	}

	// 4. Initialize Database Connection
	ctx := context.Background()
	pool, err := db.Connect(ctx)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer pool.Close()

	// 5. Initialize Orchestrator
	orchestrator := &rag.Orchestrator{
		DB:        pool,
		Model:     model,
		Tokenizer: tk,
	}

	// 6. Create MCP Server
	s := server.NewMCPServer(
		"gomlx-pgvect-rag",
		"1.0.0",
	)

	// 7. Register Tools
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

		return mcp.NewToolResultText(fmt.Sprintf("Found %d results: %+v", len(assets), assets)), nil
	})

	// ingest_asset
	ingestTool := mcp.NewTool("ingest_asset",
		mcp.WithDescription("Ingest a new file (image or text) into the multimodal RAG store."),
		mcp.WithSchema(mcp.NewParams(
			mcp.NewStringParam("path", "Local path to the file to ingest."),
		)),
	)
	s.AddTool(ingestTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		path, _ := request.Params["path"].(string)
		err := orchestrator.Ingest(ctx, path, map[string]interface{}{"source": "mcp-tool"})
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Ingestion failed: %v", err)), nil
		}

		return mcp.NewToolResultText(fmt.Sprintf("Successfully ingested: %s", path)), nil
	})

	// 8. Start the MCP Server (stdio)
	fmt.Fprintf(os.Stderr, "🚀 gomlx-pgvect-rag MCP Server running on %s\n", backend.Name())
	if err := server.ServeStdio(s); err != nil {
		log.Fatalf("MCP server error: %v", err)
	}
}
