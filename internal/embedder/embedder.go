package embedder

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
)

// EmbedMultimodalGraph builds the full GoMLX graph for text+image encoding.
func EmbedMultimodalGraph(ctx *context.Context, textIds, imagePixels *Node) *Node {
	// 1. Preprocess Images: [batch, 896, 896, 3] -> [-1, 1]
	normImages := PreprocessImageGraph(ctx, imagePixels)

	// 2. Vision Encoder (SigLIP)
	// (Assuming patch-embedding is part of SigLIPVisionEncoder for this graph)
	// result shape: [batch, 256, 1152]
	visualTokens := SigLIPVisionEncoder(ctx, normImages)

	// 3. Vision-to-Text Projection
	// Maps 1152 visual dimensions to 640 text encoder dimensions.
	ctx = ctx.In("vision_projection")
	visualTokens = layers.Dense(ctx, visualTokens, true, 640)

	// 4. Text Embedding
	// Maps token IDs to 640-dim vectors.
	ctx = ctx.In("text_embedding")
	textTokens := layers.Embedding(ctx, textIds, 262144, 640)

	// 5. Concatenate Tokens: [batch, 256 + text_len, 640]
	// Join along the sequence dimension (axis 1)
	combinedTokens := Concatenate([]*Node{visualTokens, textTokens}, 1)

	// 6. Gemma 3 Multimodal Encoder
	// (18 blocks of alternating attention)
	// result shape: [batch, 256 + text_len, 640]
	encoderHiddenStates := Gemma3Encoder(ctx, combinedTokens)

	// 7. Mean Pooling
	// result shape: [batch, 640]
	return MeanPoolingGraph(ctx, encoderHiddenStates)
}
