package embedder

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

// SigLIPTransformerBlock is a standard ViT transformer layer.
func SigLIPTransformerBlock(ctx *context.Context, x *Node, numHeads, intermediateDim int) *Node {
	// 1. Pre-Attention LayerNorm
	normX := layers.LayerNorm(ctx.In("pre_attention_norm"), x, -1).Done()

	// 2. Attention with Residual
	attn := layers.MultiHeadAttention(ctx.In("attention"), normX, normX, normX, numHeads, -1).Embeddings
	x = Add(x, attn)

	// 3. Pre-MLP LayerNorm
	normX = layers.LayerNorm(ctx.In("pre_mlp_norm"), x, -1).Done()

	// 4. MLP with Residual
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	mlpOut := layers.Dense(ctx.In("mlp/gate"), normX, true, intermediateDim)
	mlpOut = layers.GELU(mlpOut)
	mlpOut = layers.Dense(ctx.In("mlp/down"), mlpOut, true, hiddenDim)
	x = Add(x, mlpOut)

	return x
}

// SigLIPVisionEncoder processes image tokens (27 layers).
func SigLIPVisionEncoder(ctx *context.Context, x *Node) *Node {
	ctx = ctx.In("siglip_vision")
	g := x.Graph()
	
	// 1. Patch Embedding (Convolution with stride 14)
	x = layers.Convolution(ctx.In("patch_embed"), x).
		Filters(1152).
		KernelSize(14).
		Stride(14).
		NoBias().
		Done()
		
	// 2. Flatten and add Positional Embeddings
	batchSize := x.Shape().Dimensions[0]
	x = Reshape(x, batchSize, 4096, 1152)
	
	posEmbed := ctx.VariableWithShape("position_embeddings", []int{4096, 1152}).GetPrecomputed(g)
	x = Add(x, posEmbed)
	
	// 3. Token Reduction: 4096 -> 256
	x = Reshape(x, batchSize, 16, 4, 16, 4, 1152)
	x = ReduceMean(x, 2, 4)
	x = Reshape(x, batchSize, 256, 1152)
	
	numLayers := 27
	numHeads := 16
	intermediateDim := 4304
	
	for i := 0; i < numLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("%d", i))
		x = SigLIPTransformerBlock(layerCtx, x, numHeads, intermediateDim)
	}
	
	return layers.LayerNorm(ctx.In("final_norm"), x, -1).Done()
}
