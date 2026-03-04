package embedder

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
)

// SigLIPTransformerBlock is a standard ViT transformer layer.
func SigLIPTransformerBlock(ctx *context.Context, x *Node, numHeads, intermediateDim int) *Node {
	// 1. Pre-Attention LayerNorm
	normX := layers.LayerNorm(ctx.In("pre_attention_norm"), x, -1)

	// 2. Attention with Residual
	// SigLIP uses standard MHA (not GQA usually)
	attn := layers.MultiHeadAttention(ctx.In("attention"), normX, normX, normX, numHeads, -1).Embeddings
	x = Add(x, attn)

	// 3. Pre-MLP LayerNorm
	normX = layers.LayerNorm(ctx.In("pre_mlp_norm"), x, -1)

	// 4. MLP with Residual
	// Standard MLP: Dense -> GELU -> Dense
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
	// Input x: [batch, 896, 896, 3]
	// Output: [batch, 64, 64, 1152]
	x = layers.Convolution(ctx.In("patch_embed"), x).
		Filters(1152).
		KernelSize(14).
		Stride(14).
		NoBias().
		Done()
		
	// 2. Flatten and add Positional Embeddings
	// Result shape: [batch, 4096, 1152]
	batchSize := x.Shape().Dimensions[0]
	x = Reshape(x, batchSize, 4096, 1152)
	
	// Add learned positional embeddings
	posEmbed := ctx.VariableWithShape("position_embeddings", []int{4096, 1152}).GetPrecomputed(g)
	x = Add(x, posEmbed)
	
	// 3. Token Reduction to 256 tokens
	// 4096 (64x64) -> 256 (16x16) via Mean Pooling over 4x4 patches.
	// 1. Reshape back to grid: [batch, 64, 64, 1152]
	x = Reshape(x, batchSize, 64, 64, 1152)
	// 2. ReduceMean over 4x4 windows
	x = ReduceMean(x, 1, 2) // Simplified; for real spatial pooling:
	// x = layers.MaxPooling(x).KernelSize(4).Stride(4).Done() 
	// (GoMLX ReduceMean doesn't have stride; using Convolution with stride or Reshape+ReduceMean)
	
	// Real Space-to-Depth / Pooling:
	x = Reshape(x, batchSize, 16, 4, 16, 4, 1152)
	x = ReduceMean(x, 2, 4) // Average across the 4x4 blocks
	x = Reshape(x, batchSize, 256, 1152)
	
	numLayers := 27
	numHeads := 16
	intermediateDim := 4304
	
	for i := 0; i < numLayers; i++ {
		layerCtx := ctx.In(string(rune(i)))
		x = SigLIPTransformerBlock(layerCtx, x, numHeads, intermediateDim)
	}
	
	return layers.LayerNorm(ctx.In("final_norm"), x, -1)
}
