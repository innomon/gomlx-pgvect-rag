package embedder

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
)

// EncoderBlock represents a single Gemma 3 transformer layer.
func EncoderBlock(ctx *context.Context, x *Node, numHeads, numKVHeads, headDim, intermediateDim, slidingWindow int, ropeTheta float64) *Node {
	// 1. Pre-Attention RMSNorm
	normX := RMSNorm(ctx.In("pre_attention_norm"), x, 1e-6)

	// 2. Attention with Residual Connection
	attn := MultiHeadAttention(ctx, normX, numHeads, numKVHeads, headDim, slidingWindow, ropeTheta)
	x = Add(x, attn)

	// 3. Pre-MLP RMSNorm
	normX = RMSNorm(ctx.In("pre_mlp_norm"), x, 1e-6)

	// 4. MLP with Residual Connection
	mlpOut := MLP(ctx, normX, intermediateDim)
	x = Add(x, mlpOut)

	return x
}

// Gemma3Encoder assembles 18 transformer layers for T5Gemma 2-270M.
func Gemma3Encoder(ctx *context.Context, x *Node) *Node {
	ctx = ctx.In("gemma3_encoder")
	
	// Model parameters for 270M variant
	numLayers := 18
	numHeads := 4
	numKVHeads := 1
	headDim := 256
	intermediateDim := 2048
	slidingWindow := 512

	for i := 0; i < numLayers; i++ {
		layerCtx := ctx.In(string(rune(i)))
		
		// Alternating attention types (Every 6th layer is full attention)
		ropeTheta := 10000.0 // Default for sliding attention
		currentSlidingWindow := slidingWindow
		if (i+1)%6 == 0 {
			ropeTheta = 1000000.0 // Full attention theta
			currentSlidingWindow = -1 // No window
		}

		x = EncoderBlock(layerCtx, x, numHeads, numKVHeads, headDim, intermediateDim, currentSlidingWindow, ropeTheta)
	}

	// Final Layer Norm
	return RMSNorm(ctx.In("final_norm"), x, 1e-6)
}
