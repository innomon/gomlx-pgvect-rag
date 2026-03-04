package embedder

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

// RMSNorm implements Root Mean Square Layer Normalization.
// Gemma 3 uses this before each attention and MLP block.
func RMSNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	ctx = ctx.In("rms_norm")
	// x shape: [batch, seq_len, hidden_dim]
	
	// 1. Calculate Mean Square: mean(x^2)
	ms := ReduceMean(Square(x), -1)
	
	// 2. Normalize: x / sqrt(ms + eps)
	invRms := Inverse(Sqrt(AddScalar(ms, epsilon)))
	normalized := Mul(x, invRms)
	
	// 3. Scale with learnable weight (gamma)
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	gamma := ctx.VariableWithShape("weight", []int{hiddenDim}).GetPrecomputed(x.Graph())
	
	return Mul(normalized, gamma)
}

// MLP (Feed-forward) block for Gemma 3.
// Uses Gated Linear Unit (GLU) with GELU activation.
func MLP(ctx *context.Context, x *Node, intermediateDim int) *Node {
	ctx = ctx.In("mlp")
	
	// 1. Gate Projection + GELU
	gate := layers.Dense(ctx.In("gate_proj"), x, true, intermediateDim)
	gate = layers.GELU(gate)
	
	// 2. Up Projection
	up := layers.Dense(ctx.In("up_proj"), x, true, intermediateDim)
	
	// 3. Element-wise product (Gated Linear Unit)
	intermediate := Mul(gate, up)
	
	// 4. Down Projection (back to hidden_dim)
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	return layers.Dense(ctx.In("down_proj"), intermediate, true, hiddenDim)
}

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
		layerCtx := ctx.In(fmt.Sprintf("%d", i))
		
		// Alternating attention types
		ropeTheta := 10000.0
		currentSlidingWindow := slidingWindow
		if (i+1)%6 == 0 {
			ropeTheta = 1000000.0
			currentSlidingWindow = -1
		}

		x = EncoderBlock(layerCtx, x, numHeads, numKVHeads, headDim, intermediateDim, currentSlidingWindow, ropeTheta)
	}

	return RMSNorm(ctx.In("final_norm"), x, 1e-6)
}
