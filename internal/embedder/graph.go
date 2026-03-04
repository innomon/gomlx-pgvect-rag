package embedder

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// PreprocessImageGraph takes a raw image tensor [batch, height, width, 3]
// and returns a normalized version ready for the SigLIP encoder.
func PreprocessImageGraph(ctx *context.Context, rawImages *Node) *Node {
	// 1. Ensure float32 precision
	images := ConvertDType(rawImages, dtypes.Float32)

	// 2. Rescale from [0, 255] to [0, 1]
	images = DivScalar(images, 255.0)

	// 3. SigLIP Normalization: (x - 0.5) / 0.5
	mean := Const(rawImages.Graph(), 0.5)
	std := Const(rawImages.Graph(), 0.5)

	normalized := Div(Sub(images, mean), std)

	return normalized
}

// MeanPoolingGraph averages across the sequence dimension (axis 1)
func MeanPoolingGraph(ctx *context.Context, encoderHiddenStates *Node) *Node {
	return ReduceMean(encoderHiddenStates, 1)
}
