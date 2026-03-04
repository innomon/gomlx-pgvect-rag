package embedder

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/disintegration/imaging"
	"github.com/gomlx/gomlx/types/tensor"
)

const (
	// SigLIP Standard Resolution
	ImageSize = 896
)

// LoadImageAsTensor reads an image from disk, resizes it to 896x896,
// and returns a GoMLX tensor of shape [1, 896, 896, 3] in RGB.
func LoadImageAsTensor(path string) (*tensor.Tensor, error) {
	// 1. Open and decode the image
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// 2. Resize to SigLIP standard: 896x896
	// Lanczos is best for preserving features for ML
	resized := imaging.Resize(img, ImageSize, ImageSize, imaging.Lanczos)

	// 3. Convert to Float32 slice [H][W][C]
	bounds := resized.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	data := make([]float32, h*w*3)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			// Convert from 16-bit alpha-premultiplied (0-65535) to 8-bit float (0-255)
			offset := (y*w + x) * 3
			data[offset] = float32(r >> 8)
			data[offset+1] = float32(g >> 8)
			data[offset+2] = float32(b >> 8)
		}
	}

	// 4. Create GoMLX Tensor with Batch Dimension: [1, 896, 896, 3]
	t := tensor.FromFlatDataAndDimensions(data, 1, h, w, 3)
	return t, nil
}
