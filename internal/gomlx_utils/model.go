package gomlx_utils

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/nlpodyssey/safetensors"
)

// InitializeBackend sets up the XLA backend for GoMLX.
func InitializeBackend() (backends.Backend, error) {
	if os.Getenv("GOMLX_BACKEND") == "" {
		os.Setenv("GOMLX_BACKEND", "xla")
	}

	backend, err := backends.New()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize GoMLX backend: %w", err)
	}

	return backend, nil
}

// Model represents the T5Gemma 2 model context and weights.
type Model struct {
	Backend   backends.Backend
	Context   *context.Context
	ExecEmbed *graph.Exec
}

// NewModel initializes the GoMLX context.
func NewModel(backend backends.Backend) *Model {
	return &Model{
		Backend: backend,
		Context: context.New(),
	}
}

// CompileEmbed compiles the multimodal embedding graph for inference.
func (m *Model) CompileEmbed(buildFn func(ctx *context.Context, textIds, imagePixels *graph.Node) *graph.Node) {
	m.ExecEmbed = graph.NewExec(m.Backend, func(textIds, imagePixels *graph.Node) *graph.Node {
		return buildFn(m.Context, textIds, imagePixels)
	})
}

// Embed executes the compiled GoMLX graph.
func (m *Model) Embed(textIds []uint32, imageTensor *tensor.Tensor) ([]float32, error) {
	if m.ExecEmbed == nil {
		return nil, fmt.Errorf("embedding graph not compiled")
	}

	// 1. Create Text ID tensor: [batch=1, seq_len]
	// Use uint32 for vocabulary IDs
	textT := tensor.FromFlatDataAndDimensions(textIds, 1, len(textIds))

	// 2. Run inference
	results := m.ExecEmbed.Call(textT, imageTensor)
	if len(results) == 0 {
		return nil, fmt.Errorf("no output from graph execution")
	}

	// 3. Convert output tensor to []float32
	// Expecting shape [1, 640]
	outT := results[0]
	return outT.FlatData().([]float32), nil
}

// LoadSafetensors loads one or more .safetensors files into the model's context.
func (m *Model) LoadSafetensors(weightsDir string) error {
	files, err := filepath.Glob(filepath.Join(weightsDir, "*.safetensors"))
	if err != nil {
		return fmt.Errorf("failed to list safetensors files: %w", err)
	}

	if len(files) == 0 {
		return fmt.Errorf("no .safetensors files found in %s", weightsDir)
	}

	for _, file := range files {
		fmt.Printf("📦 Loading weights from %s...\n", filepath.Base(file))
		st, err := safetensors.Open(file)
		if err != nil {
			return fmt.Errorf("failed to open safetensors file %s: %w", file, err)
		}

		// Iterate through each tensor in the file
		for _, name := range st.Names() {
			t, err := st.Tensor(name)
			if err != nil {
				return fmt.Errorf("failed to read tensor %s: %w", name, err)
			}

			// Map Safetensors name to GoMLX context scope
			gomlxScope := mapHuggingFaceToGoMLX(name)

			// Safetensors data is typically little-endian raw bytes
			data := t.Data()
			shape := make([]int, len(t.Shape()))
			for i, s := range t.Shape() {
				shape[i] = int(s)
			}

			var goMLXTensor *tensor.Tensor
			switch t.Dtype() {
			case safetensors.Float32:
				floatData := *(*[]float32)(unsafe.Pointer(&data))
				// Take sub-slice because the slice header capacity might be different
				floatData = floatData[:len(data)/4]
				goMLXTensor = tensor.FromFlatDataAndDimensions(floatData, shape...)
			default:
				// Skip unsupported types for now (bfloat16 needs conversion)
				continue
			}

			// Store in context at the mapped scope
			m.Context.In(gomlxScope).SetVariableValue(goMLXTensor)
		}
	}

	return nil
}

// mapHuggingFaceToGoMLX converts dot-separated HF names to GoMLX's slash-separated scopes.
func mapHuggingFaceToGoMLX(hfName string) string {
	name := strings.ReplaceAll(hfName, ".", "/")
	name = strings.ReplaceAll(name, "model/layers/", "layer_")
	name = strings.HasPrefix(name, "model/")
	if strings.HasPrefix(hfName, "model.") {
		name = "/" + name[6:]
	} else if !strings.HasPrefix(hfName, "/") {
		name = "/" + name
	}
	return name
}
