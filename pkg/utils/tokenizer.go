package utils

import (
	"fmt"

	"github.com/daulet/tokenizers"
)

// Tokenizer wraps the daulet/tokenizers.Tokenizer for easier use.
type Tokenizer struct {
	tk *tokenizers.Tokenizer
}

// NewTokenizer loads a tokenizer from a tokenizer.json file.
func NewTokenizer(path string) (*Tokenizer, error) {
	tk, err := tokenizers.FromFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer from %s: %w", path, err)
	}
	return &Tokenizer{tk: tk}, nil
}

// Encode converts text into token IDs.
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) ([]uint32, error) {
	encoding, err := t.tk.Encode(text, addSpecialTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}
	// encoding.IDs returns []uint32
	return encoding.IDs, nil
}

// Close releases the underlying tokenizer resources (important if using CGO).
func (t *Tokenizer) Close() {
	if t.tk != nil {
		t.tk.Close()
	}
}
