package embedding

import (
	"fmt"
	"testing"
)

// TestNewEmbedding 함수는 NewEmbedding이 올바르게 임베딩 구조체를 초기화하는지 테스트
func TestNewEmbedding(t *testing.T) {
	n_input_vocab := 10
	d_model := 3

	emb := NewEmbedding(n_input_vocab, d_model)
	if emb == nil {
		t.Errorf("NewEmbedding returned nil")
	}

	if len(emb.Embeddings) != n_input_vocab {
		t.Errorf("Expected %d embeddings, got %d", n_input_vocab, len(emb.Embeddings))
	}

	// 전체 임베딩 배열을 한 번에 출력
	fmt.Printf("Complete Embedding array: %v\n", emb.Embeddings)

	for i, embedding := range emb.Embeddings {
		if len(embedding) != d_model {
			t.Errorf("Embedding %d expected dimension %d, got %d", i, d_model, len(embedding))
		}
		for _, value := range embedding {
			if value < 0 || value > 1 {
				t.Errorf("Embedding value %f out of expected range [0,1]", value)
			}
		}
	}
}
