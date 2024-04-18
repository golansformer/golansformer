package embedding

import (
	"math/rand"
)

// Embedding 구조체 정의
type Embedding struct {
	Embeddings [][]float32
}

// NewEmbedding 함수는 n_input_vocab와 d_model을 받아 초기화된 Embedding 구조체를 반환
// n_input_vocab : 임베딩을 할 단어들의 개수. d_model : 임베딩 할 벡터의 차원
func NewEmbedding(n_input_vocab, d_model int) *Embedding {
	embeddings := make([][]float32, n_input_vocab)
	for i := range embeddings {
		embeddings[i] = make([]float32, d_model)
		for j := range embeddings[i] {
			embeddings[i][j] = rand.Float32() // 임의의 값으로 초기화
		}
	}
	return &Embedding{Embeddings: embeddings}
}
