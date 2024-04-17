package attention

import (
	"github.com/golansformer/golansformer/linalg"
	"math/rand"
	"testing"
)

func TestAttention_Attend(t *testing.T) {
	att := NewAttentionLayer(3)

	q := linalg.NewMatrix(3, 3)
	k := linalg.NewMatrix(3, 3)
	v := linalg.NewMatrix(3, 3)

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			q.Set(i, j, rand.Float64())
			k.Set(i, j, rand.Float64())
			v.Set(i, j, rand.Float64())
		}
	}

	attended := att.Attend(q, k, v)

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if attended.At(i, j) == 0 {
				t.Errorf("Expected a value different than 0")
			}
		}
	}
}

func TestMultiHeadAttention(t *testing.T) {
	mha := NewMultiHeadAttentionLayer(3, 8)

	q := linalg.NewMatrix(3, 3)
	k := linalg.NewMatrix(3, 3)
	v := linalg.NewMatrix(3, 3)

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			q.Set(i, j, rand.Float64())
			k.Set(i, j, rand.Float64())
			v.Set(i, j, rand.Float64())
		}
	}

	attended := mha.Attend(q, k, v)

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if attended.At(i, j) == 0 {
				t.Errorf("Expected a value different than 0")
			}
		}
	}
}
