package attention

import (
	"github.com/golansformer/golansformer/linalg"
)

type Attention struct {
	D int // Dimension
}

func NewAttentionLayer(d int) *Attention {
	return &Attention{D: d}
}

func (a *Attention) Attend(q, k, v *linalg.Matrix) *linalg.Matrix {
	scores := linalg.Dot(q, k.T())
	r, c := scores.Shape()

	probabilities := linalg.NewMatrix(r, c)

	for i := 0; i < r; i++ {
		probabilities.SetRow(i, linalg.Softmax(scores.Row(i)))
	}

	attended := linalg.Dot(probabilities, v)

	return attended
}

type MultiHeadAttention struct {
	Heads      int
	Attentions []*Attention
}

func NewMultiHeadAttentionLayer(d int, heads int) *MultiHeadAttention {
	attentions := make([]*Attention, heads)

	for i := 0; i < heads; i++ {
		attentions[i] = NewAttentionLayer(d)
	}

	return &MultiHeadAttention{Heads: heads, Attentions: attentions}
}

func (m *MultiHeadAttention) Attend(q, k, v *linalg.Matrix) *linalg.Matrix {
	heads := make([]*linalg.Matrix, m.Heads)

	for i := 0; i < m.Heads; i++ {
		heads[i] = m.Attentions[i].Attend(q, k, v)
	}

	concatenated := linalg.Concatenate(heads...)

	return concatenated
}
