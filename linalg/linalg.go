package linalg

import (
	"math"
)

type Matrix struct {
	rows       int
	cols       int
	data       []float64
	transposed bool
}

func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		rows: rows,
		cols: cols,
		data: make([]float64, rows*cols),
	}
}

func (t *Matrix) At(i, j int) float64 {
	if t.transposed {
		i, j = j, i
	}

	return t.data[i*t.cols+j]
}

func (t *Matrix) Set(i, j int, value float64) {
	if t.transposed {
		i, j = j, i
	}

	t.data[i*t.cols+j] = value
}

func (t *Matrix) T() *Matrix {
	return &Matrix{
		rows:       t.rows,
		cols:       t.cols,
		data:       t.data,
		transposed: !t.transposed,
	}
}

func (t *Matrix) Shape() (int, int) {
	if t.transposed {
		return t.cols, t.rows
	}
	return t.rows, t.cols
}

func (t *Matrix) Row(i int) *Vector {
	row := NewVector(t.cols)

	for j := 0; j < t.cols; j++ {
		row.Set(j, t.At(i, j))
	}

	return row
}

func (t *Matrix) SetRow(i int, row *Vector) {
	for j := 0; j < t.cols; j++ {
		t.Set(i, j, row.At(j))
	}
}

type Vector struct {
	size int
	data []float64
}

func NewVector(size int) *Vector {
	return &Vector{
		size: size,
		data: make([]float64, size),
	}
}

func (v *Vector) At(i int) float64 {
	return v.data[i]
}

func (v *Vector) Set(i int, value float64) {
	v.data[i] = value
}

func Softmax(x *Vector) *Vector {
	var _max float64 // Overflow prevention

	for i := 0; i < x.size; i++ {
		if x.At(i) > _max {
			_max = x.At(i)
		}
	}

	var sum float64
	softmax := make([]float64, x.size)

	for i := 0; i < x.size; i++ {
		softmax[i] = math.Exp(x.At(i) - _max)
		sum += softmax[i]
	}

	for i := 0; i < x.size; i++ {
		softmax[i] /= sum
	}

	return &Vector{size: x.size, data: softmax}
}

func Dot(x, y *Matrix) *Matrix {
	if x.cols != y.rows {
		panic("invalid matrix dimensions")
	}

	dot := NewMatrix(x.rows, y.cols)

	for i := 0; i < x.rows; i++ {
		for j := 0; j < y.cols; j++ {
			for k := 0; k < x.cols; k++ {
				dot.Set(i, j, dot.At(i, j)+x.At(i, k)*y.At(k, j))
			}
		}
	}

	return dot
}

func Concatenate(vectors ...*Matrix) *Matrix {
	rows := vectors[0].rows
	cols := 0

	for _, v := range vectors {
		if v.rows != rows {
			panic("invalid matrix dimensions")
		}

		cols += v.cols
	}

	concatenated := NewMatrix(rows, cols)

	var j int

	for _, v := range vectors {
		for i := 0; i < v.cols; i++ {
			for k := 0; k < v.rows; k++ {
				concatenated.Set(k, j, v.At(k, i))
			}

			j++
		}
	}

	return concatenated
}
