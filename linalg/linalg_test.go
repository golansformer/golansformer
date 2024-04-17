package linalg

import (
	"testing"
)

func TestSoftmax(t *testing.T) {
	vector := NewVector(3)

	vector.Set(0, 1)
	vector.Set(1, 2)
	vector.Set(2, 3)

	result := Softmax(vector)

	if result.At(0) != 0.09003057317038046 {
		t.Errorf("Expected 0.09003057317038046 but got %f", result.At(0))
	}
}

func TestDot(t *testing.T) {
	matrix1 := NewMatrix(2, 3)
	matrix2 := NewMatrix(3, 2)

	matrix1.Set(0, 0, 1)
	matrix1.Set(0, 1, 2)
	matrix1.Set(0, 2, 3)
	matrix1.Set(1, 0, 4)
	matrix1.Set(1, 1, 5)
	matrix1.Set(1, 2, 6)

	matrix2.Set(0, 0, 7)
	matrix2.Set(0, 1, 8)
	matrix2.Set(1, 0, 9)
	matrix2.Set(1, 1, 10)
	matrix2.Set(2, 0, 11)
	matrix2.Set(2, 1, 12)

	result := Dot(matrix1, matrix2)

	if result.At(0, 0) != 58 {
		t.Errorf("Expected 58 but got %f", result.At(0, 0))
	}

	if result.At(0, 1) != 64 {
		t.Errorf("Expected 64 but got %f", result.At(0, 1))
	}

	if result.At(1, 0) != 139 {
		t.Errorf("Expected 139 but got %f", result.At(1, 0))
	}

	if result.At(1, 1) != 154 {
		t.Errorf("Expected 154 but got %f", result.At(1, 1))
	}
}

func TestConcatenate(t *testing.T) {
	matrix1 := NewMatrix(2, 3)
	matrix2 := NewMatrix(2, 3)

	matrix1.Set(0, 0, 1)
	matrix1.Set(0, 1, 2)
	matrix1.Set(0, 2, 3)
	matrix1.Set(1, 0, 4)
	matrix1.Set(1, 1, 5)
	matrix1.Set(1, 2, 6)

	matrix2.Set(0, 0, 7)
	matrix2.Set(0, 1, 8)
	matrix2.Set(0, 2, 9)
	matrix2.Set(1, 0, 10)
	matrix2.Set(1, 1, 11)
	matrix2.Set(1, 2, 12)

	result := Concatenate(matrix1, matrix2)

	if result.At(0, 0) != 1 {
		t.Errorf("Expected 1 but got %f", result.At(0, 0))
	}

	if result.At(0, 1) != 2 {
		t.Errorf("Expected 2 but got %f", result.At(0, 1))
	}

	if result.At(0, 2) != 3 {
		t.Errorf("Expected 3 but got %f", result.At(0, 2))
	}

	if result.At(0, 3) != 7 {
		t.Errorf("Expected 7 but got %f", result.At(0, 3))
	}

	if result.At(0, 4) != 8 {
		t.Errorf("Expected 8 but got %f", result.At(0, 4))
	}

	if result.At(0, 5) != 9 {
		t.Errorf("Expected 9 but got %f", result.At(0, 5))
	}
}

func TestMatrix_At(t *testing.T) {
	matrix := NewMatrix(2, 3)

	matrix.Set(0, 0, 1)
	matrix.Set(0, 1, 2)
	matrix.Set(0, 2, 3)
	matrix.Set(1, 0, 4)
	matrix.Set(1, 1, 5)
	matrix.Set(1, 2, 6)

	if matrix.At(0, 0) != 1 {
		t.Errorf("Expected 1 but got %f", matrix.At(0, 0))
	}

	if matrix.At(0, 1) != 2 {
		t.Errorf("Expected 2 but got %f", matrix.At(0, 1))
	}

	if matrix.At(0, 2) != 3 {
		t.Errorf("Expected 3 but got %f", matrix.At(0, 2))
	}

	if matrix.At(1, 0) != 4 {
		t.Errorf("Expected 4 but got %f", matrix.At(1, 0))
	}

	if matrix.At(1, 1) != 5 {
		t.Errorf("Expected 5 but got %f", matrix.At(1, 1))
	}

	if matrix.At(1, 2) != 6 {
		t.Errorf("Expected 6 but got %f", matrix.At(1, 2))
	}
}

func TestMatrix_Set(t *testing.T) {
	matrix := NewMatrix(2, 3)

	matrix.Set(0, 0, 1)
	matrix.Set(0, 1, 2)
	matrix.Set(0, 2, 3)
	matrix.Set(1, 0, 4)
	matrix.Set(1, 1, 5)
	matrix.Set(1, 2, 6)

	if matrix.At(0, 0) != 1 {
		t.Errorf("Expected 1 but got %f", matrix.At(0, 0))
	}

	if matrix.At(0, 1) != 2 {
		t.Errorf("Expected 2 but got %f", matrix.At(0, 1))
	}

	if matrix.At(0, 2) != 3 {
		t.Errorf("Expected 3 but got %f", matrix.At(0, 2))
	}

	if matrix.At(1, 0) != 4 {
		t.Errorf("Expected 4 but got %f", matrix.At(1, 0))
	}

	if matrix.At(1, 1) != 5 {
		t.Errorf("Expected 5 but got %f", matrix.At(1, 1))
	}

	if matrix.At(1, 2) != 6 {
		t.Errorf("Expected 6 but got %f", matrix.At(1, 2))
	}
}

func TestMatrix_T(t *testing.T) {
	// 1 2 3
	// 4 5 6
	matrix := NewMatrix(2, 3)

	r, c := matrix.Shape()

	if r != 2 {
		t.Errorf("Expected 2 but got %d", r)
	}

	if c != 3 {
		t.Errorf("Expected 3 but got %d", c)
	}

	matrix.Set(0, 0, 1)
	matrix.Set(0, 1, 2)
	matrix.Set(0, 2, 3)
	matrix.Set(1, 0, 4)
	matrix.Set(1, 1, 5)
	matrix.Set(1, 2, 6)

	// 1 4
	// 2 5
	// 3 6
	transposed := matrix.T()

	if transposed.At(0, 0) != 1 {
		t.Errorf("Expected 1 but got %f", transposed.At(0, 0))
	}

	if transposed.At(0, 1) != 4 {
		t.Errorf("Expected 4 but got %f", transposed.At(0, 1))
	}

	if transposed.At(1, 0) != 2 {
		t.Errorf("Expected 2 but got %f", transposed.At(1, 0))
	}

	if transposed.At(1, 1) != 5 {
		t.Errorf("Expected 5 but got %f", transposed.At(1, 1))
	}

	if transposed.At(2, 0) != 3 {
		t.Errorf("Expected 3 but got %f", transposed.At(2, 0))
	}

	if transposed.At(2, 1) != 6 {
		t.Errorf("Expected 6 but got %f", transposed.At(2, 1))
	}

	r, c = transposed.Shape()

	if r != 3 {
		t.Errorf("Expected 3 but got %d", r)
	}

	if c != 2 {
		t.Errorf("Expected 2 but got %d", c)
	}
}
