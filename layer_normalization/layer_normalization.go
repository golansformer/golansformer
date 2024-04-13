package layer_normalization

import (
	"math"
)

func LayerNormalization(input []float64) []float64 {
    
	// 각 차원 평균
    var mean, variance float64
    for _, val := range input {
        mean += val
    }
    mean /= float64(len(input))

	// 각 차원 분산
    for _, val := range input {
        variance += math.Pow(val-mean, 2)
    }
    variance /= float64(len(input))

    // 정규화
    eps := 1e-6 // 분모가 0을 방지하기 위한 epsilon
    normalized := make([]float64, len(input))
    for i, val := range input {
        normalized[i] = (val - mean) / math.Sqrt(variance+eps)
    }

    return normalized
}