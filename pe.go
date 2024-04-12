import (
	"fmt"
	"math"
)

// (max_len, d_model) 형태의 배열을 초기화하여, 위치 인코딩을 저장
type PE struct {
	pe [][]float64
}

func PositionalEncoding(d_model int, max_len int) *PE {
	// 배열 초기화 (max_len, d_model)
	pe := make([][]float64, max_len)
	for i:=range pe {
		pe[i] = make([]float64, d_model)
	}
	for pos := 0;pos < max_len;pos++{
		for i := 0;i<d_model;i++{
			div_term := math.Exp(-math.Log(10000.0) * float64(i) / float64(d_model)) // 분모 수식 구현
			if i % 2 == 0 {
				pe[pos][i] = math.Sin(float64(pos)*div_term) // 짝수 인덱스
			} else{
				pe[pos][i] = math.Cos(float64(pos)*div_term) // 홀수 인덱스
			}
		}
	}

	return &PE{pe: pe}
}

// 위치 정보가 추가된 임베딩 반환
func (pe *PE) Forward(x [][]float64) [][]float64 {
    max_len := len(pe.pe)
    for i := range x {
        for j := range x[i] {
            if i < max_len {
                x[i][j] += pe.pe[i][j]  //summation
            }
        }
    }
    return x
}
