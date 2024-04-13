package residual_connection

//Residual Connection
func ResidualConnection(previousOutput, currentOutput []float64) []float64 {
	if len(previousOutput) != len(currentOutput) {
		panic("이전 레이어와 현재 레이어의 출력 길이가 다릅니다.")
	}

	result := make([]float64, len(previousOutput))

	for i := range previousOutput {
		result[i] = previousOutput[i] + currentOutput[i]
	}

	return result
}


