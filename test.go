// test 파일
package main

import (
	"fmt"
	rc "golansformer/residual_connection"
)

func main() {
	// 이전 레이어와 현재 레이어의 출력 값 설정(임의의 값)
	previousOutput := []float64{1.0, 2.0, 3.0}
	currentOutput := []float64{0.5, 1.5, 2.5}

	// Residual Connection
	output := rc.ResidualConnection(previousOutput, currentOutput)

	// 결과 출력
	fmt.Println("Residual Connection 결과:", output)

}