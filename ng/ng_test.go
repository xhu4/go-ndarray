package ng

import (
	"reflect"
	"slices"
	"testing"
)

func TestGetShape(t *testing.T) {
	tests := []struct {
		data any
		want Shape
	}{
		{[][]struct{}{{{}, {}}, {{}, {}}}, []int{2, 2}},
		{
			[][][]float32{
				{
					{0, 0, 0},
					{0, 0, 0},
				},
				{
					{0, 0, 0},
					{0, 0, 0},
				},
			},
			[]int{2, 2, 3},
		},
	}
	for _, test := range tests {
		got, err := getShape(reflect.ValueOf(test.data))
		if err != nil {
			t.Errorf("getShape(%v) error: %v", test.data, err)
		} else if !slices.Equal(got, test.want) {
			t.Errorf("getShape(%v) = %v (want %v)", test.data, got, test.want)
		}
	}

	fails := []struct {
		data any
	}{
		{[][][]int{}},
		{[][][]int{{}, {}}},
		{
			[][][]float32{
				{
					{0, 0, 0},
					{0, 0, 0},
				},
				{
					{0, 0, 0, 1},
					{0, 0, 0, 1},
				},
			},
		},
	}
	for _, fail := range fails {
		got, err := getShape(reflect.ValueOf(fail.data))
		if err == nil {
			t.Errorf("getShape(%v) = %v", fail.data, got)
		}
	}
}

func TestNewZeros(t *testing.T) {
	data := [][]float64{{1.0, 1.1}, {2.0, 2.1}, {3.0, 3.1}}
	mat, err := NewArray[float64](data)
	if err != nil {
		t.Errorf("NewArray() error: %v", err)
	}
	for i := range 3 {
		for j := range 2 {
			if *mat.At(i, j) != data[i][j] {
				t.Errorf("mat[%d, %d] != data[%[1]d][%[2]d]", i, j)
			}
		}
	}
}
