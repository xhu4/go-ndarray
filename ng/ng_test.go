package ng

import (
	"bytes"
	"fmt"
	"reflect"
	"slices"
	"strconv"
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

func TestFormat(t *testing.T) {
	a, _ := NewArray[int]([][][]int{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{11, 12, 13},
			{14, 15, 16},
		},
		{
			{21, 22, 23},
			{24, 25, 26},
		},
	})
	tests := []struct {
		indentStr, elemSep, arrSep, arrStart, arrEnd string
		want                                         string
	}{
		{
			"X", ", ", "\n", "{", "}",
			`{{{1, 2, 3}
XX{4, 5, 6}}
X{{11, 12, 13}
XX{14, 15, 16}}
X{{21, 22, 23}
XX{24, 25, 26}}}`,
		},
		{
			" ", " ", "\n", "[\n", "\n]",
			`[
 [
  [
   1 2 3
  ]
  [
   4 5 6
  ]
 ]
 [
  [
   11 12 13
  ]
  [
   14 15 16
  ]
 ]
 [
  [
   21 22 23
  ]
  [
   24 25 26
  ]
 ]
]`,
		},
	}
	for _, test := range tests {
		var buf bytes.Buffer
		a.Format(&buf, func(i int) string { return strconv.Itoa(i) }, 0, test.indentStr, test.elemSep, test.arrSep, test.arrStart, test.arrEnd)
		got := buf.String()
		if got != test.want {
			t.Errorf("a.Format(indentStr=%s, elemSep=%s, arrSep=%s, arrStart=%s, arrEnd=%s) ==\n%s",
				test.indentStr, test.elemSep, test.arrSep, test.arrStart, test.arrEnd, got,
			)
			for i := range min(len(got), len(test.want)) {
				if got[i] != test.want[i] {
					t.Errorf("mismatch at %d, got=%c, want=%c", i, got[i], test.want[i])
				}
			}
			if len(got) != len(test.want) {
				t.Errorf("len(got)=%d, expect %d", len(got), len(test.want))
			}
		}
	}
}

func ExampleNDArray() {
	a, _ := NewArray[int]([][][]int{
		{
			{1, 2, 3, 4, 5},
			{6, 7, 8, 9, 10},
		},
		{
			{11, 12, 13, 14, 15},
			{16, 17, 18, 19, 20},
		},
		{
			{21, 22, 23, 24, 25},
			{26, 27, 28, 29, 30},
		},
	})
	s := a.Slicing(S1(1), S3(nil, nil, -1), SAll())
	fmt.Printf("%v", s)
	// Output:
	// [[16 17 18 19 20]
	//  [11 12 13 14 15]]
}
