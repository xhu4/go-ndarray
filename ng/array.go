// Package ng mimics numpy.
package ng

import (
	"bytes"
	"fmt"
	"iter"
	"reflect"
	"slices"
	"strings"

	"golang.org/x/exp/constraints"
)

type Number interface {
	constraints.Float | constraints.Integer | constraints.Complex
}

// NDArray represents n-dimensional arrays like np.ndarray.
type NDArray[E Number] struct {
	data   []E
	shape  Shape
	stride Stride
	offset int
}

func (arr NDArray[E]) String() string {
	var buf bytes.Buffer
	arr.Format(&buf, func(e E) string { return fmt.Sprint(e) }, 0, " ", " ", "\n", "[", "]")
	return buf.String()
}

// Subarrays iterate through subarrays of arr of the last n-1 dimensions.
// This is useful to recursively handle ndarrays.
func (arr NDArray[E]) Subarrays() iter.Seq2[int, NDArray[E]] {
	if arr.Dim() <= 1 {
		panic(fmt.Errorf("subarrays: cannot iterate over %d dim array", arr.Dim()))
	}
	return func(yield func(int, NDArray[E]) bool) {
		for i := range arr.shape[0] {
			if !yield(i, arr.Slice(SAt(i))) {
				return
			}
		}
	}
}

func (arr NDArray[E]) Format(buf *bytes.Buffer, formatter func(E) string, indent int, indentStr string, elemSep string, arrSep string, arrStart string, arrEnd string) {
	// arr is 1D array
	indentNewline := "\n" + strings.Repeat(indentStr, indent)
	indentPlus1Newline := indentNewline + indentStr
	arrStartIndented := strings.ReplaceAll(arrStart, "\n", indentPlus1Newline)
	arrEndIndented := strings.ReplaceAll(arrEnd, "\n", indentNewline)
	if arr.Dim() == 1 {
		elemSepIndented := strings.ReplaceAll(elemSep, "\n", indentPlus1Newline)
		buf.WriteString(arrStartIndented)
		for i := range arr.shape[0] {
			if i != 0 {
				buf.WriteString(elemSepIndented)
			}
			buf.WriteString(formatter(*arr.At(i)))
		}
		buf.WriteString(arrEndIndented)
		return
	}

	arrSepIndented := strings.ReplaceAll(arrSep, "\n", indentPlus1Newline)
	// arr is at least 2D, recursively prints the subarrays
	buf.WriteString(arrStartIndented)
	for i, subarr := range arr.Subarrays() {
		if i != 0 {
			buf.WriteString(arrSepIndented)
		}
		subarr.Format(buf, formatter, indent+1, indentStr, elemSep, arrSep, arrStart, arrEnd)
	}
	buf.WriteString(arrEndIndented)
}

// Dim returns the dimension of this array.
func (arr NDArray[E]) Dim() int {
	return len(arr.shape)
}

// At returns the pointer to the element at given index.
// Index can be negative. -1 represents the last element.
// Panics if index is out of bound.
func (arr NDArray[E]) At(i ...int) *E {
	checkIndexInBound(i, arr.shape)
	return &arr.data[arr.offset+resolveIndex(i, arr.stride, arr.shape)]
}

// Slice the ndarray. Each input corresponds to a dimension.
// Each input can be created using either
//   - S(From(i), To(j), Step(k)) to slice a dimension from i to j (exclusive) every k steps, or
//   - SAt(i) to take the i-th element of this dimension, which reduces the dimension of the array by 1.
//
// If you are familiar with numpy ndarray, here are some correspondences.
//   - arr.Slice(S(From(3), To(-1)), SAt(2)) is equivalent to arr[3:-1, 2].
//   - arr.Slice(S(Step(-1)), S(To(2))) is equivalent to arr[::-1, :2].
func (arr NDArray[E]) Slice(s ...slicer) NDArray[E] {
	if len(s) > arr.Dim() {
		panic(fmt.Errorf("slicing with %d specs on a %d-dim array", len(s), arr.Dim()))
	}
	newShape := make([]int, 0)
	newStride := make([]int, 0)
	newOffset := arr.offset
	for i, si := range s {
		shape, stride, offsetChange := si.slice(arr.shape[i], arr.stride[i])
		newOffset += offsetChange
		if shape == nil {
			continue
		}
		newStride = append(newStride, stride)
		newShape = append(newShape, *shape)
	}
	// pad S() for missing dimensions
	if len(s) < arr.Dim() {
		newStride = slices.Concat(newStride, arr.stride[len(s):arr.Dim()])
		newShape = slices.Concat(newShape, arr.shape[len(s):arr.Dim()])
	}
	return NDArray[E]{
		data:   arr.data,
		shape:  newShape,
		stride: newStride,
		offset: newOffset,
	}
}

// Shape returns the shape of this array, as []int.
func (arr NDArray[E]) Shape() Shape {
	return slices.Clone(arr.shape)
}

// Size returns the total number of elements in this array.
func (arr NDArray[E]) Size() int {
	return arr.shape.Size()
}

// NewZeros returns a new zero-valued NDArray with given shape.
func NewZeros[E Number](size ...int) NDArray[E] {
	totalSize := Shape(size).Size()
	data := make([]E, totalSize)
	return NDArray[E]{
		data, size, compactStride(size), 0,
	}
}

// New returns an NDArray using given data.
// The data needs to be slice/array of E or slice/array of slice/array of E or etc.
// The shape needs to be uniform or it will return an error.
func New[E Number](data any) (NDArray[E], error) {
	var result NDArray[E]
	shape, err := getShape(reflect.ValueOf(data))
	if err != nil {
		return result, fmt.Errorf("new array: %v", err)
	}
	result = NewZeros[E](shape...)
	flatten(reflect.ValueOf(data), result.data, result.shape)
	return result, nil
}

func MustNew[E Number](data any) NDArray[E] {
	arr, err := New[E](data)
	if err != nil {
		panic(err)
	}
	return arr
}

func (arr NDArray[E]) Equal(other NDArray[E]) bool {
	if !arr.shape.Equal(other.shape) {
		return false
	}

	if arr.Dim() == 0 {
		return true
	}

	if arr.Dim() == 1 {
		for i := range arr.shape[0] {
			if *arr.At(i) != *other.At(i) {
				return false
			}
		}
		return true
	}

	next1, stop1 := iter.Pull2(arr.Subarrays())
	next2, stop2 := iter.Pull2(other.Subarrays())
	defer stop1()
	defer stop2()

	for {
		_, sub1, ok1 := next1()
		_, sub2, ok2 := next2()
		if !ok1 || !ok2 {
			return true
		}
		if !sub1.Equal(sub2) {
			return false
		}
	}
}

func flatten[E Number](data reflect.Value, dst []E, shape Shape) {
	if cap(dst) < shape.Size() {
		panic(fmt.Errorf("flatten: dst cap %d < shape size %d", cap(dst), shape.Size()))
	}
	if len(shape) == 0 {
		return
	}

	if len(shape) == 1 {
		copy(dst[:shape[0]], data.Interface().([]E))
		return
	}

	subshape := shape[1:]
	subsize := subshape.Size()
	for i := range data.Len() {
		flatten(data.Index(i), dst[subsize*i:], subshape)
	}
}

func resolveIndex(query Index, stride Stride, shape Shape) int {
	// assume len already checked
	result := 0
	for i := range len(query) {
		q := query[i]
		if q < 0 {
			q = shape[i] + q
		}
		result += q * stride[i]
	}
	return result
}

func checkIndexInBound(query Index, shape Shape) {
	if len(query) != len(shape) {
		panic(fmt.Errorf("query (%v) indices have wrong dimension (dim = %d)", query, len(shape)))
	}
	for i := range len(query) {
		if query[i] >= shape[i] || query[i] < -shape[i] {
			panic(fmt.Errorf("query (%v) out of bound at %d-th dim, shape = %v", query, i, shape))
		}
	}
}

// Row major (C-stype), last index moves fastest
func compactStride(shape Shape) Stride {
	if shape == nil {
		return nil
	}
	if len(shape) == 0 {
		return make([]int, 0)
	}
	var stride Stride = make([]int, len(shape))
	start, step, end := len(shape)-1, -1, -1
	stride[start] = 1
	for i := start + step; i != end; i += step {
		prev := i - step
		stride[i] = stride[prev] * shape[prev]
	}
	return stride
}

func getShape(val reflect.Value) (Shape, error) {
	kind := val.Kind()
	switch kind {
	case reflect.Array, reflect.Slice:
		length := val.Len()
		if length == 0 {
			return nil, fmt.Errorf("get shape: empty input array")
		}
		var subshape Shape = nil
		for i := range length {
			thissubshape, err := getShape(val.Index(i))
			if err != nil {
				return nil, err
			}
			if subshape == nil {
				subshape = thissubshape
			} else if !subshape.Equal(thissubshape) {
				return nil, fmt.Errorf("get shape: not square")
			}
		}
		return append([]int{length}, subshape...), nil
	default:
		return make([]int, 0), nil
	}
}
