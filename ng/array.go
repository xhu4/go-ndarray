// Package ng mimics numpy.
package ng

import (
	"fmt"
	"reflect"
	"slices"

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
}

// Dim returns the dimension of this array.
func (arr NDArray[E]) Dim() int {
	return len(arr.shape)
}

// At returns the pointer to the element at given index.
// Panics if index is out of bound.
func (arr NDArray[E]) At(i ...int) *E {
	checkIndexInBound(i, arr.shape)
	return &arr.data[resolveIndex(i, arr.stride)]
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
		data, size, compactStride(size),
	}
}

// NewArray returns an NDArray using given data.
// The data needs to be slice/array of E or slice/array of slice/array of E or etc.
// The shape needs to be uniform or it will return an error.
func NewArray[E Number](data any) (NDArray[E], error) {
	var result NDArray[E]
	shape, err := getShape(reflect.ValueOf(data))
	if err != nil {
		return result, fmt.Errorf("new array: %v", err)
	}
	result = NewZeros[E](shape...)
	flatten(reflect.ValueOf(data), result.data, result.shape)
	return result, nil
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

func resolveIndex(query Index, stride Stride) int {
	// assume len already checked
	result := 0
	for i := range len(query) {
		result += query[i] * stride[i]
	}
	return result
}

func checkIndexInBound(query Index, shape Shape) {
	if len(query) != len(shape) {
		panic(fmt.Errorf("query (%v) indices have wrong dimension (dim = %d)", query, len(shape)))
	}
	for i := range len(query) {
		if query[i] >= shape[i] {
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
