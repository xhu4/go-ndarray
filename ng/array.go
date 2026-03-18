// Package ng mimics numpy.
package ng

import (
	"bytes"
	"fmt"
	"iter"
	"math"
	"reflect"
	"slices"
	"strings"
	"unsafe"

	"golang.org/x/exp/constraints"
)

type Number interface {
	Real | constraints.Complex
}

type Real interface {
	constraints.Float | constraints.Integer
}

// NDArray represents n-dimensional arrays like np.ndarray.
type NDArray[E Number] struct {
	data   []E
	shape  Shape
	stride Stride
	offset int
}

// String converts the nd array to a multi-line string.
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

// Format formats arr into buf.
func (arr NDArray[E]) Format(buf *bytes.Buffer, formatter func(E) string, indent int, indentStr string, elemSep string, arrSep string, arrStart string, arrEnd string) {
	if arr.Dim() == 0 {
		buf.WriteString(formatter(*arr.At()))
		return
	}
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

func (arr NDArray[E]) allElems(seq iter.Seq[Index]) iter.Seq2[Index, *E] {
	return func(yield func(Index, *E) bool) {
		for idx := range seq {
			if !yield(idx, arr.At(idx...)) {
				return
			}
		}
	}
}

// AllElems iterates through all elements, last dimension changes first.
func (arr NDArray[E]) AllElems() iter.Seq2[Index, *E] { return arr.allElems(arr.shape.indicesL()) }

// AllElemsF iterates through all elements, first dimension changes first.
func (arr NDArray[E]) AllElemsF() iter.Seq2[Index, *E] { return arr.allElems(arr.shape.indicesF()) }

func (arr NDArray[E]) Assign(other NDArray[E]) error {
	if !arr.shape.Equal(other.shape) {
		return fmt.Errorf("assign: shape mismatch (%v <- %v)", arr.shape, other.shape)
	}

	if arr.stride.Equal(other.stride) && isContiguous(arr.shape, arr.stride) {
		copy(
			arr.data[arr.offset:arr.offset+arr.Size()],
			other.data[other.offset:other.offset+other.Size()],
		)
		return nil
	}
	for idx := range arr.shape.indicesL() {
		*arr.At(idx...) = *other.At(idx...)
	}
	return nil
}

func (arr NDArray[E]) Clone() NDArray[E] {
	newArr := NewZeros[E](arr.shape...)
	err := newArr.Assign(arr)
	if err != nil {
		panic(fmt.Errorf("clone: %v", err))
	}
	return newArr
}

// Reshape reshapes the array to the given shape.
// The new shape should be compatible with the original shape. Otherwise it will return an error.
// One shape dimension can be -1, in which case the size is derived.
// A copy is made only if needed.
func (arr NDArray[E]) Reshape(shape ...int) (NDArray[E], error) {
	resolved := false
	for i, s := range shape {
		if s < 0 && s != -1 {
			return arr, fmt.Errorf("reshape: invalid shape[%v] = %v", i, s)
		} else if s == -1 {
			if resolved {
				return arr, fmt.Errorf("reshape: only one -1 is allowed, got %v", shape)
			}
			shapeSize := -Shape(shape).Size()
			arrSize := arr.Size()
			if arr.Size()%shapeSize != 0 {
				return arr, fmt.Errorf("reshape: shape %v cannot divide arr size of %v", shape, arrSize)
			}
			shape[i] = arrSize / shapeSize
		}
	}

	if isContiguousRowMajor(arr.shape, arr.stride) {
		return NDArray[E]{
			arr.data,
			shape,
			compactStride(shape),
			arr.offset,
		}, nil
	} else if isContiguousReverseRowMajor(arr.shape, arr.stride) {
		return NDArray[E]{
			arr.data,
			shape,
			negated(compactStride(shape)),
			arr.offset,
		}, nil
	} else if isContiguousColMajor(arr.shape, arr.stride) {
		return NDArray[E]{
			arr.data,
			shape,
			compactStrideColMajor(shape),
			arr.offset,
		}, nil
	} else if isContiguousReverseColMajor(arr.shape, arr.stride) {
		return NDArray[E]{
			arr.data,
			shape,
			negated(compactStrideColMajor(shape)),
			arr.offset,
		}, nil
	}
	clone := arr.Clone()
	return clone.Reshape(shape...)
}

// MustReshape reshapes arr to given shape. Panics if error happens.
func (arr NDArray[E]) MustReshape(shape ...int) NDArray[E] {
	ret, err := arr.Reshape(shape...)
	if err != nil {
		panic(err)
	}
	return ret
}

// SharesMemory returns true if lhs and rhs shares the same underlying array.
func SharesMemory[E Number](lhs NDArray[E], rhs NDArray[E]) bool {
	return unsafe.SliceData(lhs.data) == unsafe.SliceData(rhs.data)
}

func reversed(s []int) []int {
	ret := slices.Clone(s)
	slices.Reverse(ret)
	return ret
}

func negated(s []int) []int {
	ret := slices.Clone(s)
	for i := range ret {
		ret[i] = -ret[i]
	}
	return ret
}

func isContiguous(shape Shape, stride Stride) bool {
	if len(shape) != len(stride) {
		panic("shape and stride not in the same dimension")
	}
	if len(shape) == 0 {
		return true
	}
	return isContiguousRowMajor(shape, stride) ||
		isContiguousReverseRowMajor(shape, stride) ||
		isContiguousColMajor(shape, stride) ||
		isContiguousReverseColMajor(shape, stride)
}

func isContiguousRowMajor(shape Shape, stride Stride) bool {
	// Row-major contiguous layout: the last index moves fastest.
	// This implies:
	//   stride[len-1] == 1
	//   stride[i] == stride[i+1] * shape[i+1] for 0 <= i < len-1
	expect := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if stride[i] != expect {
			return false
		}
		expect *= shape[i]
	}
	return true
}

func isContiguousReverseRowMajor(shape Shape, stride Stride) bool {
	return isContiguousRowMajor(shape, negated(stride))
}

func isContiguousColMajor(shape Shape, stride Stride) bool {
	return isContiguousRowMajor(reversed(shape), reversed(negated(stride)))
}

func isContiguousReverseColMajor(shape Shape, stride Stride) bool {
	return isContiguousColMajor(shape, negated(stride))
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

func nElem[E Real](start E, end E, step E) int {
	total := end - start
	return max(0, int(math.Ceil(float64(total)/float64(step))))
}

// Arange returns a 1D array from start to (excluding) stop with adjacent elements step apart.
func Arange[E Real](start E, stop E, step E) NDArray[E] {
	size := nElem(start, stop, step)
	arr := NewZeros[E](size)
	for i := range size {
		*arr.At(i) = start + step*E(i)
	}
	return arr
}

// LinspaceStep returns (arr, step).
// arr is an evenly spaced 1D array arr of shape {num} within the interval [start, end].
// step is the step size, the difference between two adjacent elements in the array.
func LinspaceStep[E Real](start E, end E, num int) (arr NDArray[E], step E) {
	arr = NewZeros[E](num)
	step = (end - start) / E(num)
	for i := range num {
		*arr.At(i) = start + step*E(i)
	}
	return arr, step
}

// Linspace returns an evenly spaced 1D array arr of shape {num} within the interval [start, end]
func Linspace[E Real](start E, end E, num int) NDArray[E] {
	arr, _ := LinspaceStep(start, end, num)
	return arr
}

// NewScalar returns a zero-dim ndarray of a single value
func NewScalar[E Number](data E) NDArray[E] {
	array := make([]E, 1)
	array[0] = data
	return NDArray[E]{
		array, make([]int, 0), make([]int, 0), 0,
	}
}

func (arr NDArray[E]) Equal(other NDArray[E]) bool {
	if !arr.shape.Equal(other.shape) {
		return false
	}

	if arr.Dim() == 0 {
		return *arr.At() == *other.At()
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

func compactStrideColMajor(shape Shape) Stride {
	return reversed(compactStride(reversed(shape)))
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
