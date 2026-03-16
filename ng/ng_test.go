package ng

import (
	"bytes"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"testing"
)

func assertPanic(t *testing.T, f func(), what string) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%v did not panic.", what)
		}
	}()
	f()
}

func TestAt(t *testing.T) {
	x := MustNew[int]([][]int{{1, 2}, {3, 4}})
	assertPanic(t, func() { x.At(1) }, "x.At(1)")
	assertPanic(t, func() { x.At(1, 2, 0) }, "x.At(1, 2, 0)")
	assertPanic(t, func() { x.At(3, 2) }, "x.At(3, 2)")
}

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
	mat, err := New[float64](data)
	if err != nil {
		t.Errorf("New() error: %v", err)
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
	{
		a, _ := New[int]([][][]int{
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

	{
		tests := []struct {
			arr  NDArray[int]
			want string
		}{
			{
				arr:  NewScalar(5),
				want: "5",
			}, {
				arr:  MustNew[int]([]int{-1}),
				want: "[-1]",
			}, {
				arr:  MustNew[int]([][][]int{{{0}}}),
				want: "[[[0]]]",
			},
		}
		for _, test := range tests {
			if got := test.arr.String(); got != test.want {
				t.Errorf("%#v = %[1]v", test.arr)
			}
		}
	}
}

func TestEqual(t *testing.T) {
	if !NewScalar(5).Equal(NewScalar(5)) {
		t.Errorf("5 != 5")
	}
	if NewScalar(5).Equal(NewScalar(6)) {
		t.Errorf("5 == 6")
	}
	if NewScalar(5).Equal(MustNew[int]([]int{5})) {
		t.Errorf("5 == [5]")
	}
}

func TestSlicing(t *testing.T) {
	a := MustNew[int]([][][]int{
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

	tests := []struct {
		slices []slicer
		want   NDArray[int]
	}{
		{
			[]slicer{SAt(0)},
			MustNew[int]([][]int{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}),
		}, {
			[]slicer{S(), SAt(1), S(From(3), Step(-2))},
			MustNew[int]([][]int{{9, 7}, {19, 17}, {29, 27}}),
		}, {
			[]slicer{S(From(2)), SAt(1), S(To(-2), Step(-1))},
			MustNew[int]([][]int{{30}}),
		},
	}

	for _, test := range tests {
		got := a.Slice(test.slices...)
		if !got.Equal(test.want) {
			t.Errorf("a.Slice%v = \n%v, want \n%v", test.slices, got, test.want)
		}
	}
}

func TestAllElems(t *testing.T) {
	a := MustNew[int]([][]int{{1, 2, 3}, {4, 5, 6}})

	t.Run("AllElemsL", func(t *testing.T) {
		wantIndices := []Index{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}}
		wantVals := []int{1, 2, 3, 4, 5, 6}
		var gotIndices []Index
		var gotVals []int
		for idx, p := range a.AllElemsL() {
			gotIndices = append(gotIndices, slices.Clone(idx))
			gotVals = append(gotVals, *p)
		}
		if !reflect.DeepEqual(gotIndices, wantIndices) {
			t.Errorf("AllElemsL indices = %v, want %v", gotIndices, wantIndices)
		}
		if !slices.Equal(gotVals, wantVals) {
			t.Errorf("AllElemsL values = %v, want %v", gotVals, wantVals)
		}
	})

	t.Run("AllElemsF", func(t *testing.T) {
		wantIndices := []Index{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}}
		wantVals := []int{1, 4, 2, 5, 3, 6}
		var gotIndices []Index
		var gotVals []int
		for idx, p := range a.AllElemsF() {
			gotIndices = append(gotIndices, slices.Clone(idx))
			gotVals = append(gotVals, *p)
		}
		if !reflect.DeepEqual(gotIndices, wantIndices) {
			t.Errorf("AllElemsF indices = %v, want %v", gotIndices, wantIndices)
		}
		if !slices.Equal(gotVals, wantVals) {
			t.Errorf("AllElemsF values = %v, want %v", gotVals, wantVals)
		}
	})
}

func TestAssign(t *testing.T) {
	src := MustNew[int]([][]int{{1, 2}, {3, 4}})
	dst := NewZeros[int](2, 2)
	if err := dst.Assign(src); err != nil {
		t.Fatalf("Assign error: %v", err)
	}
	if !dst.Equal(src) {
		t.Errorf("after Assign: dst = %v, want %v", dst, src)
	}

	// shape mismatch
	other := NewZeros[int](3, 2)
	if err := dst.Assign(other); err == nil {
		t.Errorf("Assign with shape mismatch should return an error")
	}

	// assign non-contiguous (sliced) src
	big := MustNew[int]([][]int{{10, 20, 30}, {40, 50, 60}})
	col := big.Slice(S(), SAt(1)) // [20, 50]
	dst1 := NewZeros[int](2)
	if err := dst1.Assign(col); err != nil {
		t.Fatalf("Assign non-contiguous error: %v", err)
	}
	if !dst1.Equal(MustNew[int]([]int{20, 50})) {
		t.Errorf("Assign non-contiguous: got %v", dst1)
	}
}

func TestClone(t *testing.T) {
	src := MustNew[int]([][]int{{1, 2}, {3, 4}})
	clone := src.Clone()
	if !clone.Equal(src) {
		t.Errorf("Clone() = %v, want %v", clone, src)
	}
	// mutating clone must not affect src
	*clone.At(0, 0) = 99
	if *src.At(0, 0) == 99 {
		t.Errorf("Clone() shares data with original")
	}
}

// TestAssignFastPathSourceOffset tests Assign when the source is a 1-D view
// with a non-zero offset.  The fast-path copy uses
//
//	copy(arr.data[arr.offset : arr.Size()], other.data[other.offset : other.Size()])
//
// but arr.Size() / other.Size() are element counts, not absolute end indices.
// When other.offset != 0, the end index must be other.offset+other.Size();
// as written it silently copies the wrong (too-short) region.
func TestAssignFastPathSourceOffset(t *testing.T) {
	src := MustNew[int]([]int{10, 20, 30, 40, 50})
	view := src.Slice(S(From(2))) // [30 40 50], offset=2, stride=[1]
	dst := NewZeros[int](3)       // offset=0, stride=[1] → triggers fast path
	if err := dst.Assign(view); err != nil {
		t.Fatalf("Assign error: %v", err)
	}
	want := MustNew[int]([]int{30, 40, 50})
	if !dst.Equal(want) {
		t.Errorf("Assign (offset source): got %v, want %v", dst, want)
	}
}

// TestAssignFastPathDestOffset tests Assign when the destination is a 1-D view
// with a non-zero offset that exceeds Size().  The buggy fast path computes
// arr.data[arr.offset : arr.Size()], which panics when offset > Size().
func TestAssignFastPathDestOffset(t *testing.T) {
	buf := NewZeros[int](10)
	dst := buf.Slice(S(From(5), To(8))) // offset=5, shape=[3], stride=[1]
	src := MustNew[int]([]int{1, 2, 3})

	panicked := func() (p any) {
		defer func() { p = recover() }()
		_ = dst.Assign(src)
		return nil
	}()
	if panicked != nil {
		t.Fatalf("Assign panicked: %v (slice bounds bug: arr.data[offset:Size()] when offset > Size())", panicked)
	}
	for i, want := range []int{1, 2, 3} {
		if got := *buf.At(5 + i); got != want {
			t.Errorf("buf[%d] = %d, want %d", 5+i, got, want)
		}
	}
}

// TestAssignFastPath2DOffset tests Assign for a 2-D view that shares the same
// stride as a compact array but has a non-zero offset.
//
// Currently isContiguousRowMajor incorrectly checks stride[0]==1 (column-major)
// instead of stride[last]==1 (row-major), so isContiguous returns false for
// standard 2-D arrays and the fast path is never taken — masking Bug 1 for 2-D.
// This test will start failing once isContiguous is fixed to correctly identify
// row-major arrays, unless the slice-bounds bug in the fast path is also fixed.
func TestAssignFastPath2DOffset(t *testing.T) {
	big := MustNew[int]([][]int{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
	}) // shape=[4,3], compact stride=[3,1]
	view := big.Slice(S(From(1), To(3))) // rows 1–2: [[4 5 6][7 8 9]], offset=3, stride=[3,1]
	dst := NewZeros[int](2, 3)           // stride=[3,1] — same as view
	if err := dst.Assign(view); err != nil {
		t.Fatalf("Assign error: %v", err)
	}
	want := MustNew[int]([][]int{{4, 5, 6}, {7, 8, 9}})
	if !dst.Equal(want) {
		t.Errorf("Assign 2-D offset view: got %v, want %v", dst, want)
	}
}

// TestIndicesNoAliasing verifies that each Index yielded by indicesL / indicesF
// is independent so callers can safely store it without cloning.
//
// The current implementation reuses and mutates a single idx slice: every entry
// appended without cloning aliases the same backing array.  After the iterator
// finishes, nextIndex* resets the array to all-zeros, so all stored entries
// appear as [0 0 …] rather than their original values.  The test detects this
// by checking the last collected index, which should be the shape's last index
// ([1 2] for shape [2 3]) but reads [0 0] when aliasing is present.
func TestIndicesNoAliasing(t *testing.T) {
	s := Shape{2, 3}

	t.Run("indicesL", func(t *testing.T) {
		var collected []Index
		for idx := range s.indicesL() {
			collected = append(collected, idx) // no clone — intentional
		}
		// Last index must still be [1 2]; aliasing leaves it as [0 0] (post-loop reset).
		wantLast := Index{1, 2}
		last := collected[len(collected)-1]
		if !last.Equal(wantLast) {
			t.Errorf("collected[last] = %v, want %v (aliasing: all entries share the same backing slice)", last, wantLast)
		}
	})

	t.Run("indicesF", func(t *testing.T) {
		var collected []Index
		for idx := range s.indicesF() {
			collected = append(collected, idx) // no clone — intentional
		}
		// For first-dim-fastest order the last index is also [1 2].
		wantLast := Index{1, 2}
		last := collected[len(collected)-1]
		if !last.Equal(wantLast) {
			t.Errorf("collected[last] = %v, want %v (aliasing: all entries share the same backing slice)", last, wantLast)
		}
	})
}

func ExampleNDArray() {
	a := MustNew[int]([][][]int{
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
	s := a.Slice(SAt(1), S(Step(-1)))
	fmt.Printf("%v", s)
	// Output:
	// [[16 17 18 19 20]
	//  [11 12 13 14 15]]
}
