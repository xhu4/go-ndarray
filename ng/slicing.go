package ng

import (
	"bytes"
	"fmt"
	"strconv"
)

type slicer interface {
	// Slice a single dimension of a given array.
	// If newLen is nil, the dimension is collapsed (when indexing a single dimension).
	slice(length int, stride int) (newLen *int, newStride int, offsetDiff int)
}

// evenSlicer represents python-like slicing.
//
// numpy i:j:k is equivalent to evenSlicer{begin:1, end:j, step:k}
// numpy i: is equivalent to evenSlicer{begin:i, end:nil}
// The end is exclusive. Begin and end can be negative. -1 means the last element.
// step cannot be 0
//
// Use S() to create a evenSlicer. E.g.
// - : -> S()
// - i:j:k -> S(From(i), To(j), Step(k))
// - i::k -> S(From(i), Step(k))
type evenSlicer struct {
	begin, end *int
	step       int
}

// indexSlicer represents python-like single index slicing.
// It reduces the array dimension.
//
// Use SAt(i) to create a indexSlicer.
type indexSlicer int

func nElem(begin int, end int, step int) int {
	minusOne := -1
	if step < 0 {
		minusOne = 1
	}
	return max(0, ((end-begin)+minusOne)/step+1)
}

func (s evenSlicer) slice(length int, stride int) (newLen *int, newStride int, offsetDiff int) {
	var begin, end int
	if s.begin == nil {
		if s.step >= 0 {
			begin = 0
		} else {
			begin = length - 1
		}
	} else {
		begin = *s.begin
	}
	if s.end == nil {
		if s.step >= 0 {
			end = length
		} else {
			end = -1
		}
	} else {
		end = *s.end
	}
	if begin >= length || end > length || begin < 0 || end < -1 {
		panic(fmt.Errorf("slicing out of bound: slice %v on length %v", s, length))
	}
	newLen = new(int)
	*newLen = nElem(begin, end, s.step)
	newStride = stride * s.step
	if s.step < 0 {
		offsetDiff = stride * begin
	}
	return
}

func (s indexSlicer) slice(length int, stride int) (newLen *int, newStride int, offsetDiff int) {
	if s < 0 {
		s = indexSlicer(length + int(s))
	}
	return nil, 0, int(s) * stride
}

func (s indexSlicer) String() string {
	return strconv.Itoa(int(s))
}

type sliceArg func(*evenSlicer)

func From(i int) sliceArg {
	return func(s *evenSlicer) {
		s.begin = &i
	}
}

func To(i int) sliceArg {
	return func(s *evenSlicer) {
		s.end = &i
	}
}

func Step(i int) sliceArg {
	if i == 0 {
		panic("step size 0 is not allowed. Use SAt for singular slicing")
	}
	return func(s *evenSlicer) {
		s.step = i
	}
}

func S(args ...sliceArg) evenSlicer {
	s := evenSlicer{nil, nil, 1}
	for _, arg := range args {
		arg(&s)
	}
	return s
}

func SAt(i int) indexSlicer {
	return indexSlicer(i)
}

func (s evenSlicer) String() string {
	itoa := func(i *int) string {
		if i == nil {
			return ""
		}
		return strconv.Itoa(*i)
	}
	var buf bytes.Buffer
	buf.WriteString(itoa(s.begin))
	buf.WriteByte(':')
	buf.WriteString(itoa(s.end))
	if s.step != 1 {
		buf.WriteByte(':')
		buf.WriteString(strconv.Itoa(s.step))
	}
	return buf.String()
}
