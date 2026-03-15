package ng

import (
	"iter"
	"slices"
)

type (
	Index  []int
	Stride []int
	Shape  []int
)

func (s Shape) Size() int {
	size := 1
	for _, n := range s {
		size *= n
	}
	return size
}

func (s Shape) Equal(other Shape) bool {
	return slices.Equal(s, other)
}

func (s Stride) Equal(other Stride) bool {
	return slices.Equal(s, other)
}

func (i Index) Equal(other Index) bool {
	return slices.Equal(i, other)
}

func (i Index) Dim() int {
	return len(i)
}

func nextIndexL(idx Index, s Shape) Index {
	for i := len(s) - 1; i >= 0; i-- {
		idx[i]++
		if idx[i] == s[i] {
			idx[i] = 0
		} else {
			return idx
		}
	}
	return nil
}

func nextIndexF(idx Index, s Shape) Index {
	for i := range len(s) {
		idx[i]++
		if idx[i] == s[i] {
			idx[i] = 0
		} else {
			return idx
		}
	}
	return nil
}

func (s Shape) indicesL() iter.Seq[Index] {
	var idx Index = make([]int, len(s))
	return func(yield func(Index) bool) {
		for idx != nil {
			if !yield(idx) {
				return
			}
			idx = nextIndexL(idx, s)
		}
	}
}

func (s Shape) indicesF() iter.Seq[Index] {
	var idx Index = make([]int, len(s))
	return func(yield func(Index) bool) {
		for idx != nil {
			if !yield(idx) {
				return
			}
			idx = nextIndexF(idx, s)
		}
	}
}
