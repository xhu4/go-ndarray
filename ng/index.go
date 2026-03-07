package ng

import "slices"

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

func (i Index) Equal(other Index) bool {
	return slices.Equal(i, other)
}

func (i Index) Dim() int {
	return len(i)
}
