package ng

import (
	"bytes"
	"strconv"
)

// Slice represents python-like slicing
// [i:j:k] -> {Begin:1, End:j, Step:k}
// [i:] -> {Begin:i, End:nil}
// Unlike Go, End can be negative or nil
// End of -1 means the last element, nil means the element after the last element.
// The End is exclusive
// When Step == 0, it is a singular slice meaning a single item. The dimension of the array will be colapsed in this case.
// So Slice{1, 1, 2} and Slice{1, 0, 2} can lead to different behaviors.
type Slice struct {
	Begin, End *int
	Step       int
}

type sliceResolved struct {
	begin, step, nstep int
}

func (s Slice) String() string {
	if s.Step == 0 {
		// Singular case. s.Begin cannot be nil
		return strconv.Itoa(*s.Begin)
	}
	itoa := func(i *int) string {
		if i == nil {
			return ""
		}
		return strconv.Itoa(*i)
	}
	var buf bytes.Buffer
	buf.WriteString(itoa(s.Begin))
	buf.WriteByte(':')
	buf.WriteString(itoa(s.End))
	if s.Step != 1 {
		buf.WriteByte(':')
		buf.WriteString(strconv.Itoa(s.Step))
	}
	return buf.String()
}

// SAll returns a Slice representing all elements, starting from 0 to the end.
func SAll() Slice {
	return Slice{nil, nil, 1}
}

// S2 returns a Slice with step size 1. Set end to nil to represents the end.
// Note the end is copied into the returning Slice.
func S2(begin *int, end *int) Slice {
	return S3(deepcopy(begin), deepcopy(end), 1)
}

// S3 returns a Slice.
// Note the end is copied into the returning Slice.
func S3(begin, end *int, step int) Slice {
	if step == 0 {
		panic("S3: step cannot be 0 (use S1 for singular case)")
	}
	return Slice{deepcopy(begin), deepcopy(end), step}
}

// S1 represents a singular Slice corresponding to a single index in numpy slicing.
func S1(i int) Slice {
	return Slice{deepcopy(&i), deepcopy(&i), 0}
}

func (s Slice) IsSingular() bool {
	return s.Step == 0
}

func (s Slice) Resolve(length int) sliceResolved {
	tBeginFEnd := map[bool]int{
		true:  0,
		false: length - 1,
	}
	var inclusive sliceResolved
	if s.Begin == nil {
		inclusive.begin = tBeginFEnd[s.Step >= 0]
	} else if *s.Begin < 0 {
		inclusive.begin = length + *s.Begin
	} else {
		inclusive.begin = *s.Begin
	}

	if s.Step == 0 {
		return inclusive
	}

	var endInclusive int
	if s.End == nil {
		endInclusive = tBeginFEnd[s.Step < 0]
	} else if *s.End < 0 {
		endInclusive = length + *s.End
		if s.Step > 0 {
			endInclusive--
		} else if s.Step < 0 {
			endInclusive++
		}
	} else {
		endInclusive = *s.End
	}
	inclusive.nstep = (endInclusive-inclusive.begin)/s.Step + 1
	inclusive.step = s.Step
	return inclusive
}

func deepcopy(i *int) *int {
	if i == nil {
		return nil
	}
	copy := *i
	return &copy
}
