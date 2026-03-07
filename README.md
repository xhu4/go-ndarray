# GoNDArray - NDArray Implementation in Go

[![CI](https://github.com/xhu4/go-ndarray/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/xhu4/go-ndarray/actions/workflows/ci.yml)

Look away. This is a personal project for fun.

If you want numerical packages in Go, checkout [gonum](https://github.com/gonum/gonum) and [gosl](https://github.com/cpmech/gosl).

I started to learn Go for fun after hearing a lot of appreciations of its simplicity. This is my first Go project to help me learn it. I chose something I'm familiar with and basic: linear algebra packages.


## Features

It's still really early to say anything about this project. But my goal is to build something similar to numpy's ndarray.

Main features I'm trying to implement for now:

- [ ] simple slicing (numpy: `A[:, 1:2:-1]`, here: `A.Slice(S.All(), S.Slice3(1,2,-1))`), that refers to the same underlying data.
- [ ] arbitrary indexing (numpy: `A[some_list_of_bool]`, here: `A.Mask([]bool)`)
- [ ] matrix multiplication
- [ ] broadcasting-enabled multiplication

