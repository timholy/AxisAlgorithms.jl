# FilterMD

[![Build Status](https://travis-ci.org/timholy/FilterMD.jl.svg?branch=master)](https://travis-ci.org/timholy/FilterMD.jl)
[![Coverage Status](https://coveralls.io/repos/timholy/FilterMD.jl/badge.svg)](https://coveralls.io/r/timholy/FilterMD.jl)

FilterMD is a collection of filtering and linear algebra algorithms for multidimensional arrays.
For algorithms that would typically apply along the columns of a matrix, you can instead pick an arbitrary dimension.

Note that all functions come in two variants, a `!` version that uses pre-allocated output (where the output is
the first argument) and a version that allocates the output. Below, the `!` versions will be described.

### Tridiagonal inversion

If `F` is an LU-factorization of a tridiagonal matrix, then `A_ldiv_B_md!(dest, F, src, dim)` will solve the tridiagonal system
along dimension `dim`.
This is an in-place algorithm, and it has excellent cache behavior.

### Matrix multiplication

Multiply a matrix `M` to all 1-dimensional slices along a particular dimension.
Here you have two algorithms to choose from:

- `A_mul_B_perm!(dest, M, src, dim)` uses `permutedims` and standard BLAS-accelerated routines; it allocates temporary storage.
- `A_mul_B_md!(dest, M, src, dim)` is an in-place naive routine that does not allocate memory

In general it is very difficult to get efficient cache behavior for multidimensional multiplication, and often using `permutedims` is the best strategy.
However, there are cases where in-place operations are faster.
It's a good idea to time both and see which works better for your case.

There are optimized implementations for when `M` is a sparse matrix and when it is a 2x2 matrix.
