module FilterMD

export A_ldiv_B_md!,
    A_ldiv_B_md,
    A_mul_B_md!,
    A_mul_B_md,
    A_mul_B_perm!,
    A_mul_B_perm

@doc """
`A_ldiv_B_md(F, src, dim)` solves a tridiagonal system along dimension `dim` of `src`,
storing the result along the same dimension of the output.
Currently, `F` must be an LU-factorized tridiagonal matrix.
""" ->
A_ldiv_B_md(F, src, dim::Integer) = A_ldiv_B_md!(similar(src), F, src, dim)

@doc """
`A_mul_B_md(M, src, dim)` computes `M*x` for slices `x` of `src` along dimension `dim`,
storing the resulting vector along the same dimension of the output.
`M` must be an `AbstractMatrix`. This uses an in-place naive algorithm.
""" ->
A_mul_B_md(M::AbstractMatrix, src, dim::Integer) = A_mul_B_md!(alloc_matmul(M,src,dim), M, src, dim)

@doc """
`A_mul_B_perm(M, src, dim)` computes `M*x` for slices `x` of `src` along dimension `dim`, storing the
resulting vector along the same dimension of the output.
`M` must be an `AbstractMatrix`. This uses `permutedims` to make dimension
`dim` into the first dimension, performs a standard matrix multiplication, and restores the original
dimension ordering. In many cases, this algorithm exhibits the best cache behavior.
""" ->
A_mul_B_perm(M::AbstractMatrix, src, dim::Integer) = A_mul_B_perm!(alloc_matmul(M,src,dim), M, src, dim)

function alloc_matmul(M,src,dim)
    sz = [size(src)...]
    sz[dim] = size(M,1)
    Array(promote_type(eltype(M), eltype(src)), sz...)
end

include("tridiag.jl")
include("matmul.jl")

end # module
