isdefined(Base, :__precompile__) && __precompile__()

module AxisAlgorithms

using WoodburyMatrices
using Compat

export A_ldiv_B_md!,
    A_ldiv_B_md,
    A_mul_B_md!,
    A_mul_B_md,
    A_mul_B_perm!,
    A_mul_B_perm

@doc """
`A_ldiv_B_md(F, src, dim)` solves `F\b` for slices `b` of `src` along dimension `dim`,
storing the result along the same dimension of the output.
Currently, `F` must be an LU-factorized tridiagonal matrix or a Woodbury matrix.
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

function alloc_matmul{S,N}(M,src::AbstractArray{S,N},dim)
    sz = [size(src)...]
    sz[dim] = size(M,1)
    T = Base.promote_op(@functorize(*), eltype(M), S) # functorize macro used to support julia 0.4
    Array(T, sz...)::Array{T,N}
end

include("tridiag.jl")
include("matmul.jl")
include("woodbury.jl")

end # module
