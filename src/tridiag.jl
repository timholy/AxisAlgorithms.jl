"""
`A_ldiv_B_md!(dest, F, src, dim)` solves a tridiagonal system along dimension `dim` of `src`,
storing the result in `dest`. Currently, `F` must be an LU-factorized tridiagonal matrix.
If desired, you may safely use the same array for both `src` and `dest`, so that this becomes an
in-place algorithm.
"""
function A_ldiv_B_md!(dest, F, src, dim::Integer)
    1 <= dim <= max(ndims(dest),ndims(src)) || throw(DimensionMismatch("The chosen dimension $dim is larger than $(ndims(src)) and $(ndims(dest))"))
    n = size(F, 1)
    n == size(src, dim) && n == size(dest, dim) || throw(DimensionMismatch("Sizes $n, $(size(src,dim)), and $(size(dest,dim)) do not match"))
    size(dest) == size(src) || throw(DimensionMismatch("Sizes $(size(dest)), $(size(src)) do not match"))
    check_matrix(F)
    R1 = CartesianIndices(size(dest)[1:dim-1])
    R2 = CartesianIndices(size(dest)[dim+1:end])
    _A_ldiv_B_md!(dest, F, src, R1, R2)
end
_A_ldiv_B_md(F, src, R1::CartesianIndices, R2::CartesianIndices) =
    _A_ldiv_B_md!(similar(src, promote_type(eltype(F), eltype(src))), F, src, R1, R2)

# Solving along the first dimension
function _A_ldiv_B_md!(dest, F::LU{T,<:Tridiagonal{T}}, src,  R1::CartesianIndices{<:CartesianIndex{0}}, R2::CartesianIndices) where {T}
    n = size(F, 1)
    dl = F.factors.dl
    d  = F.factors.d
    du = F.factors.du
    # Forward substitution
    @inbounds for I2 in R2
        dest[1, I2] = src[1, I2]
        for i = 2:n       # note: cannot use @simd here!
            dest[i, I2] = src[i, I2] - dl[i-1]*dest[i-1, I2]
        end
    end
    # Backward substitution
    dinv = 1 ./ d
    @inbounds for I2 in R2
        dest[n, I2] /= d[n]
        for i = n-1:-1:1  # note: cannot use @simd here!
            dest[i, I2] = (dest[i, I2] - du[i]*dest[i+1, I2])*dinv[i]
        end
    end
    dest
end

# Solving along any other dimension
function _A_ldiv_B_md!(dest, F::LU{T,<:Tridiagonal{T}}, src, R1::CartesianIndices, R2::CartesianIndices) where {T}
    n = size(F, 1)
    dl = F.factors.dl
    d  = F.factors.d
    du = F.factors.du
    # Forward substitution
    @inbounds for I2 in R2
        @simd for I1 in R1
            dest[I1, 1, I2] = src[I1, 1, I2]
        end
        for i = 2:n
            @simd for I1 in R1
                dest[I1, i, I2] = src[I1, i, I2] - dl[i-1]*dest[I1, i-1, I2]
            end
        end
    end
    # Backward substitution
    dinv = 1 ./ d
    for I2 in R2
        @simd for I1 in R1
            dest[I1, n, I2] *= dinv[n]
        end
        for i = n-1:-1:1
            @simd for I1 in R1
                dest[I1, i, I2] = (dest[I1, i, I2] - du[i]*dest[I1, i+1, I2])*dinv[i]
            end
        end
    end
    dest
end

function check_matrix(F::LU{T,<:Tridiagonal{T}}) where {T}
    n = size(F,1)
    for i = 1:n
        F.ipiv[i] == i || error("For efficiency, pivoting is not supported")
    end
    nothing
end
