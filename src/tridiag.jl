import Base.LinAlg.LU, Base.getindex, Base.setindex!

@doc """
`A_ldiv_B_md!(dest, F, src, dim)` solves a tridiagonal system along dimension `dim` of `src`,
storing the result in `dest`. Currently, `F` must be an LU-factorized tridiagonal matrix.
""" ->
function A_ldiv_B_md!{T}(dest, F::LU{T,Tridiagonal{T}}, src, dim::Integer)
    1 <= dim <= max(ndims(dest),ndims(src)) || throw(DimensionMismatch("The chosen dimension $dim is larger than $(ndims(src)) and $(ndims(dest))"))
    n = size(F, 1)
    n == size(src, dim) && n == size(dest, dim) || throw(DimensionMismatch("Sizes $n, $(size(src,dim)), and $(size(dest,dim)) do not match"))
    size(dest) == size(src) || throw(DimensionMismatch("Sizes $(size(dest)), $(size(src)) do not match"))
    for i = 1:n
        F.ipiv[i] == i || error("For efficiency, pivoting is not supported")
    end
    R1 = CartesianRange(size(dest)[1:dim-1])
    R2 = CartesianRange(size(dest)[dim+1:end])
    _A_ldiv_B_md!(dest, F, src, R1, R2)
end

# Filtering along the first dimension
function _A_ldiv_B_md!{T,CI<:CartesianIndex{0}}(dest, F::LU{T,Tridiagonal{T}}, src,  R1::CartesianRange{CI}, R2)
    n = size(F, 1)
    dl = F.factors.dl
    d  = F.factors.d
    du = F.factors.du
    # Forward substitution
    @inbounds for I2 in R2
        dest[1, I2] = src[1, I2]
        for i = 2:n
            dest[i, I2] = src[i, I2] - dl[i-1]*dest[i-1, I2]
        end
    end
    # Backward substitution
    dinv = 1./d
    @inbounds for I2 in R2
        dest[n, I2] /= d[n]
        for i = n-1:-1:1
            dest[i, I2] = (dest[i, I2] - du[i]*dest[i+1, I2])*dinv[i]
        end
    end
    dest
end

# Filtering along any other dimension
function _A_ldiv_B_md!{T}(dest, F::LU{T,Tridiagonal{T}}, src, R1, R2)
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
    dinv = 1./d
    for I2 in R2
        @simd for I1 in R1
#             dest[I1, n, I2] /= d[n]
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
