@static if VERSION < v"0.7.0-DEV.481"
    const _ldiv! = isdefined(:ldiv!) ? ldiv! : Base.A_ldiv_B!
else
    const _ldiv! = (@isdefined ldiv!) ? ldiv! : Base.A_ldiv_B!
end
const _lu = @static VERSION < v"0.7.0-DEV.3449" ? Base.lufact : lu

let
    d  = 2 .+ rand(5)
    dl = rand(4)
    du = rand(4)
    M = Tridiagonal(dl, d, du)
    F = _lu(M)
    src = rand(5,5,5)
    for dim = 1:3
        dest1 = mapslices(x->_ldiv!(F, x), copy(src), dims=dim)
        dest2 = similar(src)
        AxisAlgorithms.A_ldiv_B_md!(dest2, F, src, dim)
        @test dest1 ≈ dest2
    end
end
