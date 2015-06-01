d  = 2+rand(5)
dl = rand(4)
du = rand(4)
M = Tridiagonal(dl, d, du)
F = lufact(M)
src = rand(5,5,5)
for dim = 1:3
    dest1 = mapslices(x->A_ldiv_B!(F, x), copy(src), dim)
    dest2 = similar(src)
    AxisAlgorithms.A_ldiv_B_md!(dest2, F, src, dim)
    @test_approx_eq dest1 dest2
end
