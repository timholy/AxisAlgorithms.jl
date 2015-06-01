using WoodburyMatrices

d  = 2+rand(5)
dl = -rand(4)
du = -rand(4)
M = Tridiagonal(dl, d, du)
F = lufact(M)
U = sprand(5,2,0.2)
V = sprand(2,5,0.2)
C = rand(2,2)
W = Woodbury(F, U, C, V)

src = rand(5, 8)
@test_approx_eq W\src AxisAlgorithms.A_ldiv_B_md(W, src, 1)
src = rand(5, 5, 5)
for dim = 1:3
    dest1 = mapslices(x->W\x, copy(src), dim)
    dest2 = similar(src)
    AxisAlgorithms.A_ldiv_B_md!(dest2, W, src, dim)
    @test_approx_eq dest1 dest2
end
