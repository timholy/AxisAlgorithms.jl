n = 5
m = 3
src = rand(n,n,n)
M = rand(m,n)
for dim = 1:3
    dest1 = mapslices(b->M*b, src, dim)
    sz = fill(n,3)
    sz[dim] = m
    dest2 = rand(sz...)
    FilterMD.A_mul_B_md!(dest2, M, src, dim)
    @test_approx_eq dest1 dest2
    rand!(dest2)
    FilterMD.A_mul_B_perm!(dest2, M, src, dim)
    @test_approx_eq dest1 dest2
end

# Test size-checking
for dim = 1:3
    dest1 = mapslices(b->M*b, src, dim)
    sz = fill(n,3)
    sz[dim] = m+1
    dest2 = rand(sz...)
    @test_throws DimensionMismatch FilterMD.A_mul_B_md!(dest2, M, src, dim)
    @test_throws DimensionMismatch FilterMD.A_mul_B_perm!(dest2, M, src, dim)
    sz[dim] = m
    sz[mod1(dim+1,3)] = 1
    dest2 = rand(sz...)
    @test_throws DimensionMismatch FilterMD.A_mul_B_md!(dest2, M, src, dim)
    @test_throws DimensionMismatch FilterMD.A_mul_B_perm!(dest2, M, src, dim)
end
