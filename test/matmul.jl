n = 5
m = 3
src = rand(n,n,n)
for M in (rand(m,n), sprand(m,n,0.2))
    for dim = 1:3
        dest1 = mapslices(b->M*b, src, dim)
        sz = fill(n,3)
        sz[dim] = m
        dest2 = rand(sz...)
        FilterMD.A_mul_B_md!(dest2, M, src, dim)
        @test_approx_eq dest1 dest2
        if !issparse(M)
            rand!(dest2)
            FilterMD.A_mul_B_perm!(dest2, M, src, dim)
            @test_approx_eq dest1 dest2
        end
    end
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

# Test 1x1 and 2x2 cases
n = 5
for dim = 1:3
    sz = fill(n,3)
    sz[dim] = 1
    src = rand(sz...)
    M = fill(3.2,1,1)
    dest1 = mapslices(b->M*b, src, dim)
    dest2 = FilterMD.A_mul_B_md(M, src, dim)
    @test_approx_eq dest1 dest2
    sz[dim] = 2
    src = rand(sz...)
    M = rand(2,2)
    dest1 = mapslices(b->M*b, src, dim)
    dest2 = FilterMD.A_mul_B_md(M, src, dim)
    @test_approx_eq dest1 dest2
end
