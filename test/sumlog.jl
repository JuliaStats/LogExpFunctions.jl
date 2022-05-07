@testset "sumlog" begin
    @testset for T in [Float16, Float32, Float64, BigFloat]
        for x in (
                T[1,2,3], 
                10 .* rand(T, 1000),
                fill(nextfloat(T(1.0)), 1000),
                fill(prevfloat(T(2.0)), 1000),
            )
            @test sumlog(x) isa T
            
            @test (@inferred sumlog(x)) ≈ sum(log, x)

            y = @view x[1:min(end, 100)]
            @test (@inferred sumlog(y')) ≈ sum(log, y)

            tup = tuple(y...)
            @test (@inferred sumlog(tup)) ≈ sum(log, tup)
            #
            # gen = (sqrt(a) for a in y)
            # # `eltype` of a `Base.Generator` returns `Any`
            # @test_broken (@inferred sumlog(gen)) ≈ sum(log, gen)

            # nt = NamedTuple{tuple(Symbol.(1:100)...)}(tup)
            # @test (@inferred sumlog(y)) ≈ sum(log, y)

            z = x .+ im .* Random.shuffle(x)
            @test (@inferred sumlog(z)) ≈ sum(log, z)
        end

        # With dims
        m = 1 .+ rand(T, 10, 10)
        sumlog(m; dims=1) ≈ sum(log, m; dims=1)
        sumlog(m; dims=2) ≈ sum(log, m; dims=2)
        
        # Iterator
        @test sumlog(x^2 for x in m) ≈ sumlog(abs2, m) ≈ sumlog(*, m, m) ≈ sum(log.(m.^2))
        @test sumlog(x for x in Any[1, 2, 3+im, 4]) ≈ sum(log, Any[1, 2, 3+im, 4])
        
        # NaN, Inf
        if T != BigFloat  # exponent fails here
            @test isnan(sumlog(T[1, 2, NaN]))
            @test isinf(sumlog(T[1, 2, Inf]))
            @test sumlog(T[1, 2, 0.0]) == -Inf
            @test sumlog(T[1, 2, -0.0]) == -Inf
        end
        
        # Empty
        @test sumlog(T[]) isa T
        @test eltype(sumlog(T[]; dims=1)) == T
        @test sumlog(x for x in T[]) isa T

        # Negative
        @test_throws DomainError sumlog(T[1, -2, 3])  # easy
        @test_throws DomainError sumlog(T[1, -2, -3]) # harder

    end
    @testset "Int" begin
        @test sumlog([1,2,3]) isa Float64
        @test sumlog([1,2,3]) ≈ sum(log, [1,2,3])
        @test sumlog([1 2; 3 4]; dims=1) ≈ sum(log, [1 2; 3 4]; dims=1)
        @test sumlog(Int(x) for x in Float64[1,2,3]) ≈ sum(log, [1,2,3])
    end
end