using Test: @test, @test_throws, @testset, @inferred
using Statistics: mean, std, var
using LogExpFunctions: logmeanexp, logstdexp, logvarexp,
    logmeanexp_and_logvarexp, logmeanexp_and_logstdexp

# Count heap allocations of `f(x)` after warming it up. `f` and `x` are passed as
# arguments so that they are concretely typed inside this function (avoiding spurious
# allocations from captured, boxed locals).
allocations(f, x) = (f(x); @allocated f(x))

@testset "logmeanexp, logvarexp, logstdexp arrays" begin
    for T in (Float32, Float64)
        X = randn(T, 5, 3, 2)
        for dims in (2, (1, 2), :)
            @test logmeanexp(X; dims=dims) ≈ log.(mean(exp.(X); dims=dims))
            for corrected in (true, false)
                @test logvarexp(X; dims=dims, corrected=corrected) ≈
                    log.(var(exp.(X); dims=dims, corrected=corrected))
                @test logstdexp(X; dims=dims, corrected=corrected) ≈
                    log.(std(exp.(X); dims=dims, corrected=corrected))
            end
        end
        @test @inferred(logmeanexp(X)) ≈ log(mean(exp, X))
        @test @inferred(logvarexp(X)) ≈ log(var(exp.(X)))
        @test @inferred(logstdexp(X)) ≈ log(std(exp.(X)))
    end
end

@testset "logmeanexp, logvarexp, logstdexp iterators" begin
    x = randn(Float32, 20)
    xt = Tuple(x)
    xg = (v for v in x)
    xf = Iterators.filter(_ -> true, x)
    xe = exp.(x)

    @test @inferred(logmeanexp(xt)) ≈ log(mean(exp, xt))
    @test logmeanexp(xg) ≈ log(mean(exp, x))
    @test @inferred(logmeanexp(xf)) ≈ log(mean(exp, x))
    @test logmeanexp(Iterators.Stateful(x)) ≈ log(mean(exp, x))
    @test @inferred(logvarexp(xt)) ≈ log(var(xe))
    @test logvarexp(xt; corrected=false) ≈ log(var(xe; corrected=false))
    @test @inferred(logvarexp((v for v in x))) ≈ log(var(xe))
    @test logvarexp((v for v in x); corrected=false) ≈ log(var(xe; corrected=false))
    @test logvarexp(Iterators.Stateful(x)) ≈ log(var(xe))
    @test @inferred(logstdexp(xt)) ≈ log(std(xe))
    @test logstdexp(xt; corrected=false) ≈ log(std(xe; corrected=false))
    @test @inferred(logstdexp((v for v in x))) ≈ log(std(xe))
    @test logstdexp((v for v in x); corrected=false) ≈ log(std(xe; corrected=false))
    @test logstdexp(Iterators.Stateful(x)) ≈ log(std(xe))
    @test isnan(logvarexp((0.0,)))
    @test isnan(logstdexp((0.0,)))
    @test_throws ArgumentError logvarexp((1.0 + 0.0im, 2.0 + 0.0im))
    @test_throws ArgumentError logstdexp((1.0 + 0.0im, 2.0 + 0.0im))
    @test_throws ArgumentError logmeanexp(())
    @test_throws ArgumentError logvarexp(())
end

@testset "logmeanexp, logvarexp, logstdexp promotion and dims coverage" begin
    X = randn(Float32, 5, 3, 2)

    for dims in (1, (2, 3))
        @test eltype(@inferred(logmeanexp(X; dims=dims))) == Float32
        @test eltype(@inferred(logvarexp(X; dims=dims))) == Float32
        @test eltype(@inferred(logstdexp(X; dims=dims))) == Float32
    end

    @test typeof(@inferred(logmeanexp(X; dims=:))) == Float32
    @test typeof(@inferred(logvarexp(X; dims=:))) == Float32
    @test typeof(@inferred(logstdexp(X; dims=:))) == Float32

    X1 = reshape(randn(Float64, 8), 1, 8)
    @test all(isnan, logvarexp(X1; dims=1, corrected=true))
    @test all(isnan, logstdexp(X1; dims=1, corrected=true))
    Xsingleton = fill(0.0f0, 1, 1, 1)
    @test isnan(logvarexp(Xsingleton; dims=:, corrected=true))
    @test isnan(logstdexp(Xsingleton; dims=:, corrected=true))
end

@testset "logmeanexp_and_logvarexp, logmeanexp_and_logstdexp" begin
    for T in (Float32, Float64)
        X = randn(T, 5, 3, 2)
        for dims in (2, (1, 2), :), corrected in (true, false)
            m, v = logmeanexp_and_logvarexp(X; dims=dims, corrected=corrected)
            @test m ≈ logmeanexp(X; dims=dims)
            @test v ≈ logvarexp(X; dims=dims, corrected=corrected)
            m2, s = logmeanexp_and_logstdexp(X; dims=dims, corrected=corrected)
            @test m2 ≈ logmeanexp(X; dims=dims)
            @test s ≈ logstdexp(X; dims=dims, corrected=corrected)
        end
        # results match the reference statistics directly
        @test all(logmeanexp_and_logvarexp(X) .≈ (log(mean(exp, X)), log(var(exp.(X)))))
        @test all(logmeanexp_and_logstdexp(X) .≈ (log(mean(exp, X)), log(std(exp.(X)))))
    end

    # iterators (single pass, including one-shot iterators)
    x = randn(Float32, 20)
    xt = Tuple(x)
    xe = exp.(x)
    @test all(@inferred(logmeanexp_and_logvarexp(xt)) .≈ (log(mean(exp, xt)), log(var(xe))))
    @test all(@inferred(logmeanexp_and_logstdexp(xt)) .≈ (log(mean(exp, xt)), log(std(xe))))
    @test all(logmeanexp_and_logvarexp(Iterators.Stateful(x)) .≈ (log(mean(exp, x)), log(var(xe))))
    @test all(logmeanexp_and_logstdexp(Iterators.Stateful(x)) .≈ (log(mean(exp, x)), log(std(xe))))

    # edge cases
    @test isnan(last(logmeanexp_and_logvarexp((0.0,))))
    @test isnan(last(logmeanexp_and_logstdexp((0.0,))))
    @test_throws ArgumentError logmeanexp_and_logvarexp((1.0 + 0.0im,))
    @test_throws ArgumentError logmeanexp_and_logstdexp((1.0 + 0.0im,))
end

@testset "type stability and inference" begin
    X = randn(Float32, 5, 3, 2)
    xt = Tuple(randn(Float32, 20))
    for dims in (1, (2, 3), :)
        @test @inferred(logmeanexp_and_logvarexp(X; dims=dims)) isa Tuple
        @test @inferred(logmeanexp_and_logstdexp(X; dims=dims)) isa Tuple
    end
    @test @inferred(logmeanexp_and_logvarexp(xt)) isa NTuple{2,Float32}
    @test @inferred(logmeanexp_and_logstdexp(xt)) isa NTuple{2,Float32}

    # no Float64 promotion for Float32 inputs
    m, v = logmeanexp_and_logvarexp(X; dims=2)
    @test eltype(m) == Float32 && eltype(v) == Float32
    @test typeof(@inferred(logmeanexp_and_logvarexp(X))) == Tuple{Float32,Float32}
end

@testset "allocations" begin
    for T in (Float32, Float64)
        v = randn(T, 1000)
        tup = Tuple(randn(T, 20))
        # full reductions over arrays / iterators allocate nothing
        @test allocations(logmeanexp, v) == 0
        @test allocations(logvarexp, v) == 0
        @test allocations(logstdexp, v) == 0
        @test allocations(logmeanexp_and_logvarexp, v) == 0
        @test allocations(logmeanexp_and_logstdexp, v) == 0
        @test allocations(logvarexp, tup) == 0
        @test allocations(logmeanexp_and_logvarexp, tup) == 0
    end
end
