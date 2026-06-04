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
    # Full reductions allocate at most a small constant (no per-element / O(n) temporary):
    # the allocation count must not grow with the input size.
    for T in (Float32, Float64)
        for f in (logmeanexp, logvarexp, logstdexp,
                  logmeanexp_and_logvarexp, logmeanexp_and_logstdexp)
            @test allocations(f, randn(T, 10_000)) == allocations(f, randn(T, 100))
        end
        # genuinely allocation-free paths
        @test allocations(logmeanexp, randn(T, 10_000)) == 0
        @test allocations(logvarexp, Tuple(randn(T, 20))) == 0
    end
end

@testset "numerical robustness" begin
    # Compare against high-precision (BigFloat) references on hard cases: tight clusters
    # (var ≪ mean², where a raw second-moment formula cancels catastrophically) and
    # values large/small enough that exp would over-/under-flow Float64.
    setprecision(BigFloat, 256) do
        refmean(x) = Float64(log(mean(exp.(big.(x)))))
        refvar(x; corrected=true) = Float64(log(var(exp.(big.(x)); corrected=corrected)))
        cases = (
            1.0 .+ 1e-3 .* randn(500),   # tight cluster
            1.0 .+ 1e-6 .* randn(500),   # very tight: var ≪ mean²
            700.0 .+ randn(200),         # exp overflows Float64 (>~709)
            -700.0 .+ randn(200),        # exp underflows to 0
            1000.0 .* randn(500),        # huge dynamic range
            [3.0, 3.0 + 1e-10],          # nearly-equal pair
            [0.0, 1e-6],
        )
        for x in cases
            # atol covers near-zero results (e.g. logmeanexp([0, 1e-6]) ≈ 5e-7), where
            # an inherent cancellation makes the relative error meaningless
            @test logmeanexp(x) ≈ refmean(x) rtol = 1e-9 atol = 1e-10
            for corrected in (true, false)
                @test logvarexp(x; corrected=corrected) ≈ refvar(x; corrected=corrected) rtol = 1e-8 atol = 1e-9
                @test logstdexp(x; corrected=corrected) ≈ refvar(x; corrected=corrected) / 2 rtol = 1e-8 atol = 1e-9
            end
        end
        # array and one-shot-iterator paths agree even in the cancellation regime
        x = 1.0 .+ 1e-5 .* randn(100)
        @test logvarexp(x) ≈ logvarexp(Iterators.Stateful(x)) rtol = 1e-10
        m, v = logmeanexp_and_logvarexp(Iterators.Stateful(x))
        @test m ≈ logmeanexp(x) rtol = 1e-10
        @test v ≈ logvarexp(x) rtol = 1e-10
    end
end
