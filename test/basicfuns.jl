@testset "xlogx, xlogy, and xlog1py" begin
    @test iszero(xlogx(0))
    @test xlogx(2) ≈ 2.0 * log(2.0)
    @test_throws DomainError xlogx(-1)
    @test isnan(xlogx(NaN))

    @test iszero(xlogy(0, 1))
    @test xlogy(2, 3) ≈ 2.0 * log(3.0)
    @test_throws DomainError xlogy(1, -1)
    @test isnan(xlogy(NaN, 2))
    @test isnan(xlogy(2, NaN))
    @test isnan(xlogy(0, NaN))

    @test iszero(xlog1py(0, 0))
    @test xlog1py(2, 3) ≈ 2.0 * log1p(3.0)
    @test_throws DomainError xlog1py(1, -2)
    @test isnan(xlog1py(NaN, 2))
    @test isnan(xlog1py(2, NaN))
    @test isnan(xlog1py(0, NaN))

    # Since we allow complex/negative values, test for them. See comments in:
    # https://github.com/JuliaStats/StatsFuns.jl/pull/95

    @test xlogx(1 + im) == (1 + im) * log(1 + im)
    @test isnan(xlogx(NaN + im))
    @test isnan(xlogx(1 + NaN * im))

    @test xlogy(-2, 3) == -xlogy(2, 3)
    @test xlogy(1 + im, 3) == (1 + im) * log(3)
    @test xlogy(1 + im, 2 + im) == (1 + im) * log(2 + im)
    @test isnan(xlogy(1 + NaN * im, -1 + im))
    @test isnan(xlogy(0, -1 + NaN * im))
    @test isnan(xlogy(Inf + im * NaN, 1))
    @test isnan(xlogy(0 + im * 0, NaN))
    @test iszero(xlogy(0 + im * 0, 0 + im * Inf))

    @test xlog1py(-2, 3) == -xlog1py(2, 3)
    @test xlog1py(1 + im, 3) == (1 + im) * log1p(3)
    @test xlog1py(1 + im, 2 + im) == (1 + im) * log1p(2 + im)
    @test isnan(xlog1py(1 + NaN * im, -1 + im))
    @test isnan(xlog1py(0, -1 + NaN * im))
    @test isnan(xlog1py(Inf + im * NaN, 1))
    @test isnan(xlog1py(0 + im * 0, NaN))
    @test iszero(xlog1py(0 + im * 0, -1 + im * Inf))
end

@testset "logistic & logit" begin
    @test logistic(2) ≈ 1.0 / (1.0 + exp(-2.0))
    @test logistic(-750.0) === 0.0
    @test logistic(-740.0) > 0.0
    @test logistic(+36.0) < 1.0
    @test logistic(+750.0) === 1.0
    @test iszero(logit(0.5))
    @test logit(logistic(2)) ≈ 2.0
end

@testset "logcosh" begin
    for x in (randn(), randn(Float32))
        @test @inferred(logcosh(x)) isa typeof(x)
        @test logcosh(x) ≈ log(cosh(x))
        @test logcosh(-x) == logcosh(x)
    end

    # special values
    for x in (-Inf, Inf, -Inf32, Inf32)
        @test @inferred(logcosh(x)) === oftype(x, Inf)
    end
    for x in (NaN, NaN32)
        @test @inferred(logcosh(x)) === x
    end
end

@testset "log1psq" begin
    @test iszero(log1psq(0.0))
    @test log1psq(1.0) ≈ log1p(1.0)
    @test log1psq(2.0) ≈ log1p(4.0)
end

# log1pexp, log1mexp, log2mexp & logexpm1

@testset "log1pexp" begin
    @test log1pexp(2.0)    ≈ log(1.0 + exp(2.0))
    @test log1pexp(-2.0)   ≈ log(1.0 + exp(-2.0))
    @test log1pexp(10000)  ≈ 10000.0
    @test log1pexp(-10000) ≈ 0.0

    @test log1pexp(2f0)      ≈ log(1f0 + exp(2f0))
    @test log1pexp(-2f0)     ≈ log(1f0 + exp(-2f0))
    @test log1pexp(10000f0)  ≈ 10000f0
    @test log1pexp(-10000f0) ≈ 0f0
end

@testset "log1mexp" begin
    @test log1mexp(-1.0)  ≈ log1p(- exp(-1.0))
    @test log1mexp(-10.0) ≈ log1p(- exp(-10.0))
end

@testset "log2mexp" begin
    @test log2mexp(0.0)  ≈ 0.0
    @test log2mexp(-1.0) ≈ log(2.0 - exp(-1.0))
end

@testset "logexpm1" begin
    @test logexpm1(2.0)            ≈  log(exp(2.0) - 1.0)
    @test logexpm1(log1pexp(2.0))  ≈  2.0
    @test logexpm1(log1pexp(-2.0)) ≈ -2.0

    @test logexpm1(2f0)            ≈  log(exp(2f0) - 1f0)
    @test logexpm1(log1pexp(2f0))  ≈  2f0
    @test logexpm1(log1pexp(-2f0)) ≈ -2f0
end

@testset "log1pmx" begin
    @test iszero(log1pmx(0.0))
    @test log1pmx(1.0) ≈ log(2.0) - 1.0
    @test log1pmx(2.0) ≈ log(3.0) - 2.0
end

@testset "logmxp1" begin
    @test iszero(logmxp1(1.0))
    @test logmxp1(2.0) ≈ log(2.0) - 1.0
    @test logmxp1(3.0) ≈ log(3.0) - 2.0
end

@testset "logsumexp" begin
    @test logaddexp(2.0, 3.0)     ≈ log(exp(2.0) + exp(3.0))
    @test logaddexp(10002, 10003) ≈ 10000 + logaddexp(2.0, 3.0)

    for x in ([1.0], Complex{Float64}[1.0])
        @test @inferred(logsumexp(x)) == 1.0
        @test @inferred(logsumexp((xi for xi in x))) == 1.0
    end

    for x in ([1.0, 2.0, 3.0], Complex{Float64}[1.0, 2.0, 3.0])
        @test @inferred(logsumexp(x)) ≈ 3.40760596444438
        @test logsumexp(x .+ 1000) ≈ 1003.40760596444438
    end

    for x in ((1.0, 2.0, 3.0), map(complex, (1.0, 2.0, 3.0)))
        @test @inferred(logsumexp(x)) ≈ 3.40760596444438
    end

    _x = [[1.0, 2.0, 3.0] [1.0, 2.0, 3.0] .+ 1000.]
    for x in (_x, complex(_x))
        @test @inferred(logsumexp(x; dims=1)) ≈ [3.40760596444438 1003.40760596444438]
        @test @inferred(logsumexp(x; dims=[1, 2])) ≈ [1003.4076059644444]
        y = copy(x')
        @test @inferred(logsumexp(y; dims=2)) ≈ [3.40760596444438, 1003.40760596444438]
    end

    # check underflow
    @test logsumexp([1e-20, log(1e-20)]) ≈ 2e-20
    @test logsumexp(Complex{Float64}[1e-20, log(1e-20)]) ≈ 2e-20

    let cases = [([-Inf, -Inf], -Inf),   # correct handling of all -Inf
                 ([-Inf, -Inf32], -Inf), # promotion
                 ([-Inf32, -Inf32], -Inf32), # Float32
                 ([-Inf, Inf], Inf),
                 ([-Inf, 9.0], 9.0),
                 ([Inf, 9.0], Inf),
                 ([0, 0], log(2.0))] # non-float arguments
        for (arguments, result) in cases
            @test logaddexp(arguments...) ≡ result
            @test logsumexp(arguments) ≡ result
            @test logsumexp(complex(arguments)) ≡ complex(result)
        end
    end

    @test isnan(logsubexp(Inf, Inf))
    @test logsubexp(-Inf, -Inf) ≡ -Inf
    @test logsubexp(Inf, -Inf) ≡ Inf
    @test logsubexp(-Inf, Inf) ≡ Inf
    @test logsubexp(Inf, 9.0) ≡ Inf
    @test logsubexp(9.0, Inf) ≡ Inf
    @test logsubexp(-Inf, 9.0) ≡ 9.0
    @test logsubexp(9.0, -Inf) ≡ 9.0
    @test logsubexp(1f2, 1f2) ≡ -Inf32
    @test logsubexp(0, 0) ≡ -Inf
    @test logsubexp(3, 2) ≈ 2.541324854612918108978

    # NaN propagation
    for f in (logaddexp, logsubexp)
        @test isnan(f(NaN, 9.0))
        @test isnan(f(NaN, Inf))
        @test isnan(f(NaN, -Inf))
        @test isnan(f(9.0, NaN))
        @test isnan(f(Inf, NaN))
        @test isnan(f(-Inf, NaN))
    end

    @test isnan(logsumexp([NaN, 9.0]))
    @test isnan(logsumexp([NaN, Inf]))
    @test isnan(logsumexp([NaN, -Inf]))
    @test isnan(logsumexp(Complex{Float64}[NaN, 9.0]))
    @test isnan(logsumexp(Complex{Float64}[NaN, Inf]))
    @test isnan(logsumexp(Complex{Float64}[NaN, -Inf]))
    @test isnan(logsumexp(Complex{Float64}[NaN * im, 9.0]))
    @test isnan(logsumexp(Complex{Float64}[NaN * im, Inf]))
    @test isnan(logsumexp(Complex{Float64}[NaN * im, -Inf]))

    # logsumexp with general iterables (issue #63)
    xs = range(-500, stop = 10, length = 1000)
    @test @inferred(logsumexp(x for x in xs)) == logsumexp(xs)
    xs = range(-500 + 0.5im, stop = 10 + 30im, length = 1000)
    @test @inferred(logsumexp(x for x in xs)) == logsumexp(xs)

    # complex numbers
    xs = randn(Complex{Float64}, 10, 5)
    @test @inferred(logsumexp(xs)) ≈ log(sum(exp.(xs)))
    @test @inferred(logsumexp(xs; dims=1)) ≈ log.(sum(exp.(xs); dims=1))
    @test @inferred(logsumexp(xs; dims=2)) ≈ log.(sum(exp.(xs); dims=2))
    @test @inferred(logsumexp(xs; dims=[1, 2])) ≈ log(sum(exp.(xs); dims=[1, 2]))
    @test @inferred(logsumexp(x for x in xs)) == logsumexp(xs)
end

@testset "softmax" begin
    x = [1.0, 2.0, 3.0]
    r = exp.(x) ./ sum(exp.(x))

    # in-place versions
    for T in (Float32, Float64)
        s = Vector{T}(undef, 3)
        softmax!(s, x)
        @test s ≈ r

        fill!(s, zero(T))
        softmax!(s, x; dims=1)
        @test s ≈ r

        s = Matrix{T}(undef, 1, 3)
        softmax!(s, x)
        @test s ≈ permutedims(r)

        @test_throws DimensionMismatch softmax!(s, x; dims=1)

        fill!(s, zero(T))
        softmax!(s, permutedims(x); dims=2)
        @test s ≈ permutedims(r)

        fill!(s, zero(T))
        softmax!(s, permutedims(x); dims=1:2)
        @test s ≈ permutedims(r)
    end
    softmax!(x)
    @test x ≈ r

    for (S, T) in ((Int, Float64), (Float64, Float64), (Float32, Float32))
        x = S[1, 2, 3]
        s = softmax(x)
        @test s ≈ r
        @test eltype(s) === T

        x = repeat(S[1, 2, 3], 1, 3)
        s = softmax(x; dims=1)
        @test s ≈ repeat(r, 1, 3)
        @test eltype(s) === T

        x = repeat(S[1 2 3], 3, 1)
        s = softmax(x; dims=2)
        @test s ≈ repeat(permutedims(r), 3, 1)
        @test eltype(s) === T

        x = S[1 2 3]
        s = softmax(x; dims=1:2)
        @test s ≈ permutedims(r)
        @test eltype(s) === T
    end

    x = [1//2, 2//3, 3//4]
    r = exp.(x) ./ sum(exp.(x))
    s = softmax(x)
    @test s ≈ r
    @test eltype(s) === Float64

    # non-standard indices: #12
    x = OffsetArray(1:3, -2:0)
    s = softmax(x)
    @test s isa OffsetArray{Float64}
    @test axes(s, 1) == OffsetArrays.IdOffsetRange(-2:0)
    @test collect(s) ≈ softmax(1:3)
end
