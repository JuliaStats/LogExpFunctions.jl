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

@testset "xexpx" begin
    for x in (false, 0, 0.0, 0f0, -Inf, -Inf32)
        @test (@inferred xexpx(x)) === zero(exp(x))
    end
    for x in (NaN16, NaN32, NaN64, Inf16, Inf32, Inf64)
        @test (@inferred xexpx(x)) === x
    end
    for x in (1, true, 1.0, 1f0)
        @test (@inferred xexpx(x)) === exp(x)
    end
    for a in (2, 2f0, 2.0), x in -a:a
        @test (@inferred xexpx(x)) === x * exp(x)
    end
end

@testset "xexpy" begin
    for x in (0, 1, 1.0, 1f0, Inf, Inf32), y in (-Inf, -Inf32)
        @test (@inferred xexpy(x, y)) === zero(x * exp(y))
    end
    for x in (0, 1, 1.0, 1f0, Inf, Inf32, -Inf, -Inf32, NaN, NaN32), nan in (NaN, NaN32)
        @test (@inferred xexpy(x, nan)) === oftype(x * exp(nan), NaN)
        @test (@inferred xexpy(nan, x)) === oftype(nan * exp(x), NaN)
    end
    for x in (2, -2f0, 2.0), y in (1, -1f0, 1.0)
        @test (@inferred xexpy(x, y)) ≈ x * exp(y)
    end
    for x in (randn(), randn(Float32))
        @test xexpy(x, x) ≈ xexpx(x)
    end
    @test xexpy(0, 1000) == 0.0
    @test isnan(xexpy(0, Inf))
    @test isnan(xexpy(0, NaN))
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

@testset "logcosh and logabssinh" begin
    for x in (randn(), randn(Float32))
        @test @inferred(logcosh(x)) isa typeof(x)
        @test logcosh(x) ≈ log(cosh(x))
        @test logcosh(-x) == logcosh(x)
        @test @inferred(logabssinh(x)) isa typeof(x)
        @test logabssinh(x) ≈ log(abs(sinh(x)))
        @test logabssinh(-x) == logabssinh(x)
    end

    # special values
    for x in (-Inf, Inf, -Inf32, Inf32)
        @test @inferred(logcosh(x)) === oftype(x, Inf)
        @test @inferred(logabssinh(x)) === oftype(x, Inf)
    end
    for x in (NaN, NaN32)
        @test @inferred(logcosh(x)) === x
        @test @inferred(logabssinh(x)) === x
    end
end

@testset "log1psq" begin
    @test iszero(log1psq(0.0))
    @test log1psq(1.0) ≈ log1p(1.0)
    @test log1psq(2.0) ≈ log1p(4.0)
end

# log1pexp, log1mexp, log2mexp & logexpm1

@testset "log1pexp" begin
    for T in (Float16, Float32, Float64, BigFloat), x in 1:40
        @test (@inferred log1pexp(+log(T(x)))) ≈ T(log1p(big(x)))
        @test (@inferred log1pexp(-log(T(x)))) ≈ T(log1p(1/big(x)))
    end

    # special values
    @test (@inferred log1pexp(0)) ≈ log(2)
    @test (@inferred log1pexp(0f0)) ≈ log(2)
    @test (@inferred log1pexp(big(0))) ≈ log(2)
    @test (@inferred log1pexp(+1)) ≈ log1p(ℯ)
    @test (@inferred log1pexp(-1)) ≈ log1p(ℯ) - 1

    # large arguments
    @test (@inferred log1pexp(1e4)) ≈ 1e4
    @test (@inferred log1pexp(1f4)) ≈ 1f4
    @test iszero(@inferred log1pexp(-1e4))
    @test iszero(@inferred log1pexp(-1f4))

    # (almost) zero results
    for T in (Float16, Float32, Float64), x in (log(nextfloat(zero(T))), log(nextfloat(zero(T))) - 1)
        @test @inferred(log1pexp(x)) === log1p(exp(x))
    end

    # hard-coded thresholds
    for T in (Float16, Float32, Float64)
        @test LogExpFunctions._log1pexp_thresholds(zero(T)) === invoke(LogExpFunctions._log1pexp_thresholds, Tuple{Real}, zero(T))
    end

    # compare to accurate but slower implementation
    correct_log1pexp(x::Real) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
    # large range needed to cover all branches, for all floats (from Float16 to BigFloat)
    for T in (Int, Float16, Float32, Float64, BigInt, BigFloat), x in -300:300
        @test (@inferred log1pexp(T(x))) ≈ float(T)(correct_log1pexp(big(x)))
    end
    # test BigFloat with multiple precisions
    for prec in (10, 20, 50, 100), x in -300:300
        setprecision(prec) do
            y = big(float(x))
            @test @inferred(log1pexp(y)) ≈ correct_log1pexp(y)
        end
    end
end

@testset "log1mexp" begin
    for T in (Float64, Float32, Float16)
        @test @inferred(log1mexp(-T(1))) isa T
        @test log1mexp(-T(1))  ≈ log1p(- exp(-T(1)))
        @test log1mexp(-T(10)) ≈ log1p(- exp(-T(10)))
    end
end

@testset "log2mexp" begin
    for T in (Float64, Float32, Float16)
        @test @inferred(log2mexp(T(0))) isa T
        @test iszero(log2mexp(T(0)))
        @test log2mexp(-T(1)) ≈ log(2 - exp(-T(1)))
    end
end

@testset "logexpm1" begin
    for T in (Float64, Float32, Float16)
        @test @inferred(logexpm1(T(2))) isa T
        @test logexpm1(T(2))            ≈  log(exp(T(2)) - 1)
        @test logexpm1(log1pexp(T(2)))  ≈  T(2)
        @test logexpm1(log1pexp(-T(2))) ≈ -T(2)
    end
end

@testset "log1pmx" begin
    @test iszero(log1pmx(0.0))
    @test log1pmx(1.0) ≈ log(2.0) - 1.0
    @test log1pmx(2.0) ≈ log(3.0) - 2.0

    @test iszero(log1pmx(0f0))
    @test log1pmx(1f0) ≈ log(2f0) - 1f0
    @test log1pmx(2f0) ≈ log(3f0) - 2f0

    for x in -0.5:0.1:10
        @test log1pmx(Float32(x)) ≈ Float32(log1pmx(x))
    end
end

@testset "logmxp1" begin
    @test iszero(logmxp1(1.0))
    @test logmxp1(2.0) ≈ log(2.0) - 1.0
    @test logmxp1(3.0) ≈ log(3.0) - 2.0

    @test iszero(logmxp1(1f0))
    @test logmxp1(2f0) ≈ log(2f0) - 1f0
    @test logmxp1(3f0) ≈ log(3f0) - 2f0

    for x in 0.1:0.1:10
        @test logmxp1(Float32(x)) ≈ Float32(logmxp1(x))
    end
end

@testset "logsumexp" begin
    Ts = (Int, Float32, Float64)
    for T1 in Ts, T2 in Ts
        a = T1(2)
        b = T2(3)

        x = @inferred(logaddexp(a, b))
        @test x ≈ log(exp(a) + exp(b))
        @test typeof(x) === float(Base.promote_typeof(a, b))

        y = @inferred(logaddexp(a + 10_000, b + 10_000))
        @test y ≈ 10_000 + log(exp(a) + exp(b))
        @test typeof(y) === typeof(x)
    end

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
        expected = [3.40760596444438 1003.40760596444438]
        @test @inferred(logsumexp(x; dims=1)) ≈ expected
        out = Array{eltype(x)}(undef, 1, 2)
        @test @inferred(logsumexp!(out, x)) ≈ expected
        @test out ≈ expected

        y = copy(x')
        expected = [3.40760596444438, 1003.40760596444438]
        @test @inferred(logsumexp(y; dims=2)) ≈ expected
        out = Array{eltype(y)}(undef, 2)
        @test @inferred(logsumexp!(out, y)) ≈ expected
        @test out ≈ expected

        expected = [1003.4076059644444]
        @test @inferred(logsumexp(x; dims=[1, 2])) ≈ expected
        out = Array{eltype(x)}(undef, 1)
        @test @inferred(logsumexp!(out, x)) ≈ expected
        @test out ≈ expected
    end

    # check underflow
    @test logsumexp([1e-20, log(1e-20)]) ≈ 2e-20
    @test logsumexp(Complex{Float64}[1e-20, log(1e-20)]) ≈ 2e-20
    @test logsumexp!([1.0], [1e-20, log(1e-20)]) ≈ [2e-20]
    @test logsumexp!(Complex{Float64}[1.0], Complex{Float64}[1e-20, log(1e-20)]) ≈ [2e-20]

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

            FT = float(eltype(arguments))
            out = [one(FT)]
            @test logsumexp!(out, arguments)[1] ≡ result
            @test out[1] ≡ result

            out = [one(complex(FT))]
            @test logsumexp!(out, complex(arguments))[1] ≡ complex(result)
            @test out[1] ≡ complex(result)
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
    @test isnan(logsumexp!([1.0], [NaN, 9.0])[1])
    @test isnan(logsumexp!([1.0], [NaN, Inf])[1])
    @test isnan(logsumexp!([1.0], [NaN, -Inf])[1])
    @test isnan(logsumexp!(Complex{Float64}[1.0], Complex{Float64}[NaN, 9.0])[1])
    @test isnan(logsumexp!(Complex{Float64}[1.0], Complex{Float64}[NaN, Inf])[1])
    @test isnan(logsumexp!(Complex{Float64}[1.0], Complex{Float64}[NaN, -Inf])[1])
    @test isnan(logsumexp!(Complex{Float64}[1.0], Complex{Float64}[NaN * im, 9.0])[1])
    @test isnan(logsumexp!(Complex{Float64}[1.0], Complex{Float64}[NaN * im, Inf])[1])
    @test isnan(logsumexp!(Complex{Float64}[1.0], Complex{Float64}[NaN * im, -Inf])[1])

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

    # output arrays with abstract eltype
    xs = randn(2, 4)
    out = [missing, 1.0]
    expected = logsumexp(xs; dims=2)
    @test logsumexp!(out, xs) ≈ expected
    @test out ≈ expected

    @testset "ForwardDiff" begin
        # vector with finite numbers
        x = randn(10)
        ∇x = unthunk(rrule(logsumexp, x)[2](1)[2])
        @test ForwardDiff.gradient(logsumexp, x) ≈ ∇x

        # issue #59
        x = vcat(-Inf, randn(9))
        ∇x = unthunk(rrule(logsumexp, x)[2](1)[2])
        @assert all(isfinite, ∇x)
        @test ForwardDiff.gradient(logsumexp, x) ≈ ∇x
    end
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

@testset "cloglog and cexpexp" begin
    cloglog_big(x::T) where {T} = T(log(-log(1 - BigFloat(x))))
    cexpexp_big(x::T) where {T} = 1 - exp(-exp(BigFloat(x)))

    for T in (Float64, Float32, Float16)
        @test @inferred(cloglog(T(1//2))) isa T
        @test @inferred(cexpexp(T(0))) isa T
        for x in 0.1:0.1:0.9
            @test cloglog(T(x)) ≈ cloglog_big(T(x))
            # Julia bug for Float32 and Float16 initially introduced in https://github.com/JuliaLang/julia/pull/37440
            # and fixed in https://github.com/JuliaLang/julia/pull/50989
            if T === Float64 || VERSION < v"1.7.0-DEV.887" || VERSION >= v"1.11.0-DEV.310"
                @test cexpexp(T(x)) ≈ cexpexp_big(T(x))
            end
        end
    end
    for _ in 1:10
        randf = rand(Float64)
        @test cloglog(randf) ≈ cloglog_big(randf)
        randi = rand(Int)
        @test cexpexp(randi) ≈ cexpexp_big(randi)
    end

    @test cloglog(0) == -Inf
    @test cloglog(1) == Inf
    @test cloglog((ℯ - 1) / ℯ) == 0

    @test cexpexp(Inf) == 1.0
    @test cexpexp(-Inf) == 0.0
    @test cexpexp(0) == (ℯ - 1) / ℯ
end
