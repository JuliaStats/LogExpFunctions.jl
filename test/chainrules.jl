@testset "chainrules.jl" begin
    x = rand()
    test_frule(xlogx, x)
    test_rrule(xlogx, x)

    # Test `iszero(x)` branches
    test_frule(xlogy, 0.0, 1.0; fdm = forward_fdm(5, 1), nans = true)
    test_rrule(xlogy, 0.0, 1.0; fdm = forward_fdm(5, 1), nans = true)
    @test iszero(last(frule((NoTangent(), ZeroTangent(), 1.), xlog1py, 0.0, -1.0)))
    @test iszero(last(last(rrule(xlog1py, 0.0, -1.0))(1.)))

    for x in (-x, 0.0, x)
        y = rand()
        test_frule(xlogy, x, y)
        test_rrule(xlogy, x, y)

        for z in (-y, y)
            test_frule(xlog1py, x, z)
            test_rrule(xlog1py, x, z)
        end
    end

    @testset "xexpx" begin
        # regular branch
        test_scalar(xexpx, randn())
        # special cases (manually since FiniteDifferences/ChainRulesTestUtils fails at -Inf)
        @test @inferred(frule((NoTangent(), rand()), xexpx, -Inf)) === (0.0, 0.0)
        Ω, back = @inferred(rrule(xexpx, -Inf))
        @test Ω === 0.0
        @test back(rand()) === (NoTangent(), 0.0)
    end

    @testset "xexpy" begin
        # regular branch
        test_frule(xexpy, randn(), randn())
        test_rrule(xexpy, randn(), randn())
        # special cases (manually since FiniteDifferences/ChainRulesTestUtils fails at -Inf)
        @test @inferred(frule((NoTangent(), rand(), rand()), xexpy, x, -Inf)) === (0.0, 0.0)
        Ω, back = @inferred(rrule(xexpy, x, -Inf))
        @test Ω === 0.0
        @test back(rand()) === (NoTangent(), 0.0, 0.0)
    end

    test_frule(logit, x)
    test_rrule(logit, x)

    for x in (-randexp(), randexp())
        test_frule(log1psq, x)
        test_rrule(log1psq, x)
    end
    # no overflow of `x^2`
    for x in (-1e200, 1e200)
        @test last(frule((NoTangent(), 1.0), log1psq, x)) ≈ 2 / x rtol=4*eps()
        @test last(last(rrule(log1psq, x))(1.0)) ≈ 2 / x rtol=4*eps()
    end

    # test all `Float64` and `Float32` branches of `logistic`
    for x in (-821.4, -23.5, 12.3, 41.2)
        test_frule(logistic, x)
        test_rrule(logistic, x)
    end
    for x in (-123.2f0, -21.4f0, 8.3f0, 21.5f0)
        test_frule(logistic, x; rtol=1f-3, atol=1f-3)
        test_rrule(logistic, x; rtol=1f-3, atol=1f-3)
    end
    # derivative is accurate for large `|x|` in a relative sense (no cancellation)
    for x in (40.0, -40.0, 30f0, -30f0)
        ∂x = exp(-abs(x))
        @test last(frule((NoTangent(), one(x)), logistic, x)) ≈ ∂x rtol=4*eps(typeof(x))
        @test last(last(rrule(logistic, x))(one(x))) ≈ ∂x rtol=4*eps(typeof(x))
    end

    for x in (-randexp(), randexp())
        test_frule(logcosh, x)
        test_rrule(logcosh, x)
        test_frule(logabssinh, x)
        test_rrule(logabssinh, x)
        test_frule(logabstanh, x)
        test_rrule(logabstanh, x)
    end

    @testset "log1pexp" begin
        for absx in (0, 1, 2, 10, 15, 20, 40), x in (-absx, absx)
            test_scalar(log1pexp, Float64(x))
            test_scalar(log1pexp, Float32(x); rtol=1f-3, atol=1f-3)
        end
    end

    for x in (-10.2, -3.3, -0.3)
        test_frule(log1mexp, x)
        test_rrule(log1mexp, x)
    end

    for x in (-10.2, -3.3, -0.3, 0.5)
        test_frule(log2mexp, x)
        test_rrule(log2mexp, x)
    end

    # test all branches of `logexpm1`
    for x in (5.2, 21.4, 41.5)
        test_frule(logexpm1, x)
        test_rrule(logexpm1, x)
    end
    for x in (4.3f0, 12.5f0, 21.2f0)
        test_frule(logexpm1, x; rtol=1f-3, atol=1f-3)
        test_rrule(logexpm1, x; rtol=1f-3, atol=1f-3)
    end
    # special case (manually since FiniteDifferences/ChainRulesTestUtils fails at Inf)
    @test frule((NoTangent(), 1.0), logexpm1, Inf) === (Inf, 1.0)
    @test last(last(rrule(logexpm1, Inf))(1.0)) === 1.0

    # accuracy for small `|x|` (where `Ω` is large in magnitude)
    for (f, x) in ((log1mexp, -1e-12), (logexpm1, 6.039463507942533e-8))
        ∂x = -inv(expm1(-big(x)))
        @test last(frule((NoTangent(), 1.0), f, x)) ≈ ∂x rtol=4*eps()
        @test last(last(rrule(f, x))(1.0)) ≈ ∂x rtol=4*eps()
    end

    for x in (-randexp(), randexp())
        test_scalar(loglogistic, x)
        test_scalar(log1mlogistic, x)
    end
    for x in (-10.2, -3.3, -0.3)
        test_scalar(logitexp, x)
        test_scalar(logit1mexp, x)
    end

    test_scalar(log1pmx, rand())

    test_scalar(logmxp1, 0.5 + rand())

    for x in (-randexp(), randexp()), y in (-randexp(), randexp())
        test_frule(logaddexp, x, y)
        test_rrule(logaddexp, x, y)

        test_frule(logsubexp, x, y)
        test_rrule(logsubexp, x, y)
    end
    # nearly equal arguments (manually since finite differences are inaccurate close to
    # the singularity at `x == y`)
    for (x, y) in ((1.0, 1.0 + 1e-6), (1.0 + 1e-6, 1.0))
        d = big(x) - big(y)
        ∂x, ∂y = -inv(expm1(-d)), -inv(expm1(d))
        @test last(frule((NoTangent(), 1.0, 0.0), logsubexp, x, y)) ≈ ∂x rtol=4*eps()
        @test last(frule((NoTangent(), 0.0, 1.0), logsubexp, x, y)) ≈ ∂y rtol=4*eps()
        _, back = rrule(logsubexp, x, y)
        _, x̄, ȳ = back(1.0)
        @test x̄ ≈ ∂x rtol=4*eps()
        @test ȳ ≈ ∂y rtol=4*eps()
    end
    # accuracy for arguments of large magnitude
    let (x, y) = (300.0, 301.0)
        d = big(x) - big(y)
        ∂x, ∂y = logistic(d), logistic(-d)
        _, back = rrule(logaddexp, x, y)
        _, x̄, ȳ = back(1.0)
        @test x̄ ≈ ∂x rtol=4*eps()
        @test ȳ ≈ ∂y rtol=4*eps()
    end
    # special cases (manually since FiniteDifferences/ChainRulesTestUtils fails at Inf)
    for (f, ∂x, ∂y) in ((logaddexp, 1.0, 0.0), (logsubexp, 1.0, -0.0))
        Ω, back = rrule(f, Inf, 1.0)
        @test Ω === Inf
        @test back(1.0) === (NoTangent(), ∂x, ∂y)
        Ω, back = rrule(f, 1.0, Inf)
        @test Ω === Inf
        @test back(1.0) === (NoTangent(), ∂y, ∂x)
        @test last(frule((NoTangent(), 1.0, 0.0), f, Inf, 1.0)) === ∂x
    end

    for x in (randn(10), randn(10, 8)), dims in (:, 1, 1:2, 2)
        dims isa Colon || all(d <= ndims(x) for d in dims) || continue
        test_frule(logsumexp, x; fkwargs=(dims=dims,))
        test_rrule(logsumexp, x; fkwargs=(dims=dims,))
    end

    for x in (randn(10), randn(10, 8))
        test_frule(softmax, x)
        test_rrule(softmax, x)

        for dims in (1, 1:2, 2)
            all(d <= ndims(x) for d in dims) || continue
            test_frule(softmax, x; fkwargs=(dims=dims,))
            test_rrule(softmax, x; fkwargs=(dims=dims,))
        end
    end

    test_scalar(cloglog, rand())

    test_scalar(cexpexp, rand())
end
