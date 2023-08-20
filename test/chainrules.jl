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

    # test all `Float64` and `Float32` branches of `logistic`
    for x in (-821.4, -23.5, 12.3, 41.2)
        test_frule(logistic, x)
        test_rrule(logistic, x)
    end
    for x in (-123.2f0, -21.4f0, 8.3f0, 21.5f0)
        test_frule(logistic, x; rtol=1f-3, atol=1f-3)
        test_rrule(logistic, x; rtol=1f-3, atol=1f-3)
    end

    for x in (-randexp(), randexp())
        test_frule(logcosh, x)
        test_rrule(logcosh, x)
        test_frule(logabssinh, x)
        test_rrule(logabssinh, x)
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

    test_scalar(log1pmx, rand())

    test_scalar(logmxp1, 0.5 + rand())

    for x in (-randexp(), randexp()), y in (-randexp(), randexp())
        test_frule(logaddexp, x, y)
        test_rrule(logaddexp, x, y)

        test_frule(logsubexp, x, y)
        test_rrule(logsubexp, x, y)
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
