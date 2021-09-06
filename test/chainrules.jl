@testset "chainrules.jl" begin
    x = rand()
    test_frule(xlogx, x)
    test_rrule(xlogx, x)
    for x in (-x, 0.0, x)
        y = rand()
        test_frule(xlogy, x, y)
        test_rrule(xlogy, x, y)

        for z in (-y, y)
            test_frule(xlog1py, x, z)
            test_rrule(xlog1py, x, z)
        end
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

    # test all branches of `log1pexp`
    for x in (-20.9, 15.4, 41.5)
        test_frule(log1pexp, x)
        test_rrule(log1pexp, x)
    end
    for x in (8.3f0, 12.5f0, 21.2f0)
        test_frule(log1pexp, x; rtol=1f-3, atol=1f-3)
        test_rrule(log1pexp, x; rtol=1f-3, atol=1f-3)
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
        for r in (similar(x), similar(x, 1, size(x)...))
            test_frule(softmax!, r, x)
            test_rrule(softmax!, r, x)
        end
    end
end
