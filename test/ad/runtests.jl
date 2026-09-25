using LogExpFunctions
using Enzyme
using ForwardDiff
using Mooncake
using Test

# issue #128
@testset "logsumexp at ties" begin
    # `t` and `tie(t, p)` are exactly equal at `t = p.x0`
    p = (x0 = 0.37, c = -2.5)
    tie(t, p) = p.x0 + p.c * (t - p.x0)
    dref = (1 + p.c) / 2
    fs = (
        tuple = (t, p) -> logsumexp((t, tie(t, p))),
        vector = (t, p) -> logsumexp([t, tie(t, p)]),
        generator = (t, p) -> logsumexp(x for x in (t, tie(t, p))),
        dims = (t, p) -> logsumexp([t tie(t, p)]; dims=2)[1],
        # abstract eltype and > 1024 elements: combines partial sums
        abstract = (t, p) -> logsumexp(Number[fill(t, 1024); fill(tie(t, p), 1024)]),
    )
    @testset "$name" for (name, f) in pairs(fs)
        y = f(p.x0, p)

        @test ForwardDiff.derivative(t -> f(t, p), p.x0) ≈ dref

        df, val = autodiff(ForwardWithPrimal, f, Duplicated(p.x0, 1.0), Const(p))
        @test val ≈ y
        @test df ≈ dref
        (df, _), val = autodiff(ReverseWithPrimal, f, Active, Active(p.x0), Const(p))
        @test val ≈ y
        @test df ≈ dref

        @testset "Mooncake $mode" for (mode, prepare) in pairs((
            forward = Mooncake.prepare_derivative_cache,
            reverse = Mooncake.prepare_gradient_cache,
        ))
            cache = prepare(f, p.x0, p)
            val, (_, df, _) = Mooncake.value_and_gradient!!(cache, f, p.x0, p)
            @test val ≈ y
            @test df ≈ dref
        end
    end
end
