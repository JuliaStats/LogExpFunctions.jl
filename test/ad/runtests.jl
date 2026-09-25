using LogExpFunctions
using Enzyme
using ForwardDiff
using Mooncake
using Test

# issue #128
@testset "logsumexp at ties" begin
    # `t` and `2t - 0.37` are exactly equal at `t = 0.37`
    x0 = 0.37
    dref = 1.5
    fs = (
        tuple = t -> logsumexp((t, 2t - 0.37)),
        vector = t -> logsumexp([t, 2t - 0.37]),
        generator = t -> logsumexp(x for x in (t, 2t - 0.37)),
        dims = t -> logsumexp([t 2t - 0.37]; dims=2)[1],
        # abstract eltype and > 1024 elements: combines partial sums
        abstract = t -> logsumexp(Number[fill(t, 1024); fill(2t - 0.37, 1024)]),
    )
    @testset "$name" for (name, f) in pairs(fs)
        y = f(x0)

        @test ForwardDiff.derivative(f, x0) ≈ dref

        df, val = autodiff(ForwardWithPrimal, f, Duplicated(x0, 1.0))
        @test val ≈ y
        @test df ≈ dref
        (df,), val = autodiff(ReverseWithPrimal, f, Active, Active(x0))
        @test val ≈ y
        @test df ≈ dref

        @testset "Mooncake $mode" for (mode, prepare) in pairs((
            forward = Mooncake.prepare_derivative_cache,
            reverse = Mooncake.prepare_gradient_cache,
        ))
            cache = prepare(f, x0)
            val, (_, df) = Mooncake.value_and_gradient!!(cache, f, x0)
            @test val ≈ y
            @test df ≈ dref
        end
    end
end
