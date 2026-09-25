using LogExpFunctions
using Enzyme
using ForwardDiff
using Mooncake
using Test

function mooncake_derivative(f, x)
    cache = Mooncake.prepare_gradient_cache(f, x)
    return Mooncake.value_and_gradient!!(cache, f, x)[2][2]
end

# issue #128
@testset "logsumexp at ties" begin
    fs = (
        tuple = t -> logsumexp((t, 2t - 1)),
        vector = t -> logsumexp([t, 2t - 1]),
        generator = t -> logsumexp(x for x in (t, 2t - 1)),
        dims = t -> logsumexp([t 2t - 1]; dims=2)[1],
        # abstract eltype and > 1024 elements: combines partial sums
        abstract = t -> logsumexp(Number[fill(t, 1024); fill(2t - 1, 1024)]),
    )
    @testset "$name" for (name, f) in pairs(fs)
        @test ForwardDiff.derivative(f, 1.0) ≈ 1.5
        @test only(autodiff(Forward, f, Duplicated(1.0, 1.0))) ≈ 1.5
        @test only(only(autodiff(Reverse, f, Active, Active(1.0)))) ≈ 1.5
        @test mooncake_derivative(f, 1.0) ≈ 1.5
    end
end
