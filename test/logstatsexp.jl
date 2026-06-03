using Test: @test, @testset, @inferred
using Statistics: mean, std, var
using LogExpFunctions: logmeanexp, logstdexp, logvarexp

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
        @test @inferred(logvarexp(X)) ≈ log(var(exp, X))
        @test @inferred(logstdexp(X)) ≈ log(std(exp, X))
    end
end

@testset "logmeanexp, logvarexp, logstdexp iterators" begin
    x = randn(Float32, 20)
    xt = Tuple(x)
    xg = (v for v in x)

    @test @inferred(logmeanexp(xt)) ≈ log(mean(exp, xt))
    @test logmeanexp(xg) ≈ log(mean(exp, x))
    @test @inferred(logvarexp(xt)) ≈ log(var(exp, xt))
    @test logvarexp(xt; corrected=false) ≈ log(var(exp, xt; corrected=false))
    @test @inferred(logstdexp(xt)) ≈ log(std(exp, xt))
    @test logstdexp(xt; corrected=false) ≈ log(std(exp, xt; corrected=false))
end
