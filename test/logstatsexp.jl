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
    @test @inferred(logvarexp(xt)) ≈ log(var(xe))
    @test logvarexp(xt; corrected=false) ≈ log(var(xe; corrected=false))
    @test @inferred(logvarexp((v for v in x))) ≈ log(var(xe))
    @test logvarexp((v for v in x); corrected=false) ≈ log(var(xe; corrected=false))
    @test @inferred(logstdexp(xt)) ≈ log(std(xe))
    @test logstdexp(xt; corrected=false) ≈ log(std(xe; corrected=false))
    @test @inferred(logstdexp((v for v in x))) ≈ log(std(xe))
    @test logstdexp((v for v in x); corrected=false) ≈ log(std(xe; corrected=false))
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
end
