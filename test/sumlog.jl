@testset "sumlog" begin
    for T in [Int, Float16, Float32, Float64, BigFloat]
        for x in [10 .* rand(1000), repeat([nextfloat(1.0)], 1000), repeat([prevfloat(2.0)], 1000)]
            @test (@inferred sumlog(x)) ≈ sum(log, x)

            y = view(x, 1:100)
            @test (@inferred sumlog(y)) ≈ sum(log, y)

            tup = tuple(y...)
            @test (@inferred sumlog(tup)) ≈ sum(log, tup)

            gen = (sqrt(a) for a in y)
            @test_broken (@inferred sumlog(gen)) ≈ sum(log, gen)

            nt = NamedTuple{tuple(Symbol.(1:100)...)}(tup)
            @test (@inferred sumlog(y)) ≈ sum(log, y)

            i = Random.shuffle(x)
            z = x .+ i * im
            @test (@inferred sumlog(z)) ≈ sum(log, z)
        end

    end
end