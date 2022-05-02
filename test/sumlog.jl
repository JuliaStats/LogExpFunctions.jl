@testset "sumlog" begin
    for T in [Int, Float16, Float32, Float64, BigFloat]
        for x in [10 .* rand(1000), repeat([nextfloat(1.0)], 1000), repeat([prevfloat(2.0)], 1000)]
            @test (@inferred sumlog(x)) ≈ sum(log, x)
        end
    end
end