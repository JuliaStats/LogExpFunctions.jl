@testset "inverse.jl" begin
    InverseFunctions.test_inverse(log1pexp, randn())
    InverseFunctions.test_inverse(logexpm1, randexp())

    InverseFunctions.test_inverse(log1mexp, -randexp())

    InverseFunctions.test_inverse(log2mexp, log(2) - randexp())

    InverseFunctions.test_inverse(logistic, randn())
    InverseFunctions.test_inverse(logit, rand())

    InverseFunctions.test_inverse(cloglog, rand())
    InverseFunctions.test_inverse(cexpexp, rand())
end
