@testset "inverse.jl" begin
    InverseFunctions.test_inverse(log1pexp, randn())
    InverseFunctions.test_inverse(softplus, randn())
    InverseFunctions.test_inverse(Base.Fix2(softplus, randexp()), randn())

    InverseFunctions.test_inverse(logexpm1, randexp())
    InverseFunctions.test_inverse(invsoftplus, randexp())
    InverseFunctions.test_inverse(Base.Fix2(invsoftplus, randexp()), randexp())

    InverseFunctions.test_inverse(log1mexp, -randexp())

    InverseFunctions.test_inverse(log2mexp, log(2) - randexp())

    InverseFunctions.test_inverse(logistic, randn())
    InverseFunctions.test_inverse(logit, rand())

    InverseFunctions.test_inverse(cloglog, rand())
    InverseFunctions.test_inverse(cexpexp, rand())

    InverseFunctions.test_inverse(loglogistic, randexp())
    InverseFunctions.test_inverse(logitexp, -randexp())

    InverseFunctions.test_inverse(log1mlogistic, randexp())
    InverseFunctions.test_inverse(logit1mexp, -randexp())
end
