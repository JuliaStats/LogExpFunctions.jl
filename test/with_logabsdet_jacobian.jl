@testset "with_logabsdet_jacobian" begin
    derivative(f, x) = ChainRulesTestUtils.frule((ChainRulesTestUtils.NoTangent(), 1), f, x)[2]

    x = randexp()

    ChangesOfVariables.test_with_logabsdet_jacobian(log1pexp, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(log1pexp, -x, derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(logexpm1, x, derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(log1mexp, -x, derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(log2mexp, log(2) - x, derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(logistic, -x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(logistic, x, derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(logit, rand(), derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(logcosh, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(logcosh, -x, derivative)

    dloglogistic(x) = logistic(-x)
    dlog1mlogistic(x) = -logistic(x)
    dlogitexp(x) = inv(1 - exp(x))
    dlogit1mexp(x) = -inv(1 - exp(x))
    derivative(::Type{loglogistic}, x) = dloglogistic(x)
    derivative(::Type{log1mlogistic}, x) = dlog1mlogistic(x)
    derivative(::Type{logitexp}, x) = dlogitexp(x)
    derivative(::Type{logit1mexp}, x) = dlogit1mexp(x)

    ChangesOfVariables.test_with_logabsdet_jacobian(loglogistic, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(logitexp, -x, derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(log1mlogistic, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(logit1mexp, -x, derivative)
end
