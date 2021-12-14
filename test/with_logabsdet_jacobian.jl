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
end
