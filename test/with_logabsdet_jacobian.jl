@testset "with_logabsdet_jacobian" begin
    derivative(f, x) = ChainRulesTestUtils.frule((ChainRulesTestUtils.NoTangent(), 1), f, x)[2]
    derivative(::typeof(softplus), x) = derivative(log1pexp, x)
    derivative(f::Base.Fix2{typeof(softplus),<:Real}, x) = derivative(log1pexp, f.x * x)
    derivative(::typeof(invsoftplus), x) = derivative(logexpm1, x)
    derivative(f::Base.Fix2{typeof(invsoftplus),<:Real}, x) = derivative(logexpm1, f.x * x)

    x = randexp()
    y = randexp()

    ChangesOfVariables.test_with_logabsdet_jacobian(log1pexp, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(log1pexp, -x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(softplus, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(softplus, -x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(Base.Fix2(softplus, y), x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(Base.Fix2(softplus, y), -x, derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(logexpm1, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(invsoftplus, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(Base.Fix2(invsoftplus, y), x, derivative)

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
    derivative(f::typeof(loglogistic), x) = dloglogistic(x)
    derivative(f::typeof(log1mlogistic), x) = dlog1mlogistic(x)
    derivative(f::typeof(logitexp), x) = dlogitexp(x)
    derivative(f::typeof(logit1mexp), x) = dlogit1mexp(x)

    ChangesOfVariables.test_with_logabsdet_jacobian(loglogistic, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(logitexp, -x, derivative)

    ChangesOfVariables.test_with_logabsdet_jacobian(log1mlogistic, x, derivative)
    ChangesOfVariables.test_with_logabsdet_jacobian(logit1mexp, -x, derivative)
end
