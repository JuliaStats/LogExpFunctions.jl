function ChangesOfVariables.with_logabsdet_jacobian(::typeof(log1pexp), x::Real)
    y = log1pexp(x)
    return y, x - y
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logexpm1), x::Real)
    y = logexpm1(x)
    return y, (x - y)
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(log1mexp), x::Real)
    y = log1mexp(x)
    return y, (x - y)
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(log2mexp), x::Real)
    y = log2mexp(x)
    return y, (x - y)
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logit), x::Real)
    y = logit(x)
    y, log(inv(x * (1 - x)))
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logistic), x::Real)
    y = logistic(x)
    y, log(y * (1 - y))
end


#=
ChainRulesCore.@scalar_rule(log1pexp(x::Real), (logistic(x),))
ChainRulesCore.@scalar_rule(logexpm1(x::Real), (exp(x - Ω),))

ChainRulesCore.@scalar_rule(log1mexp(x::Real), (-exp(x - Ω),))

ChainRulesCore.@scalar_rule(log2mexp(x::Real), (-exp(x - Ω),))

ChainRulesCore.@scalar_rule(logistic(x::Real), (Ω * (1 - Ω),))

ChainRulesCore.@scalar_rule(logit(x::Real), (inv(x * (1 - x)),))
=#
