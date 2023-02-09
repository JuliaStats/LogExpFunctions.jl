module LogExpFunctionsChangesOfVariablesExt

using LogExpFunctions
import ChangesOfVariables
import LogExpFunctions.IrrationalConstants

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(log1pexp), x::Real)
    y = log1pexp(x)
    return y, x - y
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logexpm1), x::Real)
    y = logexpm1(x)
    return y, x - y
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(log1mexp), x::Real)
    y = log1mexp(x)
    return y, x - y
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(log2mexp), x::Real)
    y = log2mexp(x)
    return y, x - y
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logit), x::Real)
    y = logit(x)
    y, -log(x * (1 - x))
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logistic), x::Real)
    y = logistic(x)
    y, log(y * (1 - y))
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logcosh), x::Real)
    abs_x = abs(x)
    a = - 2 * abs_x
    z = log1pexp(a)
    y = abs_x + z - IrrationalConstants.logtwo
    return y, log1mexp(a) - z
end

end # module
