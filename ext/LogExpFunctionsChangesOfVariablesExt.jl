module LogExpFunctionsChangesOfVariablesExt

using LogExpFunctions
import ChangesOfVariables
import LogExpFunctions.IrrationalConstants

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(log1pexp), x::Real)
    y = log1pexp(x)
    return y, x - y
end
function ChangesOfVariables.with_logabsdet_jacobian(::typeof(softplus), x::Real)
    return ChangesOfVariables.with_logabsdet_jacobian(log1pexp, x)
end
function ChangesOfVariables.with_logabsdet_jacobian(f::Base.Fix2{typeof(softplus),<:Real}, x::Real)
    y = f(x)
    return y, f.x * (x - y)
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logexpm1), x::Real)
    y = logexpm1(x)
    return y, x - y
end
function ChangesOfVariables.with_logabsdet_jacobian(::typeof(invsoftplus), x::Real)
    return ChangesOfVariables.with_logabsdet_jacobian(logexpm1, x)
end
function ChangesOfVariables.with_logabsdet_jacobian(f::Base.Fix2{typeof(invsoftplus),<:Real}, x::Real)
    y = f(x)
    return y, f.x * (x - y)
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

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(loglogistic), x::Real)
    y = loglogistic(x)
    return y, y - x
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(log1mlogistic), x::Real)
    y = log1mlogistic(x)
    return y, x + y
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logitexp), x::Real)
    y = logitexp(x)
    return y, y - x
end

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(logit1mexp), x::Real)
    y = logit1mexp(x)
    return y, -y - x
end

end # module
