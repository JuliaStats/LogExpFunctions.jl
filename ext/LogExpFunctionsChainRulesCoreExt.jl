module LogExpFunctionsChainRulesCoreExt

using LogExpFunctions
import ChainRulesCore

import LogExpFunctions.LinearAlgebra

function _Ω_∂_xlogx(x::Real)
    logx = log(x)
    y = x * logx
    Ω = iszero(x) ? zero(y) : y
    ∂x = 1 + logx
    return Ω, ∂x
end
function ChainRulesCore.frule((_, Δx), ::typeof(xlogx), x::Real)
    Ω, ∂x = _Ω_∂_xlogx(x)
    ΔΩ = ∂x * Δx
    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(xlogx), x::Real)
    Ω, ∂x = _Ω_∂_xlogx(x)
    xlogx_pullback(ΔΩ) = (ChainRulesCore.NoTangent(), ∂x * ΔΩ)
    return Ω, xlogx_pullback
end

function _Ω_∂_xlogy(x::Real, y::Real)
    logy = log(y)
    z = x * logy
    w = x / y
    if iszero(x) && !isnan(y)
        Ω = zero(z)
        ∂y = zero(w)
    else
        Ω = z
        ∂y = w
    end
    ∂x = logy
    return Ω, ∂x, ∂y
end
function ChainRulesCore.frule((_, Δx, Δy), ::typeof(xlogy), x::Real, y::Real)
    Ω, ∂x, ∂y = _Ω_∂_xlogy(x, y)
    ΔΩ = muladd(∂x, Δx, ∂y * Δy)
    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(xlogy), x::Real, y::Real)
    Ω, ∂x, ∂y = _Ω_∂_xlogy(x, y)
    xlogy_pullback(ΔΩ) = (ChainRulesCore.NoTangent(), ∂x * ΔΩ, ∂y * ΔΩ)
    return Ω, xlogy_pullback
end

function _Ω_∂_xlog1py(x::Real, y::Real)
    log1py = log1p(y)
    z = x * log1py
    w = x / (1 + y)
    if iszero(x) && !isnan(y)
        Ω = zero(z)
        ∂y = zero(w)
    else
        Ω = z
        ∂y = w
    end
    ∂x = log1py
    return Ω, ∂x, ∂y
end
function ChainRulesCore.frule((_, Δx, Δy), ::typeof(xlog1py), x::Real, y::Real)
    Ω, ∂x, ∂y = _Ω_∂_xlog1py(x, y)
    ΔΩ = muladd(∂x, Δx, ∂y * Δy)
    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(xlog1py), x::Real, y::Real)
    Ω, ∂x, ∂y = _Ω_∂_xlog1py(x, y)
    xlog1py_pullback(ΔΩ) = (ChainRulesCore.NoTangent(), ∂x * ΔΩ, ∂y * ΔΩ)
    return Ω, xlog1py_pullback
end

function _Ω_∂_xexpx(x::Real)
    expx = exp(x)
    if iszero(expx)
        Ω = expx
        ∂x = expx
    else
        Ω = x * expx
        ∂x = (1 + x) * expx
    end
    return Ω, ∂x
end
function ChainRulesCore.frule((_, Δx), ::typeof(xexpx), x::Real)
    Ω, ∂x = _Ω_∂_xexpx(x)
    ΔΩ = ∂x * Δx
    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(xexpx), x::Real)
    Ω, ∂x = _Ω_∂_xexpx(x)
    xexpx_pullback(ΔΩ) = (ChainRulesCore.NoTangent(), ∂x * ΔΩ)
    return Ω, xexpx_pullback
end

function _Ω_∂_xexpy(x::Real, y::Real)
    expy = exp(y)
    result = x * expy
    Ω = iszero(expy) && !isnan(x) ? zero(result) : result
    ∂x = expy
    ∂y = Ω
    return Ω, ∂x, ∂y
end
function ChainRulesCore.frule((_, Δx, Δy), ::typeof(xexpy), x::Real, y::Real)
    Ω, ∂x, ∂y = _Ω_∂_xexpy(x, y)
    ΔΩ = muladd(∂x, Δx, ∂y * Δy)
    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(xexpy), x::Real, y::Real)
    Ω, ∂x, ∂y = _Ω_∂_xexpy(x, y)
    xexpy_pullback(ΔΩ) = (ChainRulesCore.NoTangent(), ∂x * ΔΩ, ∂y * ΔΩ)
    return Ω, xexpy_pullback
end

# logistic(x) * logistic(-x), avoiding cancellation in `1 - logistic(x)`
function _∂_logistic(x::Real)
    e = exp(-abs(x))
    return e / (1 + e)^2
end
# 2x / (1 + x^2), avoiding overflow of `x^2` for large `|x|`
_∂_log1psq(x::Real) = abs(x) < 1 ? 2 * x / (1 + x^2) : 2 / (x + inv(x))

# `(x / (1 + x) - log1p(x)) / x^2`, with the limit `-1/2` at `x = 0`. The direct form
# cancels for small `|x|`, so a series is used there. For `|x| ≥ 1/2` the numerator is
# `logmxp1(1 / (1 + x))` (`log(y) - y + 1` at `y = 1 / (1 + x)`), which is accurate; `/ x / x`
# avoids the overflow of `x^2`.
function _∂_log1pdiv(x::Real)
    if abs(x) < 0.5
        # `-1/2 + 2x/3 - 3x^2/4 + …` = ∑_{k ≥ 0} (-1)^(k + 1) (k + 1) x^k / (k + 2)
        s = -one(x) / 2
        p = one(x)
        for k in 1:precision(x)
            p *= x
            term = oftype(x, k + 1) / oftype(x, k + 2) * p
            s_new = isodd(k) ? s + term : s - term
            s_new == s && break
            s = s_new
        end
        return s
    end
    return logmxp1(inv(1 + x)) / x / x
end

# `((exp(x) - 1) (x - 1) + x) / x^2`, with the limit `1/2` at `x = 0`. The closed forms
# cancel near `0` and for `x < 0` up to `|x| ≈ 2`, where a series is used. For `x ≥ 2` the
# `exp(x) = exp(x/2)^2` factorization keeps the result finite for as long as it is
# representable, even where `exp(x)` itself overflows.
function _∂_expm1div(x::Real)
    if abs(x) < 2
        # `1/2 + x/3 + x^2/8 + …` = ∑_{k ≥ 0} (k + 1) x^k / (k + 2)!
        s = one(x) / 2
        t = s
        for k in 1:precision(x)
            t *= x / oftype(x, k + 2)
            term = oftype(x, k + 1) * t
            s_new = s + term
            s_new == s && break
            s = s_new
        end
        return s
    elseif x > 0
        h = exp(x / 2)
        return h * ((x - 1) / x) * (h / x) + inv(x)^2
    end
    return (exp(x) * (x - 1) + one(x)) / x / x
end

ChainRulesCore.@scalar_rule(logistic(x::Real), (_∂_logistic(x),))
ChainRulesCore.@scalar_rule(logit(x::Real), (inv(x * (1 - x)),))
ChainRulesCore.@scalar_rule(logcosh(x::Real), tanh(x))
ChainRulesCore.@scalar_rule(logabssinh(x::Real), coth(x))
ChainRulesCore.@scalar_rule(logabstanh(x::Real), 2 / sinh(2 * x))
ChainRulesCore.@scalar_rule(log1psq(x::Real), (_∂_log1psq(x),))
ChainRulesCore.@scalar_rule(log1pexp(x::Real), (logistic(x),))
ChainRulesCore.@scalar_rule(log1mexp(x::Real), (-inv(expm1(-x)),))
ChainRulesCore.@scalar_rule(log2mexp(x::Real), (-exp(x - Ω),))
ChainRulesCore.@scalar_rule(logexpm1(x::Real), (-inv(expm1(-x)),))
ChainRulesCore.@scalar_rule(log1pmx(x::Real), (-x / (x + 1),))
ChainRulesCore.@scalar_rule(logmxp1(x::Real), ((1 - x) / x,))
ChainRulesCore.@scalar_rule(log1pdiv(x::Real), (_∂_log1pdiv(float(x)),))
ChainRulesCore.@scalar_rule(expm1div(x::Real), (_∂_expm1div(float(x)),))
ChainRulesCore.@scalar_rule(loglogistic(x::Real), (logistic(-x),))
ChainRulesCore.@scalar_rule(log1mlogistic(x::Real), (-logistic(x),))
ChainRulesCore.@scalar_rule(logitexp(x::Real), (-inv(expm1(x)),))
ChainRulesCore.@scalar_rule(logit1mexp(x::Real), (inv(expm1(x)),))

# the one with respect to the larger argument as `1 - t` (no cancellation since
# 0 ≤ t ≤ 1/2 for `logaddexp` and t ≤ 0 for `logsubexp`).
# The handling of `x == y` mirrors the primal implementations.
# (0 ≤ t ≤ 1/2 for `logaddexp` and t ≤ 0 for `logsubexp`, so `1 - t` is accurate).
# `x - y` is `NaN` if `x == y == ±Inf`.
function _∂_logaddexp(x::Real, y::Real)
    Δ = x == y && isinf(x) ? zero(x - y) : abs(x - y)
    t = logistic(-Δ)
    return x >= y ? (1 - t, t) : (t, 1 - t)
end
function _∂_logsubexp(x::Real, y::Real)
    Δ = x == y == -Inf ? zero(x - y) : abs(x - y)
    t = -inv(expm1(Δ))
    return x >= y ? (1 - t, t) : (t, 1 - t)
end
ChainRulesCore.@scalar_rule(
    logaddexp(x::Real, y::Real),
    @setup((∂x, ∂y) = _∂_logaddexp(x, y)),
    (∂x, ∂y),
)
ChainRulesCore.@scalar_rule(
    logsubexp(x::Real, y::Real),
    @setup((∂x, ∂y) = _∂_logsubexp(x, y)),
    (∂x, ∂y),
)

function ChainRulesCore.frule((_, Δx), ::typeof(logsumexp), x::AbstractArray{<:Real}; dims=:)
    Ω = logsumexp(x; dims=dims)
    ΔΩ = sum(exp.(x .- Ω) .* Δx; dims=dims)
    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(logsumexp), x::AbstractArray{<:Real}; dims=:)
    Ω = logsumexp(x; dims=dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function logsumexp_pullback(Ω̄)
        x̄ = ChainRulesCore.InplaceableThunk(
            Δ -> Δ .+= Ω̄ .* exp.(x .- Ω),
            ChainRulesCore.@thunk(project_x(Ω̄ .* exp.(x .- Ω))),
        )
        return ChainRulesCore.NoTangent(), x̄
    end
    return Ω, logsumexp_pullback
end

# no rules for mutating functions currently:
# https://juliadiff.org/ChainRulesCore.jl/stable/writing_good_rules.html#Which-functions-need-rules?
function ChainRulesCore.frule((_, Δx), ::typeof(softmax), x::AbstractArray{<:Real}; dims=:)
    Ω = softmax(x; dims=dims)
    ΔΩ = if dims === (:)
        Ω .* (Δx .- LinearAlgebra.dot(Ω, Δx))
    else
        ΩΔx = Ω .* Δx
        ΩΔx .- Ω .* sum(ΩΔx; dims=dims)
    end
    return Ω, ΔΩ
end
function ChainRulesCore.rrule(::typeof(softmax), x::AbstractArray{<:Real}; dims=:)
    Ω = softmax(x; dims=dims)
    Ωcopy = copy(Ω)
    project_x = ChainRulesCore.ProjectTo(x)
    function softmax_pullback(Ω̄)
        x̄ = if dims === (:)
            ChainRulesCore.InplaceableThunk(
                Δ -> Δ .+= Ωcopy .* (Ω̄ .- LinearAlgebra.dot(Ωcopy, Ω̄)),
                ChainRulesCore.@thunk(project_x(Ωcopy .* (Ω̄ .- LinearAlgebra.dot(Ωcopy, Ω̄)))),
            )
        else
            ΩcopyΩ̄  = Ωcopy .* Ω̄
            ChainRulesCore.InplaceableThunk(
                Δ -> Δ .+= ΩcopyΩ̄  .- Ωcopy .* sum(ΩcopyΩ̄; dims=dims),
                ChainRulesCore.@thunk(project_x(ΩcopyΩ̄  .- Ωcopy .* sum(ΩcopyΩ̄; dims=dims))),
            )
        end
        return ChainRulesCore.NoTangent(), x̄
    end
    return Ω, softmax_pullback
end

ChainRulesCore.@scalar_rule(cloglog(x::Real), (-inv((1 - x) * log1p(-x)),))
ChainRulesCore.@scalar_rule(cexpexp(x::Real), (-xexpx(-exp(x)),))

end # module
