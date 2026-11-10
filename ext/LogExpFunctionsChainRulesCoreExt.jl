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

# logistic'(x) = logistic(x) * logistic(-x) = e / (1 + e)^2 with e = exp(-|x|)
# (avoids the cancellation in `1 - logistic(x)` for large `x`)
function _∂_logistic(x::Real)
    e = exp(-abs(x))
    return e / (1 + e)^2
end
# 2x / (1 + x^2), avoiding overflow of `x^2` for large `|x|`
_∂_log1psq(x::Real) = abs(x) < 1 ? 2 * x / (1 + x^2) : 2 / (x + inv(x))

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
ChainRulesCore.@scalar_rule(loglogistic(x::Real), (logistic(-x),))
ChainRulesCore.@scalar_rule(log1mlogistic(x::Real), (-logistic(x),))
ChainRulesCore.@scalar_rule(logitexp(x::Real), (-inv(expm1(x)),))
ChainRulesCore.@scalar_rule(logit1mexp(x::Real), (inv(expm1(x)),))

# Both `logaddexp` and `logsubexp` satisfy f(x + c, y + c) = f(x, y) + c, hence ∂x + ∂y = 1.
# The partial derivative `t` with respect to the smaller argument is computed directly,
# the one with respect to the larger argument as `1 - t` (no cancellation since
# 0 ≤ t ≤ 1/2 for `logaddexp` and t ≤ 0 for `logsubexp`).
# The handling of `x == y` mirrors the primal implementations.
function _∂_logaddexp(x::Real, y::Real)
    Δ = x == y ? zero(x - y) : abs(x - y)
    t = logistic(-Δ)
    return x >= y ? (1 - t, t) : (t, 1 - t)
end
function _∂_logsubexp(x::Real, y::Real)
    Δ = x == y && (isfinite(x) || x < 0) ? zero(x - y) : abs(x - y)
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
