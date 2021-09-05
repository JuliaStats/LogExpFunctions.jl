ChainRulesCore.@scalar_rule(xlogx(x::Real), (1 + log(x),))
ChainRulesCore.@scalar_rule(xlogy(x::Real, y::Real), (log(y), x / y,))

ChainRulesCore.@scalar_rule(logistic(x::Real), (Ω * (1 - Ω),))
ChainRulesCore.@scalar_rule(logit(x::Real), (inv(x * (1 - x)),))
ChainRulesCore.@scalar_rule(log1psq(x::Real), (2 * x / (1 + x^2),))
ChainRulesCore.@scalar_rule(log1pexp(x::Real), (logistic(x),))
ChainRulesCore.@scalar_rule(log1mexp(x::Real), (-exp(x - Ω),))
ChainRulesCore.@scalar_rule(log2mexp(x::Real), (-exp(x - Ω),))
ChainRulesCore.@scalar_rule(logexpm1(x::Real), (exp(x - Ω),))

ChainRulesCore.@scalar_rule(logaddexp(x::Real, y::Real), (exp(x - Ω), exp(y - Ω)))
ChainRulesCore.@scalar_rule(
    logsubexp(x::Real, y::Real),
    (x > y ? exp(x - Ω) : -exp(x - Ω), x > y ? -exp(y - Ω) : exp(y - Ω)),
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

function ChainRulesCore.frule(
    (_, _, Δx), ::typeof(softmax!), r::AbstractArray{<:Real}, x::AbstractArray{<:Real},
)
    softmax!(r, x)
    _Δx = reshape(Δx, size(r))
    Δr = r .* (_Δx .- LinearAlgebra.dot(r, _Δx))
    return r, Δr
end
function ChainRulesCore.rrule(
    ::typeof(softmax!), r::AbstractArray{<:Real}, x::AbstractArray{<:Real},
)
    softmax!(r, x)
    project_x = ChainRulesCore.ProjectTo(x)
    rcopy = copy(reshape(r, size(x)))
    function softmax!_pullback(r̄)
        _r̄ = reshape(r̄, size(rcopy))
        x̄ = ChainRulesCore.InplaceableThunk(
            Δ -> Δ .+= rcopy .* (_r̄ .- LinearAlgebra.dot(rcopy, _r̄)),
            ChainRulesCore.@thunk(project_x(rcopy .* (_r̄ .- LinearAlgebra.dot(rcopy, _r̄)))),
        )
        return ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), x̄
    end
    return r, softmax!_pullback
end
