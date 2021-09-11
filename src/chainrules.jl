ChainRulesCore.@scalar_rule(xlogx(x::Real), (1 + log(x),))
ChainRulesCore.@scalar_rule(xlogy(x::Real, y::Real), (log(y), x / y,))
ChainRulesCore.@scalar_rule(xlog1py(x::Real, y::Real), (log1p(y), x / (1 + y),))

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
