# Numerically stable `log`-of-statistics-of-`exp` reductions:
# logmeanexp, logvarexp, logstdexp.

"""
$(SIGNATURES)

Compute `log(mean(exp, X))` in a numerically stable way.

`X` should be an iterator of numbers. For an array, `dims` selects the dimensions to reduce
over, returning `log.(mean(exp.(X); dims))`.

See also [`logmeanexp!`](@ref).
"""
function logmeanexp(X)
    lse, n = _logsumexp_count(identity, X)
    return lse - log(oftype(lse, n))
end
logmeanexp(X::AbstractArray{<:Number}; dims=:) = _logmeanexp(X, dims)
_logmeanexp(X, ::Colon) = (lse = logsumexp(X); lse - log(oftype(lse, length(X))))
_logmeanexp(X, dims) = logmeanexp!(_reduced_similar(X, dims), X)

"""
$(SIGNATURES)

Compute [`logmeanexp`](@ref) of `X` over the singleton dimensions of `out`, and write the
result to `out`.
"""
function logmeanexp!(out::AbstractArray, X::AbstractArray{<:Number})
    logsumexp!(out, X)
    return out .-= log(_reduced_count(X, out))
end


"""
$(SIGNATURES)

Compute `log(var(exp, X; corrected))` in a numerically stable way.

`X` should be an iterator of real numbers. For an array, `dims` selects the dimensions to
reduce over, returning `log.(var(exp.(X); dims, corrected))`.

See also [`logvarexp!`](@ref).
"""
function logvarexp(X; corrected::Bool=true)
    xs = collect(X)
    return _finish_logvar(_centered_logsqdev(xs, logmeanexp(xs)), length(xs), corrected)
end
logvarexp(X::Union{Tuple,NamedTuple}; corrected::Bool=true) = _logvarexp(X, :, corrected)
logvarexp(X::AbstractArray{<:Real}; dims=:, corrected::Bool=true) = _logvarexp(X, dims, corrected)
logvarexp(X::AbstractArray; dims=:, corrected::Bool=true) = _throw_not_real()
_logvarexp(X, ::Colon, corrected::Bool) = _finish_logvar(_centered_logsqdev(X, logmeanexp(X)), length(X), corrected)
_logvarexp(X, dims, corrected::Bool) = logvarexp!(_reduced_similar(X, dims), X; corrected)

"""
$(SIGNATURES)

Compute [`logvarexp`](@ref) of `X` over the singleton dimensions of `out`, and write the
result to `out`.
"""
function logvarexp!(out::AbstractArray, X::AbstractArray{<:Real}; corrected::Bool=true)
    logmeanexp!(out, X)
    logsumexp!(out, _LogSqDev(X, out))
    out .-= log(max(0, _reduced_count(X, out) - corrected))
    return out
end
logvarexp!(out::AbstractArray, X::AbstractArray{<:Number}; corrected::Bool=true) = _throw_not_real()


"""
$(SIGNATURES)

Compute `log(std(exp, X; corrected))` in a numerically stable way.

`X` should be an iterator of real numbers. For an array, `dims` selects the dimensions to
reduce over, returning `log.(std(exp.(X); dims, corrected))`.
"""
logstdexp(X; corrected::Bool=true) = logvarexp(X; corrected) / 2
logstdexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true) = _halve!(logvarexp(X; dims, corrected))

# ---- internal helpers ----

_reduced_count(X::AbstractArray, R) = isempty(R) ? 0 : length(X) ÷ length(R)
_reduced_similar(X, dims) = similar(X, float(eltype(X)), Base.reduced_indices(axes(X), dims))

_halve!(v::Number) = v / 2
_halve!(v::AbstractArray) = v ./= 2

_throw_not_real() = throw(ArgumentError("logvarexp and logstdexp require real inputs"))

_logsqdev_term(x::Real, logmean) = 2 * logsubexp(x, logmean)
_logsqdev_term(x, logmean) = _throw_not_real()
_centered_logsqdev(X, logmean) = first(_logsumexp_count(Base.Fix2(_logsqdev_term, logmean), X))

function _finish_logvar(logsqdev::Number, n::Integer, corrected::Bool)
    return logsqdev - oftype(logsqdev, log(max(0, n - corrected)))
end

# one pass over an iterator, returning `(logsumexp(f(xᵢ)), count)`
function _logsumexp_count(f, X)
    next = iterate(X)
    isnothing(next) && throw(ArgumentError("reducing over an empty collection is not allowed"))
    x, state = next
    acc = f(x)
    n = 1
    while true
        next = iterate(X, state)
        isnothing(next) && break
        x, state = next
        acc = _logsumexp_onepass_op(acc, f(x))
        n += 1
    end
    return _logsumexp_onepass_result(acc), n
end

# lazy array of the centered terms `2 * logsubexp(xᵢ, meanⱼ)`, with `mean` broadcast over the
# reduced (singleton) dimensions. A plain `AbstractArray` (not a `Broadcasted`) so that
# `reduce`/`reducedim!` honor offset axes and a concrete element type.
struct _LogSqDev{T,N,XT<:AbstractArray,MT<:AbstractArray} <: AbstractArray{T,N}
    x::XT
    mean::MT
end
_LogSqDev(x::AbstractArray, mean::AbstractArray) =
    _LogSqDev{float(eltype(x)),ndims(x),typeof(x),typeof(mean)}(x, mean)
Base.size(s::_LogSqDev) = size(s.x)
Base.axes(s::_LogSqDev) = axes(s.x)
Base.IndexStyle(::Type{<:_LogSqDev}) = IndexCartesian()
Base.@propagate_inbounds function Base.getindex(s::_LogSqDev{T,N}, I::Vararg{Int,N}) where {T,N}
    j = map((i, ax) -> ifelse(length(ax) == 1, first(ax), i), I, axes(s.mean))
    return convert(T, _logsqdev_term(s.x[I...], s.mean[j...]))
end
