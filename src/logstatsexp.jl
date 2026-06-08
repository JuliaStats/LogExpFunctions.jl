# Numerically stable `log`-of-statistics-of-`exp` reductions.
#
# The mean is `logsumexp(X) - log(n)`. The variance uses the centered formula
#
#     log var = logsumexp(2 * logsubexp(xᵢ, logmean)) - log(n - corrected)
#
# i.e. the log of the sum of squared deviations `∑ᵢ (exp(xᵢ) - mean)²`, divided by the count.

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

"""
$(SIGNATURES)

Compute [`logmeanexp`](@ref) of `X` over the singleton dimensions of `out`, and write the
result to `out`.
"""
function logmeanexp!(out::AbstractArray, X::AbstractArray{<:Number})
    logsumexp!(out, X)
    out .-= _log_count(out, _reduced_count(X, out))
    return out
end

_logmeanexp(X, ::Colon) = (lse = logsumexp(X); lse - _log_count(lse, length(X)))
_logmeanexp(X, dims) = logmeanexp!(_reduced_similar(X, dims), X)

"""
$(SIGNATURES)

Compute `log(var(exp, X; corrected))` in a numerically stable way.

`X` should be an iterator of real numbers. For an array, `dims` selects the dimensions to
reduce over, returning `log.(var(exp.(X); dims, corrected))`.

See also [`logvarexp!`](@ref).
"""
function logvarexp(X; corrected::Bool=true)
    xs = _materialize(X)
    logmean = logmeanexp(xs)
    return _finish_logvar(_centered_logsqdev(xs, logmean), length(xs), corrected)
end
logvarexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true) =
    _logvarexp(_require_real_array(X), dims, corrected)

"""
$(SIGNATURES)

Compute [`logvarexp`](@ref) of `X` over the singleton dimensions of `out`, and write the
result to `out`.
"""
function logvarexp!(out::AbstractArray, X::AbstractArray{<:Number}; corrected::Bool=true)
    _require_real_array(X)
    logmeanexp!(out, X)
    n = _reduced_count(X, out)
    logsumexp!(out, _LogSqDev(X, out))
    out .-= _log_count(out, max(0, corrected ? n - 1 : n))
    return out
end

_logvarexp(X, ::Colon, corrected::Bool) =
    _finish_logvar(_centered_logsqdev(X, logmeanexp(X)), length(X), corrected)
_logvarexp(X, dims, corrected::Bool) = logvarexp!(_reduced_similar(X, dims), X; corrected)

"""
$(SIGNATURES)

Compute `log(std(exp, X; corrected))` in a numerically stable way.

`X` should be an iterator of real numbers. For an array, `dims` selects the dimensions to
reduce over, returning `log.(std(exp.(X); dims, corrected))`.
"""
logstdexp(X; corrected::Bool=true) = logvarexp(X; corrected) / 2
logstdexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true) =
    _halve!(logvarexp(X; dims, corrected))

# ---- internal helpers ----

# `log(n)` in the (real) float type of `R`, avoiding promotion (e.g. `Float32` to `Float64`).
# Works for scalar and array `R`; `n == 0` gives `-Inf`, yielding the expected `NaN`/`-Inf`.
_log_count(R, n::Integer) = log(convert(real(float(eltype(R))), n))

# number of elements reduced into each entry of `R` (empty `R` ⇒ 0, avoiding division by zero)
_reduced_count(X::AbstractArray, R) = isempty(R) ? 0 : length(X) ÷ length(R)

# output array for a reduction of `X` over `dims` (singleton along the reduced dimensions)
_reduced_similar(X, dims) = similar(X, float(eltype(X)), Base.reduced_indices(axes(X), dims))

_halve!(v::Number) = v / 2
_halve!(v::AbstractArray) = v ./= 2

# variance/std require real inputs; reject a non-real element type up front for a clean error
_throw_not_real() = throw(ArgumentError("logvarexp and logstdexp require real inputs"))
_require_real(x::Real) = x
_require_real(x) = _throw_not_real()
_require_real_array(X::AbstractArray{<:Real}) = X
_require_real_array(X::AbstractArray) = _throw_not_real()

# re-iterable containers are traversed in place; any other iterator is collected once, so that
# single-use iterators (e.g. `Iterators.Stateful`) survive the variance's two passes
_materialize(X) = collect(X)
_materialize(X::Union{AbstractArray,Tuple,NamedTuple,AbstractRange}) = X

# `log((exp(xᵢ) - mean)^2) = 2 * logsubexp(xᵢ, logmean)`, the term summed for the variance
_logsqdev_term(x, logmean) = 2 * logsubexp(_require_real(x), logmean)
_centered_logsqdev(X, logmean) = first(_logsumexp_count(Base.Fix2(_logsqdev_term, logmean), X))

function _finish_logvar(logsqdev, n::Integer, corrected::Bool)
    c = _log_count(logsqdev, max(0, corrected ? n - 1 : n))
    logsqdev isa Number && return logsqdev - c
    logsqdev .-= c
    return logsqdev
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
