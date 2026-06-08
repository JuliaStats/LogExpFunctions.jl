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
"""
function logmeanexp(X)
    lse, n = _logsumexp_count(identity, X)
    return lse - log(oftype(lse, n))
end

function logmeanexp(X::AbstractArray{<:Number}; dims=:)
    lse = logsumexp(X; dims)
    c = _log_count(lse, _reduced_count(X, lse))
    lse isa Number && return lse - c
    lse .-= c
    return lse
end

"""
$(SIGNATURES)

Compute `(log(mean(exp, X)), log(var(exp, X; corrected)))` in a numerically stable way.
Computing the two together is cheaper than calling [`logmeanexp`](@ref) and
[`logvarexp`](@ref) separately, since the mean is reused to center the variance.

`X` should be an iterator of real numbers. For an array, `dims` selects the dimensions to
reduce over, returning `(log.(mean(exp.(X); dims)), log.(var(exp.(X); dims, corrected)))`.
"""
function logmeanexp_and_logvarexp(X; corrected::Bool=true)
    xs = _materialize(X)  # variance needs two passes (mean, then deviations)
    logmean = logmeanexp(xs)
    logsqdev = _centered_logsqdev(xs, logmean)
    return logmean, _finish_logvar(logsqdev, length(xs), corrected)
end

function logmeanexp_and_logvarexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true)
    _require_real_array(X)
    logmean = logmeanexp(X; dims)
    return logmean, _centered_logvar(X, logmean, dims, corrected)
end

# dispatch on `dims` so the return type is concrete (no `Union` of scalar/array results)
_centered_logvar(X, logmean, ::Colon, corrected::Bool) =
    _finish_logvar(_centered_logsqdev(X, logmean), length(X), corrected)
_centered_logvar(X, logmean, dims, corrected::Bool) =
    _finish_logvar(logsumexp(2 .* logsubexp.(X, logmean); dims), _reduced_count(X, logmean), corrected)

"""
$(SIGNATURES)

Compute `log(var(exp, X; corrected))` in a numerically stable way.

`X` should be an iterator of real numbers. For an array, `dims` selects the dimensions to
reduce over, returning `log.(var(exp.(X); dims, corrected))`.
"""
logvarexp(X; corrected::Bool=true) = last(logmeanexp_and_logvarexp(X; corrected))
logvarexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true) =
    last(logmeanexp_and_logvarexp(X; dims, corrected))

"""
$(SIGNATURES)

Compute `log(std(exp, X; corrected))` in a numerically stable way.

`X` should be an iterator of real numbers. For an array, `dims` selects the dimensions to
reduce over, returning `log.(std(exp.(X); dims, corrected))`.
"""
logstdexp(X; corrected::Bool=true) = logvarexp(X; corrected) / 2
logstdexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true) =
    logvarexp(X; dims, corrected) / 2

"""
$(SIGNATURES)

Compute `(log(mean(exp, X)), log(std(exp, X; corrected)))` in a numerically stable way,
reusing the mean to center the variance.

`X` should be an iterator of real numbers. For an array, `dims` selects the dimensions to
reduce over, returning `(log.(mean(exp.(X); dims)), log.(std(exp.(X); dims, corrected)))`.
"""
function logmeanexp_and_logstdexp(X; corrected::Bool=true)
    logmean, logvar = logmeanexp_and_logvarexp(X; corrected)
    return logmean, logvar / 2
end
function logmeanexp_and_logstdexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true)
    logmean, logvar = logmeanexp_and_logvarexp(X; dims, corrected)
    return logmean, logvar / 2
end

# ---- internal helpers ----

_throw_empty() = throw(ArgumentError("reducing over an empty collection is not allowed"))

# `log(n)` in the (real) float type of `R`, avoiding promotion (e.g. `Float32` to `Float64`).
# Works for scalar and array `R`; `n == 0` gives `-Inf`, yielding the expected `NaN`/`-Inf`.
_log_count(R, n::Integer) = log(convert(real(float(eltype(R))), n))

# number of elements reduced into each entry of `R` (empty `R` ⇒ 0, avoiding division by zero)
_reduced_count(X::AbstractArray, R) = isempty(R) ? 0 : length(X) ÷ length(R)

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

# `logsumexp(2 * logsubexp(xᵢ, logmean))`, accumulated in a single pass
_centered_logsqdev(X, logmean) = first(_logsumexp_count(Base.Fix2(_logsqdev_term, logmean), X))

# divide the squared-deviation sum by the count in log space; a non-positive count (a single
# element with `corrected=true`, or an empty reduction) gives `NaN`, matching `var`
_finish_logvar(logsqdev, n::Integer, corrected::Bool) =
    logsqdev .- _log_count(logsqdev, max(0, corrected ? n - 1 : n))

# one pass over an iterator, returning `(logsumexp(f(xᵢ)), count)`
function _logsumexp_count(f, X)
    next = iterate(X)
    isnothing(next) && _throw_empty()
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
