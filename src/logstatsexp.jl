# Numerically stable `log`-of-statistics-of-`exp` reductions.
#
# The mean is `logsumexp(X) - log(n)`. The variance uses the *centered* formula
#
#     log var = logsumexp(2 * logsubexp(xᵢ, logmean)) - log(n - corrected)
#
# i.e. the log of the sum of squared deviations `∑ᵢ (exp(xᵢ) - mean)²`, divided by the
# count. Centering is essential for numerical stability: the raw-moment alternative
# `logsubexp(∑exp(2xᵢ), (∑exp(xᵢ))²/n)` cancels catastrophically when the variance is
# small relative to the mean (and can even overflow to `Inf` for nearly-equal inputs).
# Computing the variance therefore needs the mean first, hence the two helpers below.

"""
$(SIGNATURES)

Compute `log(mean(exp, X))` in a numerically stable way.

`X` should be an iterator of numbers.
"""
function logmeanexp(X)
    n = _known_length(X)
    if isnothing(n)
        # length not known ahead of time: count during the single pass
        lse, n = _logsumexp_and_count(X)
    else
        iszero(n) && _throw_empty()
        lse = logsumexp(X)
    end
    return lse - _log_count(lse, n)
end

"""
$(SIGNATURES)

Compute `log.(mean(exp.(X); dims=dims))` in a numerically stable way.
"""
function logmeanexp(X::AbstractArray{<:Number}; dims=:)
    dims isa Colon && isempty(X) && _throw_empty()
    lse = logsumexp(X; dims=dims)
    return lse .- _log_count(lse, _reduced_count(X, lse))
end

"""
$(SIGNATURES)

Compute `(log(mean(exp, X)), log(var(exp, X; corrected=corrected)))` in a numerically
stable way. Computing the two together is cheaper than calling [`logmeanexp`](@ref) and
[`logvarexp`](@ref) separately, since the mean is reused to center the variance.

`X` should be an iterator of real numbers.
"""
function logmeanexp_and_logvarexp(X; corrected::Bool=true)
    xs = _materialize(X)
    logmean = logmeanexp(xs)
    logsqdev = _centered_logsqdev(xs, logmean)
    return logmean, _finish_logvar(logsqdev, length(xs), corrected)
end

"""
$(SIGNATURES)

Compute `(log.(mean(exp.(X); dims)), log.(var(exp.(X); dims, corrected)))` in a
numerically stable way, reusing the mean to center the variance.
"""
function logmeanexp_and_logvarexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true)
    _require_real_array(X)
    logmean = logmeanexp(X; dims=dims)
    return logmean, _centered_logvar(X, logmean, dims, corrected)
end

# dispatch on `dims` so the return type is concrete (no `Union` of scalar/array results)
_centered_logvar(X, logmean, ::Colon, corrected::Bool) =  # scalar reduction, no temporary
    _finish_logvar(_centered_logsqdev(X, logmean), length(X), corrected)
_centered_logvar(X, logmean, dims, corrected::Bool) =
    _finish_logvar(logsumexp(2 .* logsubexp.(X, logmean); dims=dims), _reduced_count(X, logmean), corrected)

"""
$(SIGNATURES)

Compute `log(var(exp, X; corrected=corrected))` in a numerically stable way.

`X` should be an iterator of real numbers.
"""
logvarexp(X; corrected::Bool=true) = last(logmeanexp_and_logvarexp(X; corrected=corrected))

"""
$(SIGNATURES)

Compute `log.(var(exp.(X); dims=dims, corrected=corrected))` in a numerically stable way.
"""
logvarexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true) =
    last(logmeanexp_and_logvarexp(X; dims=dims, corrected=corrected))

"""
$(SIGNATURES)

Compute `log(std(exp, X; corrected=corrected))` in a numerically stable way.

`X` should be an iterator of real numbers.
"""
logstdexp(X; corrected::Bool=true) = logvarexp(X; corrected=corrected) / 2

"""
$(SIGNATURES)

Compute `log.(std(exp.(X); dims=dims, corrected=corrected))` in a numerically stable way.
"""
logstdexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true) =
    logvarexp(X; dims=dims, corrected=corrected) / 2

"""
$(SIGNATURES)

Compute `(log(mean(exp, X)), log(std(exp, X; corrected=corrected)))` in a numerically
stable way, reusing the mean to center the variance.

`X` should be an iterator of real numbers.
"""
function logmeanexp_and_logstdexp(X; corrected::Bool=true)
    logmean, logvar = logmeanexp_and_logvarexp(X; corrected=corrected)
    return logmean, logvar / 2
end

"""
$(SIGNATURES)

Compute `(log.(mean(exp.(X); dims)), log.(std(exp.(X); dims, corrected)))` in a
numerically stable way, reusing the mean to center the variance.
"""
function logmeanexp_and_logstdexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true)
    logmean, logvar = logmeanexp_and_logvarexp(X; dims=dims, corrected=corrected)
    return logmean, logvar / 2
end

# ---- internal helpers ----

_throw_empty() = throw(ArgumentError("reducing over an empty collection is not allowed"))

# `log(n)` in the (real) floating point type of `R`, so that no unwanted promotion of
# `R` (e.g. `Float32` to `Float64`) takes place. Works for scalar and array `R` alike.
# `n == 0` gives `log(0) == -Inf`, which produces the expected `NaN`/`-Inf` results.
_log_count(R, n::Integer) = log(convert(real(float(eltype(R))), n))

# number of elements reduced into each entry of `R`. When `R` is empty (the reduction
# produced no cells, e.g. `X` is empty along a dimension that is not being reduced) the
# count is irrelevant — the result is empty — so avoid dividing by zero.
_reduced_count(X::AbstractArray, R) = isempty(R) ? 0 : length(X) ÷ length(R)

_require_real(x::Real) = x
_require_real(x) = throw(ArgumentError("logvarexp and logstdexp require real inputs"))

# variance/std require real inputs; reject a non-real element type up front (with the same
# message as `_require_real`) so the array `dims` paths fail cleanly instead of hitting a
# `MethodError` deep inside `logsubexp`.
_require_real_array(X::AbstractArray{<:Real}) = X
_require_real_array(X::AbstractArray) = throw(ArgumentError("logvarexp and logstdexp require real inputs"))

_known_length(X) = _known_length(Base.IteratorSize(typeof(X)), X)
_known_length(::Union{Base.HasLength,Base.HasShape}, X) = length(X)
_known_length(_, X) = nothing

# Variance is centered, so we need to traverse the data twice (mean, then deviations).
# Known re-iterable containers are traversed in place; any other iterator is materialized
# once, so that one-shot iterators (e.g. `Iterators.Stateful`) are handled correctly.
# (`IteratorSize` cannot be used to detect re-iterability: `Stateful` reports `HasLength`
# on some Julia versions yet is single-use.)
_materialize(X) = collect(X)
_materialize(X::Union{AbstractArray,Tuple,NamedTuple,AbstractRange}) = X

# log of the (centered) squared deviation of a single point, `2 * logsubexp(xᵢ, logmean)`
# (i.e. `log((exp(xᵢ) - mean)^2)`). This is the term summed to form the variance.
_logsqdev_term(x, logmean) = 2 * logsubexp(_require_real(x), logmean)

# log of the sum of squared deviations, `logsumexp(2 * logsubexp(xᵢ, logmean))`,
# accumulated in a single pass without allocating an intermediate array.
function _centered_logsqdev(X, logmean)
    next = iterate(X)
    isnothing(next) && _throw_empty()
    x, state = next
    acc = _logsqdev_term(x, logmean)
    while true
        next = iterate(X, state)
        isnothing(next) && break
        x, state = next
        acc = _logsumexp_onepass_op(acc, _logsqdev_term(x, logmean))
    end
    return _logsumexp_onepass_result(acc)
end

# divide the squared-deviation sum by the count, in log space. When the (corrected) count
# is non-positive — a single element with `corrected=true`, or an empty reduction — the
# numerator and `log` of the (clamped-to-zero) denominator are both `-Inf`, giving `NaN`,
# matching `var`. Clamping to zero also avoids `log(-1)` for an empty `corrected=true` case.
_finish_logvar(logsqdev, n::Integer, corrected::Bool) =
    logsqdev .- _log_count(logsqdev, max(0, corrected ? n - 1 : n))

# Single pass over a general iterator: (logsumexp(X), count).
function _logsumexp_and_count(X)
    next = iterate(X)
    isnothing(next) && _throw_empty()
    x, state = next
    acc = x
    n = 1
    while true
        next = iterate(X, state)
        isnothing(next) && break
        x, state = next
        acc = _logsumexp_onepass_op(acc, x)
        n += 1
    end
    return _logsumexp_onepass_result(acc), n
end
