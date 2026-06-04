# Numerically stable `log`-of-statistics-of-`exp` reductions.
#
# Everything here is derived from at most two `logsumexp` accumulators,
#
#     lse  = log ∑ᵢ exp(xᵢ)          (first moment)
#     lse2 = log ∑ᵢ exp(2xᵢ)         (second moment)
#
# together with the number of elements `n`. For a general iterator both
# accumulators are computed in a single pass (important for one-shot iterators
# such as `Iterators.Stateful`); for an `AbstractArray` we reuse the optimized
# `logsumexp` and take `n` from `length`, which is cheap.

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
    lse = logsumexp(X; dims=dims)
    return lse .- _log_count(lse, _reduced_count(X, lse))
end

"""
$(SIGNATURES)

Compute `log(var(exp, X; corrected=corrected))` in a numerically stable way.

`X` should be an iterator of real numbers.
"""
function logvarexp(X; corrected::Bool=true)
    lse, lse2, n = _logmoments(X)
    return _logvar(lse, lse2, n, corrected)
end

"""
$(SIGNATURES)

Compute `log.(var(exp.(X); dims=dims, corrected=corrected))` in a numerically stable way.
"""
function logvarexp(X::AbstractArray{<:Real}; dims=:, corrected::Bool=true)
    lse, lse2, n = _logmoments(X, dims)
    return _logvar(lse, lse2, n, corrected)
end

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
logstdexp(X::AbstractArray{<:Real}; dims=:, corrected::Bool=true) =
    logvarexp(X; dims=dims, corrected=corrected) ./ 2

"""
$(SIGNATURES)

Compute `(log(mean(exp, X)), log(var(exp, X; corrected=corrected)))` in a numerically
stable way, using a single pass over the data.

`X` should be an iterator of real numbers.
"""
function logmeanexp_and_logvarexp(X; corrected::Bool=true)
    lse, lse2, n = _logmoments(X)
    return lse - _log_count(lse, n), _logvar(lse, lse2, n, corrected)
end

"""
$(SIGNATURES)

Compute `(log.(mean(exp.(X); dims)), log.(var(exp.(X); dims, corrected)))` in a numerically
stable way.
"""
function logmeanexp_and_logvarexp(X::AbstractArray{<:Real}; dims=:, corrected::Bool=true)
    lse, lse2, n = _logmoments(X, dims)
    return lse .- _log_count(lse, n), _logvar(lse, lse2, n, corrected)
end

"""
$(SIGNATURES)

Compute `(log(mean(exp, X)), log(std(exp, X; corrected=corrected)))` in a numerically
stable way, using a single pass over the data.

`X` should be an iterator of real numbers.
"""
function logmeanexp_and_logstdexp(X; corrected::Bool=true)
    logmean, logvar = logmeanexp_and_logvarexp(X; corrected=corrected)
    return logmean, logvar / 2
end

"""
$(SIGNATURES)

Compute `(log.(mean(exp.(X); dims)), log.(std(exp.(X); dims, corrected)))` in a numerically
stable way.
"""
function logmeanexp_and_logstdexp(X::AbstractArray{<:Real}; dims=:, corrected::Bool=true)
    logmean, logvar = logmeanexp_and_logvarexp(X; dims=dims, corrected=corrected)
    return logmean, logvar ./ 2
end

# ---- internal helpers ----

_throw_empty() = throw(ArgumentError("reducing over an empty collection is not allowed"))

# `log(n)` in the (real) floating point type of `R`, so that no unwanted promotion of
# `R` (e.g. `Float32` to `Float64`) takes place. Works for scalar and array `R` alike.
# `n == 0` gives `log(0) == -Inf`, which produces the expected `NaN`/`-Inf` results.
_log_count(R, n::Integer) = log(convert(real(float(eltype(R))), n))

# number of elements reduced into each entry of `R`
_reduced_count(X::AbstractArray, R) = length(X) ÷ length(R)

_require_real(x::Real) = x
_require_real(x) = throw(ArgumentError("logvarexp and logstdexp require real inputs"))

_known_length(X) = _known_length(Base.IteratorSize(typeof(X)), X)
_known_length(::Union{Base.HasLength,Base.HasShape}, X) = length(X)
_known_length(_, X) = nothing

# `log(var)` from the raw log-moments. The squared-deviation sum is
# `∑ᵢ (exp(xᵢ) - mean)² = ∑ᵢ exp(2xᵢ) - (∑ᵢ exp(xᵢ))² / n`, i.e.
# `logsubexp(lse2, 2 * lse - log(n))` in log space. For a single element
# (`n - corrected == 0`) the numerator is `-Inf` and `log` of the denominator is also
# `-Inf`, so the result is `NaN`, matching `var`.
function _logvar(lse, lse2, n::Integer, corrected::Bool)
    logn = _log_count(lse2, n)
    logdenom = _log_count(lse2, corrected ? n - 1 : n)
    return @. logsubexp(lse2, 2 * lse - logn) - logdenom
end

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

# Single pass over a general iterator of reals: (logsumexp(X), logsumexp(2X), count).
function _logmoments(X)
    next = iterate(X)
    isnothing(next) && _throw_empty()
    x, state = next
    x = _require_real(x)
    acc = x
    acc2 = 2x
    n = 1
    while true
        next = iterate(X, state)
        isnothing(next) && break
        x, state = next
        x = _require_real(x)
        acc = _logsumexp_onepass_op(acc, x)
        acc2 = _logsumexp_onepass_op(acc2, 2x)
        n += 1
    end
    return _logsumexp_onepass_result(acc), _logsumexp_onepass_result(acc2), n
end

# Array version: reuse the optimized `logsumexp` and take the count from `length`.
function _logmoments(X::AbstractArray{<:Real}, dims)
    # full reduction: a single pass avoids allocating the `2 .* X` temporary
    dims === Colon() && return _logmoments(X)
    lse = logsumexp(X; dims=dims)
    lse2 = logsumexp(2 .* X; dims=dims)
    return lse, lse2, _reduced_count(X, lse)
end
