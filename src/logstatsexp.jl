"""
$(SIGNATURES)

Compute `log(mean(exp, X))`.

`X` should be an iterator of numbers.
The result is computed in a numerically stable way.
"""
function logmeanexp(X)
    n = _iterator_length(X)
    if isnothing(n)
        lse, n = _logsumexp_count(X)
    else
        n == 0 && throw(ArgumentError("reducing over an empty collection is not allowed"))
        lse = logsumexp(X)
    end
    return lse - log(_convert_count(lse, n))
end

"""
$(SIGNATURES)

Compute `log.(mean(exp.(X); dims=dims))`.

The result is computed in a numerically stable way.
"""
function logmeanexp(X::AbstractArray{<:Number}; dims=:)
    R = logsumexp(X; dims=dims)
    n = _reduced_count(X, R)
    logn = log(_array_count_type(R, n))
    return R .- logn
end

"""
$(SIGNATURES)

Compute `log(var(exp, X; corrected=corrected))`.

`X` should be an iterator of real numbers.
The result is computed in a numerically stable way.
"""
function logvarexp(X; corrected::Bool=true, logmean=nothing)
    n = _iterator_length(X)
    if isnothing(n)
        R, n = isnothing(logmean) ? _logvariance_terms(X) : _logvariance_terms(X, logmean)
    else
        n == 0 && throw(ArgumentError("reducing over an empty collection is not allowed"))
        R = isnothing(logmean) ? _logvariance_terms_reiterable(X, n) :
            _logvariance_terms_reiterable(X, n, logmean)
    end
    denom = corrected ? n - 1 : n
    denom == 0 && return oftype(float(R), NaN)
    return R - log(_convert_count(R, denom))
end

"""
$(SIGNATURES)

Compute `log.(var(exp.(X); dims=dims, corrected=corrected))`.

The result is computed in a numerically stable way.
"""
function logvarexp(
    X::AbstractArray{<:Real}; dims=:, corrected::Bool=true, logmean=logmeanexp(X; dims=dims)
)
    R = logsumexp(2 * logsubexp.(X, logmean); dims=dims)
    n = _reduced_count(X, R)
    denom = corrected ? n - 1 : n
    denom == 0 && return _nan_like(R)
    logdenom = log(_array_count_type(R, denom))
    return R .- logdenom
end

"""
$(SIGNATURES)

Compute `log(std(exp, X; corrected=corrected))`.

`X` should be an iterator of real numbers.
The result is computed in a numerically stable way.
"""
function logstdexp(X; corrected::Bool=true, logmean=nothing)
    return logvarexp(X; corrected=corrected, logmean=logmean) / 2
end

"""
$(SIGNATURES)

Compute `log.(std(exp.(X); dims=dims, corrected=corrected))`.

The result is computed in a numerically stable way.
"""
function logstdexp(
    X::AbstractArray{<:Real}; dims=:, corrected::Bool=true, logmean=logmeanexp(X; dims=dims)
)
    return logvarexp(X; dims=dims, corrected=corrected, logmean=logmean) / 2
end

function _logsumexp_count(X)
    state = iterate(X)
    state === nothing && throw(ArgumentError("reducing over an empty collection is not allowed"))
    x, iter_state = state
    acc = x
    n = 1
    while true
        state = iterate(X, iter_state)
        state === nothing && break
        x, iter_state = state
        n += 1
        acc = _logsumexp_onepass_op(acc, x)
    end
    return _logsumexp_onepass_result(acc), n
end

_convert_count(x, n::Integer) = convert(typeof(x), n)

function _logvariance_terms(X)
    lse, lse2, n = _logsumexp2_count(X)
    logn = log(_convert_count(lse, n))
    return logsubexp(lse2, 2lse - logn), n
end
function _logvariance_terms_reiterable(X, n::Integer)
    lse = logsumexp(X)
    lse2 = logsumexp((2 * _require_real(x) for x in X))
    logn = log(_convert_count(lse, n))
    return logsubexp(lse2, 2lse - logn)
end
function _logvariance_terms_reiterable(X, n::Integer, logmean)
    logmean = _require_real(logmean)
    return logsumexp((2 * logsubexp(_require_real(x), logmean) for x in X))
end
function _logvariance_terms(X, logmean)
    logmean = _require_real(logmean)
    R, n = _logsumexp_count((2 * logsubexp(_require_real(x), logmean) for x in X))
    return R, n
end
function _logsumexp2_count(X)
    state = iterate(X)
    state === nothing && throw(ArgumentError("reducing over an empty collection is not allowed"))
    x, iter_state = state
    x = _require_real(x)
    acc = x
    acc2 = 2x
    n = 1
    while true
        state = iterate(X, iter_state)
        state === nothing && break
        x, iter_state = state
        x = _require_real(x)
        n += 1
        acc = _logsumexp_onepass_op(acc, x)
        acc2 = _logsumexp_onepass_op(acc2, 2x)
    end
    return _logsumexp_onepass_result(acc), _logsumexp_onepass_result(acc2), n
end

_reduced_count(X::AbstractArray, R::Number) = length(X)
_reduced_count(X::AbstractArray, R::AbstractArray) = length(X) ÷ length(R)
_array_count_type(R::Number, n::Integer) = convert(typeof(R), n)
_array_count_type(R::AbstractArray{<:Number}, n::Integer) = convert(eltype(R), n)
_nan_like(R::Number) = oftype(float(R), NaN)
_nan_like(R::AbstractArray{<:Number}) = fill!(R, convert(eltype(R), NaN))

_iterator_length(X) = _iterator_length(Base.IteratorSize(typeof(X)), X)
_iterator_length(::Base.HasLength, X) = length(X)
_iterator_length(::Base.HasShape, X) = length(X)
_iterator_length(::Base.SizeUnknown, X) = nothing
_iterator_length(::Any, X) = nothing

_require_real(x::Real) = x
_require_real(x) = throw(ArgumentError("logvarexp and logstdexp require real inputs"))
