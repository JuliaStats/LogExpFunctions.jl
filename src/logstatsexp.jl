"""
$(SIGNATURES)

Compute `log(mean(exp, X))`.

`X` should be an iterator of numbers.
The result is computed in a numerically stable way.
"""
function logmeanexp(X)
    lse, n = _logsumexp_count(X)
    return lse - log(_convert_count(lse, n))
end

"""
$(SIGNATURES)

Compute `log.(mean(exp.(X); dims=dims))`.

The result is computed in a numerically stable way.
"""
function logmeanexp(X::AbstractArray{<:Number}; dims=:)
    R = logsumexp(X; dims=dims)
    n = length(X) ÷ length(R)
    return _subtract_log_count(R, n)
end

"""
$(SIGNATURES)

Compute `log(var(exp, X; corrected=corrected))`.

`X` should be an iterator of numbers.
The result is computed in a numerically stable way.
"""
function logvarexp(X; corrected::Bool=true, logmean=logmeanexp(X))
    R = logsumexp((2logsubexp(x, logmean) for x in X))
    n = _count_elements(X)
    denom = corrected ? n - 1 : n
    return R - log(_convert_count(R, denom))
end

"""
$(SIGNATURES)

Compute `log.(var(exp.(X); dims=dims, corrected=corrected))`.

The result is computed in a numerically stable way.
"""
function logvarexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true, logmean=logmeanexp(X; dims=dims))
    R = logsumexp(2logsubexp.(X, logmean); dims=dims)
    n = length(X) ÷ length(R)
    denom = corrected ? n - 1 : n
    return _subtract_log_count(R, denom)
end

"""
$(SIGNATURES)

Compute `log(std(exp, X; corrected=corrected))`.

`X` should be an iterator of numbers.
The result is computed in a numerically stable way.
"""
function logstdexp(X; corrected::Bool=true, logmean=logmeanexp(X))
    return logvarexp(X; corrected=corrected, logmean=logmean) / 2
end

"""
$(SIGNATURES)

Compute `log.(std(exp.(X); dims=dims, corrected=corrected))`.

The result is computed in a numerically stable way.
"""
function logstdexp(X::AbstractArray{<:Number}; dims=:, corrected::Bool=true, logmean=logmeanexp(X; dims=dims))
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
_count_elements(X) = _count_elements(X, Base.IteratorSize(typeof(X)))
_count_elements(X, ::Union{Base.HasLength,Base.HasShape}) = length(X)
_count_elements(X, ::Any) = count(_ -> true, X)

_subtract_log_count(R::Number, n::Integer) = R - log(_convert_count(R, n))
function _subtract_log_count(R::AbstractArray{<:Number}, n::Integer)
    logn = log(convert(eltype(R), n))
    R .-= logn
    return R
end
