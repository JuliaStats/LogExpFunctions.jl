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
    lse, n = _logsumexp_count(X)
    return lse - log(oftype(lse, n))
end
logmeanexp(X::AbstractArray{<:Number}; dims=:) = _logmeanexp(X, dims)
function _logmeanexp(X, ::Colon)
    out = logsumexp(X)
    return out - oftype(out, log(length(X)))
end
function _logmeanexp(X, dims)
    out = similar(X, float(eltype(X)), Base.reduced_indices(axes(X), dims))
    return logmeanexp!(out, X)
end

"""
$(SIGNATURES)

Compute [`logmeanexp`](@ref) of `X` over the singleton dimensions of `out`, and write the
result to `out`.
"""
function logmeanexp!(out::AbstractArray, X::AbstractArray{<:Number})
    logsumexp!(out, X)
    return out .-= log(length(X) / length(out))
end


"""
$(SIGNATURES)

Compute `log(var(exp.(X); corrected))` in a numerically stable way.

`X` should be an array of real numbers; `dims` selects the dimensions to reduce over,
returning `log.(var(exp.(X); dims, corrected))`. A precomputed `logmeanexp(X; dims)` can be
passed as `logmean` to avoid recomputing it.

See also [`logvarexp!`](@ref).
"""
logvarexp(X::AbstractArray{<:Real}; dims=:, corrected::Bool=true, logmean=logmeanexp(X; dims)) = _logvarexp(X, dims, corrected, logmean)
function _logvarexp(X::AbstractArray{<:Real}, dims::Colon, corrected::Bool, logmean)
    out = logsumexp(2 .* logsubexp.(X, logmean))
    return out - oftype(out, log(max(0, length(X) - corrected)))
end
function _logvarexp(X::AbstractArray{<:Real}, dims, corrected::Bool, logmean)
    out = logsumexp(2 .* logsubexp.(X, logmean); dims)
    return out .-= log(max(0, length(X) / length(out) - corrected))
end


"""
$(SIGNATURES)

Compute [`logvarexp`](@ref) of `X` over the singleton dimensions of `out`, and write the
result to `out`.
"""
function logvarexp!(out::AbstractArray{<:Real}, X::AbstractArray{<:Real}; corrected::Bool=true)
    logmeanexp!(out, X)
    logsumexp!(out, 2 .* logsubexp.(X, out))
    return out .-= log(max(0, length(X) / length(out) - corrected))
end


"""
$(SIGNATURES)

Compute `log(std(exp.(X); dims, corrected))` in a numerically stable way.

Keyword arguments are as in [`logvarexp`](@ref).
"""
logstdexp(X::AbstractArray{<:Real}; dims=:, corrected::Bool=true, logmean=logmeanexp(X; dims)) = logvarexp(X; dims, corrected, logmean) / 2


"""
$(SIGNATURES)

Compute [`logstdexp`](@ref) of `X` over the singleton dimensions of `out`, and write the
result to `out`.
"""
function logstdexp!(out::AbstractArray{<:Real}, X::AbstractArray{<:Real}; corrected::Bool=true)
    logvarexp!(out, X; corrected)
    return out ./= 2
end


# internal function to do logsumexp over an iterator in one pass, returning also its length
function _logsumexp_count(X)
    next = iterate(X)
    isnothing(next) && throw(ArgumentError("reducing over an empty collection is not allowed"))
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
