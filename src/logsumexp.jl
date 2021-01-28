"""
$(SIGNATURES)

Compute `log(sum(exp, X))` in a numerically stable way that avoids intermediate over- and
underflow.

`X` should be an iterator of real numbers. The result is computed using a single pass over
the data.

# References

[Sebastian Nowozin: Streaming Log-sum-exp Computation](http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
"""
logsumexp(X) = _logsumexp_onepass(X)

"""
$(SIGNATURES)

Compute `log.(sum(exp.(X); dims=dims))` in a numerically stable way that avoids
intermediate over- and underflow.

The result is computed using a single pass over the data.

# References

[Sebastian Nowozin: Streaming Log-sum-exp Computation](http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
"""
logsumexp(X::AbstractArray{<:Real}; dims=:) = _logsumexp(X, dims)

_logsumexp(X::AbstractArray{<:Real}, ::Colon) = _logsumexp_onepass(X)
function _logsumexp(X::AbstractArray{<:Real}, dims)
    # Do not use log(zero(eltype(X))) directly to avoid issues with ForwardDiff (#82)
    FT = float(eltype(X))
    xmax_r = reduce(_logsumexp_onepass_op, X; dims=dims, init=(FT(-Inf), zero(FT)))
    return @. first(xmax_r) + log1p(last(xmax_r))
end

function _logsumexp_onepass(X)
    # fallback for empty collections
    isempty(X) && return log(sum(X))
    return _logsumexp_onepass_result(_logsumexp_onepass_reduce(X, Base.IteratorEltype(X)))
end

# function barrier for reductions with single element and without initial element
_logsumexp_onepass_result(x) = float(x)
_logsumexp_onepass_result((xmax, r)::Tuple) = xmax + log1p(r)

# iterables with known element type
function _logsumexp_onepass_reduce(X, ::Base.HasEltype)
    # do not perform type computations if element type is abstract
    T = eltype(X)
    isconcretetype(T) || return _logsumexp_onepass_reduce(X, Base.EltypeUnknown())

    FT = float(T)
    return reduce(_logsumexp_onepass_op, X; init=(FT(-Inf), zero(FT)))
end

# iterables without known element type
_logsumexp_onepass_reduce(X, ::Base.EltypeUnknown) = reduce(_logsumexp_onepass_op, X)

## Reductions for one-pass algorithm: avoid expensive multiplications if numbers are reduced

# reduce two numbers
function _logsumexp_onepass_op(x1, x2)
    a = x1 == x2 ? zero(x1 - x2) : -abs(x1 - x2)
    xmax = x1 > x2 ? oftype(a, x1) : oftype(a, x2)
    r = exp(a)
    return xmax, r
end

# reduce a number and a partial sum
function _logsumexp_onepass_op(x, (xmax, r)::Tuple)
    a = x == xmax ? zero(x - xmax) : -abs(x - xmax)
    if x > xmax
        _xmax = oftype(a, x)
        _r = (r + one(r)) * exp(a)
    else
        _xmax = oftype(a, xmax)
        _r = r + exp(a)
    end
    return _xmax, _r
end
_logsumexp_onepass_op(xmax_r::Tuple, x) = _logsumexp_onepass_op(x, xmax_r)

# reduce two partial sums
function _logsumexp_onepass_op((xmax1, r1)::Tuple, (xmax2, r2)::Tuple)
    a = xmax1 == xmax2 ? zero(xmax1 - xmax2) : -abs(xmax1 - xmax2)
    if xmax1 > xmax2
        xmax = oftype(a, xmax1)
        r = r1 + (r2 + one(r2)) * exp(a)
    else
        xmax = oftype(a, xmax2)
        r = r2 + (r1 + one(r1)) * exp(a)
    end
    return xmax, r
end
