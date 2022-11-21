"""
$(SIGNATURES)

Compute `log(sum(exp, X))`.

`X` should be an iterator of real or complex numbers.
The result is computed in a numerically stable way that avoids intermediate over- and underflow, using a single pass over the data.

See also [`logsumexp!`](@ref).

# References

[Sebastian Nowozin: Streaming Log-sum-exp Computation](http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
"""
logsumexp(X) = _logsumexp_onepass(X)

"""
$(SIGNATURES)

Compute `log.(sum(exp.(X); dims=dims))`.

The result is computed in a numerically stable way that avoids intermediate over- and underflow, using a single pass over the data.

See also [`logsumexp!`](@ref).

# References

[Sebastian Nowozin: Streaming Log-sum-exp Computation](http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
"""
logsumexp(X::AbstractArray{<:Number}; dims=:) = _logsumexp(X, dims)

"""
$(SIGNATURES)

Compute [`logsumexp`](@ref) of `X` over the singleton dimensions of `out`, and write results to `out`.

The result is computed in a numerically stable way that avoids intermediate over- and underflow, using a single pass over the data.

See also [`logsumexp`](@ref).

# References

[Sebastian Nowozin: Streaming Log-sum-exp Computation](http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
"""
function logsumexp!(out::AbstractArray, X::AbstractArray{<:Number})
    FT = float(eltype(X))
    xmax_r = fill!(similar(out, Tuple{FT,FT}), (FT(-Inf), zero(FT)))
    Base.reducedim!(_logsumexp_onepass_op, xmax_r, X)
    return @. out = first(xmax_r) + log1p(last(xmax_r))
end

_logsumexp(X::AbstractArray{<:Number}, ::Colon) = _logsumexp_onepass(X)
function _logsumexp(X::AbstractArray{<:Number}, dims)
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
function _logsumexp_onepass_op(x1::T, x2::T) where {T<:Number}
    xmax, a = if isnan(x1) || isnan(x2)
        # ensure that `NaN` is propagated correctly for complex numbers
        z = oftype(x1, NaN)
        z, exp(z)
    else
        real_x1 = real(x1)
        real_x2 = real(x2)
        if real_x1 > real_x2
            x1, x2 - x1
        elseif real_x1 < real_x2
            x2, x1 - x2
        else
            # handle `x1 = x2 = ±Inf` correctly
            # checking inequalities above instead of equality fixes issue #59
            x2, zero(x1 - x2)
        end
    end
    r = exp(a)
    return xmax, r
end
_logsumexp_onepass_op(x1::Number, x2::Number) = _logsumexp_onepass_op(promote(x1, x2)...)

# reduce a number and a partial sum
_logsumexp_onepass_op(x::Number, (xmax, r)::Tuple{<:Number,<:Number}) =
    _logsumexp_onepass_op(x, xmax, r)
_logsumexp_onepass_op((xmax, r)::Tuple{<:Number,<:Number}, x::Number) =
    _logsumexp_onepass_op(x, xmax, r)
_logsumexp_onepass_op(x::Number, xmax::Number, r::Number) =
    _logsumexp_onepass_op(promote(x, xmax)..., r)
function _logsumexp_onepass_op(x::T, xmax::T, r::Number) where {T<:Number}
    _xmax, _r = if isnan(x) || isnan(xmax)
        # ensure that `NaN` is propagated correctly for complex numbers
        z = oftype(x, NaN)
        z, r + exp(z)
    else
        real_x = real(x)
        real_xmax = real(xmax)
        if real_x > real_xmax
            x, (r + one(r)) * exp(xmax - x)
        elseif real_x < real_xmax
            xmax, r + exp(x - xmax)
        else
            # handle `x = xmax = ±Inf` correctly
            # checking inequalities above instead of equality fixes issue #59
            xmax, r + exp(zero(x - xmax))
        end
    end
    return _xmax, _r
end

# reduce two partial sums
function _logsumexp_onepass_op(
    (xmax1, r1)::Tuple{<:Number,<:Number}, (xmax2, r2)::Tuple{<:Number,<:Number}
)
    return _logsumexp_onepass_op(xmax1, xmax2, r1, r2)
end
function _logsumexp_onepass_op(xmax1::Number, xmax2::Number, r1::Number, r2::Number)
    return _logsumexp_onepass_op(promote(xmax1, xmax2)..., promote(r1, r2)...)
end
function _logsumexp_onepass_op(xmax1::T, xmax2::T, r1::R, r2::R) where {T<:Number,R<:Number}
    xmax, r = if isnan(xmax1) || isnan(xmax2)
        # ensure that `NaN` is propagated correctly for complex numbers
        z = oftype(xmax1, NaN)
        z, r1 + exp(z)
    else
        real_xmax1 = real(xmax1)
        real_xmax2 = real(xmax2)
        if real_xmax1 > real_xmax2
            xmax1, r1 + (r2 + one(r2)) * exp(xmax2 - xmax1)
        elseif real_xmax1 < real_xmax2
            xmax2, r2 + (r1 + one(r1)) * exp(xmax1 - xmax2)
        else
            # handle `xmax1 = xmax2 = ±Inf` correctly
            # checking inequalities above instead of equality fixes issue #59
            xmax2, r2 + (r1 + one(r1)) * exp(zero(xmax1 - xmax2))
        end
    end
    return xmax, r
end
