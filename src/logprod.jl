"""
    logprod(X::AbstractArray{T}; dims)

Compute `log(prod(x))` efficiently.

This is faster than computing `sum(log, X)`, especially for large `X`.
It works by representing the `j`th element of `x` as ``x_j = a_j  2^{b_j}`,
allowing us to write
```math
\\log \\prod_k x_j = \\log(\\prod_j a_j) + \\log{2} \\sum_j b_j.
```
"""
logprod(x) = first(logabsprod(x))

export logabsprod

function logabsprod(x::AbstractArray{T}) where {T}
    sig, ex = mapreduce(_logabsprod_op, x; init=frexp(one(T))) do xj
        float_xj = float(xj)
        frexp(float_xj)
    end
    sgn = signbit(sig) ? -one(T) : one(T)
    return (log(abs(sig)) + IrrationalConstants.logtwo * T(ex), sgn)
end

@inline function _logabsprod_op((sig1, ex1), (sig2, ex2))
    sig = sig1 * sig2
    ex = ex1 + ex2
    
    # The significand from `frexp` has magnitude in the range [0.5, 1), 
    # so multiplication will eventually underflow
    may_underflow(sig::T) where {T} = sig < sqrt(floatmin(T))
    if may_underflow(sig)
        (sig, Δex) = frexp(sig)
        ex += Δex
    end
    return sig, ex
end

"""
    logprod(x)
    logprod(f, x, ys...)

For any iterator which produces `AbstractFloat` elements,
this can use `logprod`'s fast reduction strategy.

Signature with `f` is equivalent to `sum(log, map(f, x, ys...))`
or `mapreduce(log∘f, +, x, ys...)`, without intermediate allocations.

Does not accept a `dims` keyword.
"""
logprod(f, x) = logprod(Iterators.map(f, x))
logprod(f, x, ys...) = logprod(f(xy...) for xy in zip(x, ys...))

# Iterator version, uses the same `_logprod_op`, should be the same speed.
function logprod(x)
    iter = iterate(x)
    if isnothing(iter)
        T = Base._return_type(first, Tuple{typeof(x)})
        return T <: Number ? zero(float(T)) : 0.0
    end
    x1 = float(iter[1])
    x1 isa AbstractFloat || return sum(log, x)
    x1 < 0 && Base.Math.throw_complex_domainerror(:log, x1)
    sig, ex = significand(x1), _exponent(x1)
    nonfloat = zero(x1)
    iter = iterate(x, iter[2])
    while iter !== nothing
        xj = float(iter[1])
        if xj isa AbstractFloat
            sig, ex = _logprod_op((sig, ex), (significand(xj), _exponent(xj)))
        else
            nonfloat += log(xj)
        end
        iter = iterate(x, iter[2])
    end
    return log(sig) + IrrationalConstants.logtwo * oftype(sig, ex) + nonfloat
end
