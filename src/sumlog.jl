"""
    sumlog(X::AbstractArray{T}; dims)

Compute `sum(log.(X))` with a single `log` evaluation,
provided `float(T) <: AbstractFloat`.

This is faster than computing `sum(log, X)`, especially for large `X`.
It works by representing the `j`th element of `X` as ``x_j = a_j  2^{b_j}``,
allowing us to write
```math
\\sum_j \\log{x_j} = \\log(\\prod_j a_j) + \\log{2} \\sum_j b_j
```
"""
sumlog(x::AbstractArray{T}; dims=:) where T = _sumlog(float(T), dims, x)

function _sumlog(::Type{T}, ::Colon, x) where {T<:AbstractFloat}
    sig, ex = mapreduce(_sumlog_op, x; init=(one(T), 0)) do xj
        xj < 0 && Base.Math.throw_complex_domainerror(:log, xj)
        float_xj = float(xj)
        significand(float_xj), _exponent(float_xj) 
    end
    return log(sig) + IrrationalConstants.logtwo * T(ex)
end

function _sumlog(::Type{T}, dims, x) where {T<:AbstractFloat}
    sig_ex = mapreduce(_sumlog_op, x; dims=dims, init=(one(T), 0)) do xj
        xj < 0 && Base.Math.throw_complex_domainerror(:log, xj)
        float_xj = float(xj)
        significand(float_xj), _exponent(float_xj) 
    end
    map(sig_ex) do (sig, ex)
        log(sig) + IrrationalConstants.logtwo * T(ex)
    end
end

# Fallback: `float(T)` is not always `<: AbstractFloat`, e.g. complex, dual numbers or symbolics
_sumlog(::Type, dims, x) = sum(log, x; dims)

@inline function _sumlog_op((sig1, ex1), (sig2, ex2))
    sig = sig1 * sig2
    # sig = ifelse(sig2<0, sig2, sig1 * sig2)
    ex = ex1 + ex2
    # Significands are in the range [1,2), so multiplication will eventually overflow
    if sig > floatmax(typeof(sig)) / 2
        ex += _exponent(sig)
        sig = significand(sig)
    end
    return sig, ex
end

# The exported `exponent(x)` checks for `NaN` etc, this function doesn't, which is fine as `sig` keeps track.
_exponent(x::Base.IEEEFloat) = Base.Math._exponent_finite_nonzero(x)
Base.@assume_effects :nothrow _exponent(x::AbstractFloat) = Int(exponent(x))  # e.g. for BigFloat

"""
    sumlog(x)
    sumlog(f, x, ys...)

For any iterator which produces `AbstractFloat` elements,
this can use `sumlog`'s fast reduction strategy.

Signature with `f` is equivalent to `sum(log, map(f, x, ys...))`
or `mapreduce(log∘f, +, x, ys...)`, without intermediate allocations.

Does not accept a `dims` keyword.
"""
sumlog(f, x) = sumlog(Iterators.map(f, x))
sumlog(f, x, ys...) = sumlog(f(xy...) for xy in zip(x, ys...))

# Iterator version, uses the same `_sumlog_op`, should be the same speed.
function sumlog(x)
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
            xj < 0 && Base.Math.throw_complex_domainerror(:log, xj)
            sig, ex = _sumlog_op((sig, ex), (significand(xj), _exponent(xj)))
        else
            nonfloat += log(xj)
        end
        iter = iterate(x, iter[2])
    end
    return log(sig) + IrrationalConstants.logtwo * oftype(sig, ex) + nonfloat
end
