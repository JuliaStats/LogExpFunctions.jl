"""
$(SIGNATURES)

Compute `sum(log.(X))` with a single `log` evaluation.

This is faster than computing `sum(log.(X))` or even `sum(log, X)`, in
particular as the size of `X` increases.

This works by representing the `j`th element of `X` as ``x_j = a_j  2^{b_j}``,
allowing us to write
```math
\\sum_j \\log{x_j} = \\log(\\prod_j a_j) + \\log{2} \\sum_j b_j
```
Since ``\\log{2}`` is constant, `sumlog` only requires a single `log`
evaluation.
"""
function sumlog(x)
    T = float(eltype(x))
    _sumlog(T, values(x))
end

@inline function _sumlog(::Type{T}, x) where {T<:AbstractFloat}
    sig, ex = mapreduce(_sumlog_op, x; init=(one(T), zero(exponent(one(T))))) do xj
        float_xj = float(xj)
        significand(float_xj), exponent(float_xj) 
    end
    return log(sig) + IrrationalConstants.logtwo * ex
end

@inline function _sumlog_op((sig1, ex1), (sig2, ex2))
    sig = sig1 * sig2
    ex = ex1 + ex2
    # Significands are in the range [1,2), so multiplication will eventually overflow
    if sig > floatmax(typeof(sig)) / 2
        ex += exponent(sig)
        sig = significand(sig)
    end
    return sig, ex
end

# `float(T)` is not always `isa AbstractFloat`, e.g. dual numbers or symbolics
@inline _sumlog(::Type{T}, x) where {T} = sum(log, x)
