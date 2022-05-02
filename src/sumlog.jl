using IrrationalConstants: logtwo

"""
$(SIGNATURES)

Compute `sum(log.(X))` with a single `log` evaluation.

This is faster than computing `sum(log.(X))` or even `sum(log, X)`, in particular as `X` increases.

This works by representing the `j`th element of `X` as `xⱼ = aⱼ * 2 ^ bⱼ`,
allowing us to write

    ∑ⱼ log(xⱼ) = log(∏ⱼ aⱼ) + log(2) * ∑ⱼ bⱼ

Since `log(2)` is constant, `sumlog` only requires a single `log` evaluation.
"""
function sumlog(x::AbstractArray{T}) where {T} 
    sig = one(T) 
    ex = zero(exponent(one(T)))
    bound = floatmax(T) / 2 
    for xj in x 
        sig *= significand(xj)
        ex += exponent(xj) 

        # Significands are in the rang [1,2), so multiplication will eventually overflow
        if sig > bound
            (a, b) = (significand(sig), exponent(sig))
            sig = a
            ex += b
        end
    end
    log(sig) + logtwo * ex
end