using IrrationalConstants: logtwo

"""
$(SIGNATURES)

Compute `sum(log.(X))`.

`sum(log.(X))` can be evaluated much more quickly as `sum(log, X)`. However,
this still requires computing `log` for each element of `X`. 

`sumlog(X)` can be faster still, especially an the length of `X` increases.

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