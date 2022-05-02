using IrrationalConstants: logtwo

"""
$(SIGNATURES)

Compute `sum(log.(X))` with a single `log` evaluation.

This is faster than computing `sum(log.(X))` or even `sum(log, X)`, in particular as `X` increases.

This works by representing the `j`th element of `X` as ``x_j = a_j  2^b_j``,
allowing us to write
```math
\\sum_j \\log{x_j} = \\log(\\prod_j a_j) + \\log{2} \\sum_j b_j
```
Since ``\\log{2}`` is constant, `sumlog` only requires a single `log` evaluation.
"""
function sumlog(x::AbstractArray{T}) where {T<:AbstractFloat} 
    sig = one(T) 
    ex = zero(exponent(one(T)))
    bound = floatmax(T) / 2 
    for xj in x 
        sig *= significand(xj)
        ex += exponent(xj) 

        # Significands are in the range [1,2), so multiplication will eventually overflow
        if sig > bound
            (a, b) = (significand(sig), exponent(sig))
            sig = a
            ex += b
        end
    end
    log(sig) + logtwo * ex
end

sumlog(x) = sum(log, x)