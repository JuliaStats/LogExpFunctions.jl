# scalar functions
"""
$(SIGNATURES)

Return `x * log(x)` for `x ≥ 0`, handling ``x = 0`` by taking the downward limit.

```jldoctest
julia> xlogx(0)
0.0
```
"""
function xlogx(x::Number)
    result = x * log(x)
    return iszero(x) ? zero(result) : result
end

"""
$(SIGNATURES)

Return `x * log(y)` for `y > 0` with correct limit at ``x = 0``.

```jldoctest
julia> xlogy(0, 0)
0.0
```
"""
function xlogy(x::Number, y::Number)
    result = x * log(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end

"""
$(SIGNATURES)

Return `x * log(1 + y)` for `y ≥ -1` with correct limit at ``x = 0``.

```jldoctest
julia> xlog1py(0, -1)
0.0
```
"""
function xlog1py(x::Number, y::Number)
    result = x * log1p(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end

"""
$(SIGNATURES)

Return `x * exp(x)` for `x > -Inf`, or zero if `x == -Inf`.

```jldoctest
julia> xexpx(-Inf)
0.0
```
"""
function xexpx(x::Real)
    expx = exp(x)
    return iszero(expx) ? expx : x * expx
end

"""
$(SIGNATURES)

Return `x * exp(y)` for `y > -Inf`, or zero if `y == -Inf`.

```jldoctest
julia> xexpy(1.0, -Inf)
0.0
```
"""
function xexpy(x::Real, y::Real)
    expy = exp(y)
    result = x * expy
    return iszero(expy) && !isnan(x) ? zero(result) : result
end

# The following bounds are precomputed versions of the following abstract
# function, but the implicit interface for AbstractFloat doesn't uniformly
# enforce that all floating point types implement nextfloat and prevfloat.
# @inline function _logistic_bounds(x::AbstractFloat)
#     (
#         logit(nextfloat(zero(float(x)))),
#         logit(prevfloat(one(float(x)))),
#     )
# end

@inline _logistic_bounds(x::Float16) = (Float16(-16.64), Float16(7.625))
@inline _logistic_bounds(x::Float32) = (-103.27893f0, 16.635532f0)
@inline _logistic_bounds(x::Float64) = (-744.4400719213812, 36.7368005696771)

"""
$(SIGNATURES)

The [logistic](https://en.wikipedia.org/wiki/Logistic_function) sigmoid function mapping a
real number to a value in the interval ``[0,1]``,

```math
\\sigma(x) = \\frac{1}{e^{-x} + 1} = \\frac{e^x}{1+e^x}.
```

Its inverse is the [`logit`](@ref) function.
"""
logistic(x::Real) = inv(exp(-x) + one(x))

function logistic(x::Union{Float16, Float32, Float64})
    e = exp(x)
    lower, upper = _logistic_bounds(x)
    return x < lower ? zero(x) : x > upper ? one(x) : e / (one(x) + e)
end

"""
$(SIGNATURES)

The [logit](https://en.wikipedia.org/wiki/Logit) or log-odds transformation,

```math
\\log\\left(\\frac{x}{1-x}\\right), \\text{where} 0 < x < 1
```

Its inverse is the [`logistic`](@ref) function.
"""
logit(x::Real) = log(x / (one(x) - x))

"""
$(SIGNATURES)

Return `log(cosh(x))`, carefully evaluated without intermediate calculation of `cosh(x)`.

The implementation ensures `logcosh(-x) = logcosh(x)`.
"""
function logcosh(x::Real)
    abs_x = abs(x)
    return abs_x + log1pexp(- 2 * abs_x) - IrrationalConstants.logtwo
end

"""
$(SIGNATURES)

Return `log(1+x^2)` evaluated carefully for `abs(x)` very small or very large.
"""
log1psq(x::Real) = log1p(abs2(x))
function log1psq(x::Union{Float32,Float64})
    ax = abs(x)
    ax < maxintfloat(x) ? log1p(abs2(ax)) : 2 * log(ax)
end

"""
$(SIGNATURES)

Return `log(1+exp(x))` evaluated carefully for largish `x`.

This is also called the ["softplus"](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
transformation, being a smooth approximation to `max(0,x)`. Its inverse is [`logexpm1`](@ref).

See:
 * Martin Maechler (2012) [“Accurately Computing log(1 − exp(− |a|))”](http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)

Note: different than Maechler (2012), also uses bounds specific to Float32 and Float16.
"""
function log1pexp(x::Real)
    if x > _log1pexp_thresh(x)
        return float(x)
    else
        t = log1p(exp(-abs(x)))
        return x ≤ 0 ? t : t + x
    end
end

# threshold `thresh` such that `log1pexp(x) == x` for `x > thresh`
_log1pexp_thresh(::Float64) = 33.27106466687738
_log1pexp_thresh(::Float32) = 14.556091f0
_log1pexp_thresh(::Float16) = Float16(6.24)
_log1pexp_thresh(::BigFloat) = 172.5936479594263820448907982430859654507995334557035582760493223638550118704546
_log1pexp_thresh(::Real) = _log1pexp_thresh(0.0)

# _log1p(x::Real) = log1p(x)
# _log1p(x::Real, thresh::Real) = abs(x) ≤ thresh ? x : oftype(x, log1p(x))
# _log1p(x::Float16) = _log1p(x, exp(-7))
# _log1p(x::Float32) = _log1p(x, exp(-15))
# _log1p(x::Float64) = _log1p(x, exp(-37))

# function log1pexp(x::Union{Float16,Float32,Float64})
# 	a, b, c = _log1pexp_branch_bounds(x)
#     if x ≤ a
#         return exp(x)
#     elseif x ≤ b
#         return log1p(exp(x))
#     elseif x ≤ c
#         return x + exp(-x)
#     else
#         return x
#     end
# end

#=
Given the `approx` used in a branch of log1pexp(x) above, we find the first `x` (from above
or below) that is a root of

    T(log1pexp(big(x))) - approx(T(x))

This determines the branch bounds below.
=#
@inline _log1pexp_branch_bounds(::Float64) = (-37.0, 18.0, 33.3)
@inline _log1pexp_branch_bounds(::Float32) = (-15f0, 9f0, 14.5f0)
@inline _log1pexp_branch_bounds(::Float16) = Float16.((-7, 3, 5.7))

"""
$(SIGNATURES)

Return `log(1 - exp(x))`

See:
 * Martin Maechler (2012) [“Accurately Computing log(1 − exp(− |a|))”](http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)

Note: different than Maechler (2012), no negation inside parentheses
"""
log1mexp(x::Real) = x < IrrationalConstants.loghalf ? log1p(-exp(x)) : log(-expm1(x))

"""
$(SIGNATURES)

Return `log(2 - exp(x))` evaluated as `log1p(-expm1(x))`
"""
log2mexp(x::Real) = log1p(-expm1(x))

"""
$(SIGNATURES)

Return `log(exp(x) - 1)` or the “invsoftplus” function.  It is the inverse of
[`log1pexp`](@ref) (aka “softplus”).
"""
logexpm1(x::Real) = x <= 18.0 ? log(expm1(x)) : x <= 33.3 ? x - exp(-x) : oftype(exp(-x), x)
logexpm1(x::Float32) = x <= 9f0 ? log(expm1(x)) : x <= 16f0 ? x - exp(-x) : oftype(exp(-x), x)

const softplus = log1pexp
const invsoftplus = logexpm1

"""
$(SIGNATURES)

Return `log(1 + x) - x`.

Use naive calculation or range reduction outside kernel range.  Accurate ~2ulps for all `x`.
"""
function log1pmx(x::Float64)
    if !(-0.7 < x < 0.9)
        return log1p(x) - x
    elseif x > 0.315
        u = (x-0.5)/1.5
        return _log1pmx_ker(u) - 9.45348918918356180e-2 - 0.5*u
    elseif x > -0.227
        return _log1pmx_ker(x)
    elseif x > -0.4
        u = (x+0.25)/0.75
        return _log1pmx_ker(u) - 3.76820724517809274e-2 + 0.25*u
    elseif x > -0.6
        u = (x+0.5)*2.0
        return _log1pmx_ker(u) - 1.93147180559945309e-1 + 0.5*u
    else
        u = (x+0.625)/0.375
        return _log1pmx_ker(u) - 3.55829253011726237e-1 + 0.625*u
    end
end

"""
$(SIGNATURES)

Return `log(x) - x + 1` carefully evaluated.
"""
function logmxp1(x::Float64)
    if x <= 0.3
        return (log(x) + 1.0) - x
    elseif x <= 0.4
        u = (x-0.375)/0.375
        return _log1pmx_ker(u) - 3.55829253011726237e-1 + 0.625*u
    elseif x <= 0.6
        u = 2.0*(x-0.5)
        return _log1pmx_ker(u) - 1.93147180559945309e-1 + 0.5*u
    else
        return log1pmx(x - 1.0)
    end
end

# The kernel of log1pmx
# Accuracy within ~2ulps for -0.227 < x < 0.315
function _log1pmx_ker(x::Float64)
    r = x/(x+2.0)
    t = r*r
    w = @horner(t,
                6.66666666666666667e-1, # 2/3
                4.00000000000000000e-1, # 2/5
                2.85714285714285714e-1, # 2/7
                2.22222222222222222e-1, # 2/9
                1.81818181818181818e-1, # 2/11
                1.53846153846153846e-1, # 2/13
                1.33333333333333333e-1, # 2/15
                1.17647058823529412e-1) # 2/17
    hxsq = 0.5*x*x
    r*(hxsq+w*t)-hxsq
end


"""
$(SIGNATURES)

Return `log(exp(x) + exp(y))`, avoiding intermediate overflow/undeflow, and handling
non-finite values.
"""
function logaddexp(x::Real, y::Real)
    # ensure Δ = 0 if x = y = ± Inf
    Δ = x == y ? zero(x - y) : abs(x - y)
    max(x, y) + log1pexp(-Δ)
end

Base.@deprecate logsumexp(x::Real, y::Real) logaddexp(x, y)

"""
$(SIGNATURES)

Return `log(abs(exp(x) - exp(y)))`, preserving numerical accuracy.
"""
function logsubexp(x::Real, y::Real)
    # ensure that `Δ = 0` if `x = y = - Inf` (but not for `x = y = +Inf`!)
    Δ = x == y && (isfinite(x) || x < 0) ? zero(x - y) : abs(x - y)
    return max(x, y) + log1mexp(-Δ)
end

"""
    softmax!(r::AbstractArray{<:Real}, x::AbstractArray{<:Real}=r; dims=:)

Overwrite `r` with the
[softmax transformation](https://en.wikipedia.org/wiki/Softmax_function) of `x` over
dimension `dims`.

That is, `r` is overwritten with `exp.(x)`, normalized to sum to 1 over the given
dimensions.

See also: [`softmax`](@ref)
"""
softmax!(r::AbstractArray{<:Real}, x::AbstractArray{<:Real}=r; dims=:) =
    _softmax!(r, x, dims)

"""
    softmax(x::AbstractArray{<:Real}; dims=:)

Return the
[softmax transformation](https://en.wikipedia.org/wiki/Softmax_function) of `x` over
dimension `dims`.

That is, return `exp.(x)`, normalized to sum to 1 over the given dimensions.

See also: [`softmax!`](@ref)
"""
softmax(x::AbstractArray{<:Real}; dims=:) =
    softmax!(similar(x, float(eltype(x))), x; dims=dims)

function _softmax!(r, x, ::Colon)
    length(r) == length(x) || throw(DimensionMismatch("inconsistent array lengths"))
    u = maximum(x)
    map!(r, x) do xi
        return exp(xi - u)
    end
    LinearAlgebra.lmul!(inv(sum(r)), r)
    return r
end

function _softmax!(r, x, dims)
    size(r) == size(x) || throw(DimensionMismatch("inconsistent array sizes"))
    u = maximum(x; dims=dims)
    r .= exp.(x .- u)
    if u isa Array{eltype(r)}
        # array can be reused
        sum!(u, r)
        r ./= u
    else
        r ./= sum(r; dims=dims)
    end
    return r
end
