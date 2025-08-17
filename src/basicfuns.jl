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

Return `x * exp(y)` for `y > -Inf`, or zero if `y == -Inf` or if `x == 0` and `y` is finite.

```jldoctest
julia> xexpy(1.0, -Inf)
0.0
```
"""
function xexpy(x::Real, y::Real)
    expy = exp(y)
    result = x * expy
    return (iszero(x) && isfinite(y)) || (iszero(expy) && !isnan(x)) ? zero(result) : result
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

The [logit](https://en.wikipedia.org/wiki/Logit) or log-odds transformation, defined as
```math
\\operatorname{logit}(x) = \\log\\left(\\frac{x}{1-x}\\right)
```
for ``0 < x < 1``.

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
    if (x isa Union{Float16, Float32, Float64}) && (abs_x < oftype(x, 0.7373046875))
        return logcosh_ker(x)
    end
    return abs_x + log1pexp(- 2 * abs_x) - IrrationalConstants.logtwo
end

"""
    logcosh_ker(x::Union{Float32, Float64})

The kernel of `logcosh`.

The polynomial coefficients were found using Sollya:

```sollya
prec = 500!;
points = 50001!;
accurate = log(cosh(x));
domain = [-0.125, 0.7373046875];
constrained_part = (x^2) / 2;
free_monomials_16 = [|4, 6|];
free_monomials_32 = [|4, 6, 8, 10, 12|];
free_monomials_64 = [|4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24|];
polynomial_16 = fpminimax(accurate, free_monomials_16, [|halfprecision...|], domain, constrained_part);
polynomial_32 = fpminimax(accurate, free_monomials_32, [|single...|], domain, constrained_part);
polynomial_64 = fpminimax(accurate, free_monomials_64, [|double...|], domain, constrained_part);
polynomial_16;
polynomial_32;
polynomial_64;
```
"""
function logcosh_ker(x::Union{Float16, Float32, Float64})
    x² = x * x
    if x isa Float16
        p = (
            Float16(5f-1),
            Float16(-0.08264),
            Float16(0.01793),
        )
    elseif x isa Float32
        p = (
            5f-1,
            -0.083333164f0,
            0.022217678f0,
            -0.0067060017f0,
            0.0020296266f0,
            -0.00044135848f0,
        )
    elseif x isa Float64
        p = (
            5e-1,
            -0.08333333333332801,
            0.02222222222164912,
            -0.0067460317245250445,
            0.0021869484500251714,
            -0.0007385985435311435,
            0.0002565500026777061,
            -9.084985367586575e-5,
            3.2348259905568986e-5,
            -1.1058814347469105e-5,
            3.16293199955507e-6,
            -5.312230207322749e-7,
        )
    end
    evalpoly(x², p) * x²
end

"""
$(SIGNATURES)

Return `log(abs(sinh(x)))`, carefully evaluated without intermediate calculation of `sinh(x)`.

The implementation ensures `logabssinh(-x) = logabssinh(x)`.
"""
function logabssinh(x::Real)
    abs_x = abs(x)
    return abs_x + log1mexp(- 2 * abs_x) - IrrationalConstants.logtwo
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

This is also called the ["softplus"](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
transformation (in its default parametrization, see [`softplus`](@ref)), being a smooth approximation to `max(0,x)`. 

See:
 * Martin Maechler (2012) [“Accurately Computing log(1 − exp(− |a|))”](http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
"""
log1pexp(x::Real) = _log1pexp(float(x)) # ensures that BigInt/BigFloat, Int/Float64 etc. dispatch to the same algorithm

# Approximations based on Maechler (2012)
# Argument `x` is a floating point number due to the definition of `log1pexp` above
function _log1pexp(x::Real)
    x1, x2, x3, x4 = _log1pexp_thresholds(x)
    if x < x1
        return zero(x)
    elseif x < x2
        return exp(x)
    elseif x < x3
        return log1p(exp(x))
    elseif x < x4
        return x + exp(-x)
    else
        return x
    end
end

#= The precision of BigFloat cannot be computed from the type only and computing
thresholds is slow. Therefore prefer version without thresholds in this case. =#
_log1pexp(x::BigFloat) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

#=
Returns thresholds x1, x2, x3, x4 such that:
    * log1pexp(x) = 0 for x < x1
    * log1pexp(x) ≈ exp(x) for x < x2
    * log1pexp(x) ≈ log1p(exp(x)) for x2 ≤ x < x3
    * log1pexp(x) ≈ x + exp(-x) for x3 ≤ x < x4
    * log1pexp(x) ≈ x for x ≥ x4

where the tolerances of the approximations are on the order of eps(typeof(x)).
For types for which `precision(x)` depends only on the type of `x`, the compiler
should optimize away all computations done here.
=#
@inline function _log1pexp_thresholds(x::Real)
    prec = precision(x)
    logtwo = oftype(x, IrrationalConstants.logtwo)
    x1 = (exponent(nextfloat(zero(x))) - 1) * logtwo
    x2 = -prec * logtwo
    x3 = (prec - 1) * logtwo / 2
    x4 = -x2 - log(-x2) * (1 + 1 / x2) # approximate root of e^-x == x * ϵ/2 via asymptotics of Lambert's W function
    return (x1, x2, x3, x4)
end

#=
For common types we hard-code the thresholds to make absolutely sure they are not recomputed
each time. Also, _log1pexp_thresholds is not elided by the compiler in Julia 1.0 / 1.6.
=#
@inline _log1pexp_thresholds(::Float64) = (-745.1332191019412, -36.7368005696771, 18.021826694558577, 33.23111882352963)
@inline _log1pexp_thresholds(::Float32) = (-103.97208f0, -16.635532f0, 7.9711924f0, 13.993f0)
@inline _log1pexp_thresholds(::Float16) = (Float16(-17.33), Float16(-7.625), Float16(3.467), Float16(5.86))

"""
$(SIGNATURES)

Return `log(1 - exp(x))`

See:
 * Martin Maechler (2012) [“Accurately Computing log(1 − exp(− |a|))”](http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)

Note: different than Maechler (2012), no negation inside parentheses
"""
function log1mexp(x::Real)
    # Use explicit `oftype(..)` instead of just `loghalf` to avoid CUDA issues:
    # https://github.com/JuliaStats/LogExpFunctions.jl/issues/73
    if x < oftype(float(x), IrrationalConstants.loghalf)
        return log1p(-exp(x))
    else
        return log(-expm1(x))
    end
end

"""
$(SIGNATURES)

Return `log(2 - exp(x))` evaluated carefully.
"""
log2mexp(x::Real) = log1p(-expm1(x))

"""
$(SIGNATURES)

Return `log(exp(x) - 1)` or the “invsoftplus” function.  It is the inverse of
[`log1pexp`](@ref) (aka “softplus”).
"""
logexpm1(x::Real) = x <= 18.0 ? log(expm1(x)) : x <= 33.3 ? x - exp(-x) : oftype(exp(-x), x)
logexpm1(x::Float32) = x <= 9f0 ? log(expm1(x)) : x <= 16f0 ? x - exp(-x) : oftype(exp(-x), x)

"""
$(SIGNATURES)

The generalized `softplus` function (Wiemann et al., 2024) takes an additional optional parameter `a` that control 
the approximation error with respect to the linear spline. It defaults to `a=1.0`, in which case the softplus is 
equivalent to [`log1pexp`](@ref).

See:
 * Wiemann, P. F., Kneib, T., & Hambuckers, J. (2024). Using the softplus function to construct alternative link functions in generalized linear models and beyond. Statistical Papers, 65(5), 3155-3180.
"""
softplus(x::Real) = log1pexp(x)
softplus(x::Real, a::Real) = log1pexp(a * x) / a

"""
$(SIGNATURES)

The inverse generalized `softplus` function (Wiemann et al., 2024). See [`softplus`](@ref).
"""
invsoftplus(y::Real) = logexpm1(y)
invsoftplus(y::Real, a::Real) = logexpm1(a * y) / a


"""
$(SIGNATURES)

Return `log(1 + x) - x`.

Use naive calculation or range reduction outside kernel range.  Accurate ~2ulps for all `x`.
This will fall back to the naive calculation for argument types different from `Float32, Float64`.
"""
log1pmx(x::Real) = log1p(x) - x # Naive fallback

function log1pmx(x::T) where T <: Union{Float32, Float64}
    if !(T(-0.425) < x < T(0.4)) # accurate within 2 ULPs when log2(abs(log1p(x))) > 1.5
        return log1p(x) - x
    else
        return _log1pmx_ker(x)
    end
end

"""
$(SIGNATURES)

Return `log(x) - x + 1` carefully evaluated.
This will fall back to the naive calculation for argument types different from `Float64`.
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

# Naive fallback
function logmxp1(x::Real)
    one_x = one(x)
    if 2 * x < one_x
        # for small values of `x` the other branch returns non-finite values
        return (log(x) + one_x) - x
    else
        return log1pmx(x - one_x)
    end
end

# The kernel of log1pmx
# Accuracy within ~2ulps -0.227 < x < 0.315 for Float64
# Accuracy <2.18ulps -0.425 < x < 0.425 for Float32
# parameters foudn via Remez.jl, specifically:
# g(x) = evalpoly(x, big(2)./ntuple(i->2i+1, 50))
# p = T.(Tuple(ratfn_minimax(g, (1e-3, (.425/(.425+2))^2), 8, 0)[1]))
function _log1pmx_ker(x::T) where T <: Union{Float32, Float64}
    r = x / (x+2)
    t = r*r
    if T == Float32
        p = (0.6666658f0, 0.40008822f0, 0.2827692f0, 0.26246136f0)
    else
        p = (0.6666666666666669,
             0.3999999999997768,
             0.2857142857784595,
             0.2222222142048249,
             0.18181870670924566,
             0.15382646727504887,
             0.1337701340211177,
             0.11201972567415432,
             0.143418239946679)
    end
    w = evalpoly(t, p)
    hxsq = x*x/2
    muladd(r, muladd(w, t, hxsq), -hxsq)
end


"""
$(SIGNATURES)

Return `log(exp(x) + exp(y))`, avoiding intermediate overflow/undeflow, and handling
non-finite values.
"""
function logaddexp(x::Real, y::Real)
    # Compute max = Base.max(x, y) and diff = x == y ? zero(x - y) : -abs(x - y)
    # in a faster type-stable way
    a, b = promote(x, y)
    if a < b
        diff = a - b
        max = b
    else
        # ensure diff = 0 if a = b = ± Inf
        diff = a == b ? zero(a - b) : b - a
        max = !isnan(b) ? a : b
    end
    return max + log1pexp(diff)
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

"""
$(SIGNATURES)

Compute the complementary log-log, `log(-log(1 - x))`.
"""
cloglog(x) = log(-log1p(-x))

"""
$(SIGNATURES)

Compute the complementary double exponential, `1 - exp(-exp(x))`.
"""
cexpexp(x) = -expm1(-exp(x))

#=
this uses the identity:

log(logistic(x)) = -log(1 + exp(-x))
=#
"""
$(SIGNATURES)

Return `log(logistic(x))`, computed more carefully and with fewer calls
than the naive composition of functions.

Its inverse is the [`logitexp`](@ref) function.
"""
loglogistic(x::Real) = -log1pexp(-float(x))

#=
this uses the identity:

logit(exp(x)) = log(exp(x) / (1 + exp(x))) = -log(exp(-x) - 1)
=#
"""
$(SIGNATURES)

Return `logit(exp(x))`, computed more carefully and with fewer calls than
the naive composition of functions.

Its inverse is the [`loglogistic`](@ref) function.
"""
logitexp(x::Real) = -logexpm1(-float(x))

#=
this uses the identity:

log(logistic(-x)) = -log(1 + exp(x))

that is, negation in the log-odds domain.
=#

"""
$(SIGNATURES)

Return `log(1 - logistic(x))`, computed more carefully and with fewer calls than
the naive composition of functions.

Its inverse is the [`logit1mexp`](@ref) function.
"""
log1mlogistic(x::Real) = -log1pexp(x)

#=

this uses the same identity:

-logit(exp(x)) = logit(1 - exp(x)) = log((1 - exp(x)) / exp(x)) = log(exp(-x) - 1)
=#

"""
$(SIGNATURES)

Return `logit(1 - exp(x))`, computed more carefully and with fewer calls than
the naive composition of functions.

Its inverse is the [`log1mlogistic`](@ref) function.
"""
logit1mexp(x::Real) = logexpm1(-float(x))
