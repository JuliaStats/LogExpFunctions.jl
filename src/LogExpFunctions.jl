module LogExpFunctions

using DocStringExtensions: SIGNATURES
using Base: Math.@horner

import IrrationalConstants
import LinearAlgebra

export xlogx, xlogy, xlog1py, xexpx, xexpy, logistic, logit, log1psq, log1pexp, log1mexp, log2mexp, logexpm1,
    softplus, invsoftplus, log1pmx, logmxp1, logaddexp, logsubexp, logsumexp, logsumexp!, softmax,
    softmax!, logcosh, logabssinh, cloglog, cexpexp

# expm1(::Float16) is not defined in older Julia versions,
# hence for better Float16 support we use an internal function instead
# https://github.com/JuliaLang/julia/pull/40867
if VERSION < v"1.7.0-DEV.1172"
    _expm1(x) = expm1(x)
    _expm1(x::Float16) = Float16(expm1(Float32(x)))
else
    const _expm1 = expm1
end

include("basicfuns.jl")
include("logsumexp.jl")

if !isdefined(Base, :get_extension)
    include("../ext/LogExpFunctionsChainRulesCoreExt.jl")
    include("../ext/LogExpFunctionsChangesOfVariablesExt.jl")
    include("../ext/LogExpFunctionsInverseFunctionsExt.jl")
end

end # module
