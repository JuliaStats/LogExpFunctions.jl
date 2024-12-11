module LogExpFunctions

using DocStringExtensions: SIGNATURES
using Base: Math.@horner

import IrrationalConstants
import LinearAlgebra

export xlogx, xlogy, xlog1py, xexpx, xexpy, logistic, logit, log1psq, log1pexp, log1mexp, log2mexp, logexpm1,
    softplus, invsoftplus, log1pmx, logmxp1, logaddexp, logsubexp, logsumexp, logsumexp!, softmax,
    softmax!, logcosh, logabssinh, cloglog, cexpexp,
    loglogistic, logitexp, log1mlogistic, logit1mexp

include("basicfuns.jl")
include("logsumexp.jl")

if !isdefined(Base, :get_extension)
    include("../ext/LogExpFunctionsChainRulesCoreExt.jl")
    include("../ext/LogExpFunctionsChangesOfVariablesExt.jl")
    include("../ext/LogExpFunctionsInverseFunctionsExt.jl")
end

end # module
