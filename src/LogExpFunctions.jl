module LogExpFunctions

using DocStringExtensions: SIGNATURES
using Base: Math.@horner

import IrrationalConstants
import LinearAlgebra

export xlogx, xlogy, xlog1py, xexpx, xexpy, logistic, logit, log1psq, log1pexp, log1mexp, log2mexp, logexpm1,
    softplus, invsoftplus, log1pmx, logmxp1, logaddexp, logsubexp, logsumexp, logsumexp!, softmax,
    softmax!, logcosh, cloglog, cexpexp

include("basicfuns.jl")
include("logsumexp.jl")

if !isdefined(Base, :get_extension)
    include("../ext/ChainRulesCoreExt.jl")
    include("../ext/ChangesOfVariablesExt.jl")
    include("../ext/InverseFunctionsExt.jl")
end

end # module
