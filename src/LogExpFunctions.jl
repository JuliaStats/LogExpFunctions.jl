module LogExpFunctions

using DocStringExtensions: SIGNATURES
using Base: Math.@horner

import IrrationalConstants
import LinearAlgebra

export xlogx, xlogy, logistic, logit, log1psq, log1pexp, log1mexp, log2mexp, logexpm1,
    softplus, invsoftplus, log1pmx, logmxp1, logaddexp, logsubexp, logsumexp, softmax,
    softmax!

include("basicfuns.jl")
include("logsumexp.jl")

end # module
