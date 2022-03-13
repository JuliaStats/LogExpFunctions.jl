module LogExpFunctions

using DocStringExtensions: SIGNATURES
using Base: Math.@horner

import ChainRulesCore
import ChangesOfVariables
import InverseFunctions
import IrrationalConstants
import LinearAlgebra

export xlogx, xlogy, xlog1py, xexpx, xexpy, logistic, logit, log1psq, log1pexp, log1mexp, log2mexp, logexpm1,
    softplus, invsoftplus, log1pmx, logmxp1, logaddexp, logsubexp, logsumexp, logsumexp!, softmax,
    softmax!, logcosh

include("basicfuns.jl")
include("logsumexp.jl")
include("chainrules.jl")
include("inverse.jl")
include("with_logabsdet_jacobian.jl")

end # module
