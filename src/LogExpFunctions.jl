module LogExpFunctions

using DocStringExtensions: SIGNATURES
using Base: Math.@horner, @irrational
using LinearAlgebra: LinearAlgebra

export loghalf, logtwo, logπ, log2π, log4π
export xlogx, xlogy, logistic, logit, log1psq, log1pexp, log1mexp, log2mexp, logexpm1,
    softplus, invsoftplus, log1pmx, logmxp1, logaddexp, logsubexp, logsumexp, softmax,
    softmax!

include("constants.jl")
include("basicfuns.jl")
include("logsumexp.jl")

end # module
