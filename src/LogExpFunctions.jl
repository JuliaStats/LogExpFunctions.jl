module LogExpFunctions

export xlogx, xlogy, logistic, logit, log1psq, log1pexp, log1mexp, log2mexp, logexpm1,
    log1pmx, logmxp1, logaddexp, logsubexp, logsumexp, softmax!, softmax

using DocStringExtensions: SIGNATURES

using Base: Math.@horner, @irrational

include("constants.jl")
include("basicfuns.jl")

end # module
