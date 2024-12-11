module LogExpFunctionsInverseFunctionsExt

using LogExpFunctions
import InverseFunctions

InverseFunctions.inverse(::typeof(log1pexp)) = logexpm1
InverseFunctions.inverse(::typeof(logexpm1)) = log1pexp

InverseFunctions.inverse(::typeof(log1mexp)) = log1mexp

InverseFunctions.inverse(::typeof(log2mexp)) = log2mexp

InverseFunctions.inverse(::typeof(logit)) = logistic
InverseFunctions.inverse(::typeof(logistic)) = logit

InverseFunctions.inverse(::typeof(cloglog)) = cexpexp
InverseFunctions.inverse(::typeof(cexpexp)) = cloglog

InverseFunctions.inverse(::typeof(loglogistic)) = logitexp
InverseFunctions.inverse(::typeof(logitexp)) = loglogistic

InverseFunctions.inverse(::typeof(log1mlogistic)) = logit1mexp
InverseFunctions.inverse(::typeof(logit1mexp)) = log1mlogistic

InverseFunctions.inverse(::typeof(softplus)) = invsoftplus
function InverseFunctions.inverse(f::Base.Fix2{typeof(softplus),<:Real})
    Base.Fix2(invsoftplus, f.x)
end

InverseFunctions.inverse(::typeof(invsoftplus)) = softplus
function InverseFunctions.inverse(f::Base.Fix2{typeof(invsoftplus),<:Real})
    Base.Fix2(softplus, f.x)
end

end # module
