InverseFunctions.inverse(::typeof(log1pexp)) = logexpm1
InverseFunctions.inverse(::typeof(logexpm1)) = log1pexp

InverseFunctions.inverse(::typeof(log1mexp)) = log1mexp

InverseFunctions.inverse(::typeof(log2mexp)) = log2mexp

InverseFunctions.inverse(::typeof(logit)) = logistic
InverseFunctions.inverse(::typeof(logistic)) = logit
