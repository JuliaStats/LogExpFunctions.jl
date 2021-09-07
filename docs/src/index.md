# LogExpFunctions

Various special functions based on `log` and `exp` moved from [StatsFuns.jl](https://github.com/JuliaStats/StatsFuns.jl) into a separate package, to minimize dependencies. These functions only use native Julia code, so there is no need to depend on `librmath` or similar libraries. See the discussion at [StatsFuns.jl#46](https://github.com/JuliaStats/StatsFuns.jl/issues/46).

The original authors of these functions are the StatsFuns.jl contributors.

```@docs
xlogx
xlogy
xlog1py
logistic
logit
log1psq
log1pexp
log1mexp
log2mexp
logexpm1
log1pmx
logmxp1
logaddexp
logsubexp
logsumexp
softmax!
softmax
```
