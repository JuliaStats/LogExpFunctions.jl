# LogExpFunctions

Various special functions based on `log` and `exp` moved from [StatsFuns.jl](https://github.com/JuliaStats/StatsFuns.jl) into a separate package, to minimize dependencies. These functions only use native Julia code, so there is no need to depend on `librmath` or similar libraries. See the discussion at [StatsFuns.jl#46](https://github.com/JuliaStats/StatsFuns.jl/issues/46).

The original authors of these functions are the StatsFuns.jl contributors.

LogExpFunctions supports [`InverseFunctions.inverse`](https://github.com/JuliaMath/InverseFunctions.jl) and [`ChangesOfVariables.test_with_logabsdet_jacobian`](https://github.com/JuliaMath/ChangesOfVariables.jl) for `log1mexp`, `log1pexp`, `log2mexp`, `logexpm1`, `logistic`, `logit`, and `logcosh` (no inverse).

```@docs
xlogx
xlogy
xlog1py
xexpx
xexpy
logistic
logit
logcosh
logabssinh
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
logsumexp!
softmax!
softmax
cloglog
cexpexp
```
