# LogExpFunctions

Various special functions based on `log` and `exp` moved from [StatsFuns.jl](https://github.com/JuliaStats/StatsFuns.jl) into a separate package, to minimize dependencies. These functions only use native Julia code, so there is no need to depend on `librmath` or similar libraries. See the discussion at [StatsFuns.jl#46](https://github.com/JuliaStats/StatsFuns.jl/issues/46).

The original authors of these functions are the StatsFuns.jl contributors.

```@docs
xlogx
xlogy
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
logsumexp
softmax!
softmax
```

## Constants

Additionally, LogExpFunctions.jl reexports the following logarithmic constants defined in
[IrrationalConstants.jl](https://github.com/JuliaMath/IrrationalConstants.jl).

```julia
loghalf    # log(1 / 2)
logtwo     # log(2)
logπ       # log(π)
log2π      # log(2π)
log4π      # log(4π)
```
