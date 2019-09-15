# LogExpFunctions.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/tpapp/LogExpFunctions.jl.svg?branch=master)](https://travis-ci.com/tpapp/LogExpFunctions.jl)
[![codecov.io](http://codecov.io/github/tpapp/LogExpFunctions.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/LogExpFunctions.jl?branch=master)<!-- [![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/LogExpFunctions.jl/stable) -->
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/LogExpFunctions.jl/latest)

Various special functions based on `log` and `exp` moved from [StatsFuns.jl](https://github.com/JuliaStats/StatsFuns.jl) into a separate package, to minimize dependencies. These functions only use native Julia code, so there is no need to depend on `librmath` or similar libraries. See the discussion at [StatsFuns.jl#46](https://github.com/JuliaStats/StatsFuns.jl/issues/46).

The original authors of these functions are the StatsFuns.jl contributors.
