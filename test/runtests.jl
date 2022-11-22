using LogExpFunctions
using ChainRulesTestUtils
using ChainRulesCore
using ChangesOfVariables
using FiniteDifferences
using ForwardDiff
using InverseFunctions
using OffsetArrays

using Random
using Test

Random.seed!(1234)

include("basicfuns.jl")
include("chainrules.jl")
include("inverse.jl")
include("with_logabsdet_jacobian.jl")
