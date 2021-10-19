using LogExpFunctions
using ChainRulesTestUtils
using InverseFunctions
using OffsetArrays

using Random
using Test

Random.seed!(1234)

include("basicfuns.jl")
include("chainrules.jl")
include("inverse.jl")
