using LogExpFunctions
using ChainRulesTestUtils
using OffsetArrays

using Random
using Test

Random.seed!(1234)

include("basicfuns.jl")
include("chainrules.jl")
