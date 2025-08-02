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

include("common/ULPError.jl")
using .ULPError

Random.seed!(1234)

include("basicfuns.jl")
include("chainrules.jl")
include("inverse.jl")
include("with_logabsdet_jacobian.jl")

# QA
import JET
JET.report_package("LogExpFunctions")
import Aqua
Aqua.test_all(LogExpFunctions)
