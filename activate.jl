using Revise
using Pkg

# Package
Pkg.activate("C:/Users/domma/Dropbox/Software/LogExpFunctions.jl/")

using LogExpFunctions
using CairoMakie


xrange = range(-1.5, 1.5, length=100)
yexp = exp.(xrange)
ysoftplus1 = softplus.(xrange)
ysoftplus2 = softplus.(xrange; a=2)
ysoftplus3 = softplus.(xrange; a=3)

ysoftplus5 = softplus.(xrange; a=5)
ysoftplus10 = softplus.(xrange; a=10)


# f = lines(xrange, yexp, color=:black)
f = lines(xrange, ysoftplus1, color=:red)
lines!(xrange, ysoftplus2, color=:orange)
lines!(xrange, ysoftplus3, color=:darkorange)
lines!(xrange, ysoftplus5, color=:green)
lines!(xrange, ysoftplus10, color=:blue)

ablines!(0, 1, color=:grey, linestyle=:dash)
f

softplus(0; a=3)