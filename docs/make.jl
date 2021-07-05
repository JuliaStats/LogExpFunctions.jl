using Documenter, LogExpFunctions

# Doctest setup
DocMeta.setdocmeta!(
    LogExpFunctions, :DocTestSetup, :(using LogExpFunctions); recursive=true
)

makedocs(;
    modules=[LogExpFunctions],
    format=Documenter.HTML(),
    sitename="LogExpFunctions.jl",
    pages=Any["index.md"],
    checkdocs=:exports,
    strict=true,
)

deploydocs(; repo="github.com/JuliaStats/LogExpFunctions.jl.git")
