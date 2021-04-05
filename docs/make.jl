using Documenter, LogExpFunctions

makedocs(
    modules = [LogExpFunctions],
    format = Documenter.HTML(),
    checkdocs = :exports,
    sitename = "LogExpFunctions.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/JuliaStats/LogExpFunctions.jl.git",
)
