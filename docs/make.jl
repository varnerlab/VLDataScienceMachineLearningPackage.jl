using Documenter, VLDataScienceMachineLearningPackage

push!(LOAD_PATH,"../src/")

makedocs(
    sitename="VLDataScienceMachineLearningPackage",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [VLDataScienceMachineLearningPackage],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git", branch = "gh-pages", target = "build"
)