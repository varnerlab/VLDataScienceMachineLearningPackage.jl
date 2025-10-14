using Documenter, VLDataScienceMachineLearningPackage

push!(LOAD_PATH,"../src/")

makedocs(
    sitename="VLDataScienceMachineLearningPackage",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [VLDataScienceMachineLearningPackage],
    pages = [
        "Home" => "index.md",
        "Data" => "data.md",
        "Types" => "types.md",
        "Factory" => "factory.md",
        "Text" => "text.md",
        "Wolfram" => "wolfram.md",
        "Graphs" => "graphs.md",
        "Linear" => "solvers.md",
        "Binary Classification" => "binaryclassification.md",
        "Reinforcement Learning" => [
            "Markov Decision Processes" => "mdp.md",
        ],
    ], 
)

deploydocs(
    repo = "github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git", branch = "gh-pages", target = "build"
)