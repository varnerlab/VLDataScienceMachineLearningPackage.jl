using VLDataScienceMachineLearningPackage
using Test

@testset "VLDataScienceMachineLearningPackage.jl" begin

    @testset "Perceptron and the pocket algorithm" begin
        mistakes(X, y, β) = count(y .* (X*β) .≤ 0); # training examples on the wrong side of the boundary, or on it
        perceptron() = build(MyPerceptronClassificationModel, (parameters = zeros(3), mistakes = 0));

        # separable data: both versions stop with no training mistakes and return the same coefficients -
        X = [1.0 1.0 1.0; 2.0 1.5 1.0; -1.0 -1.0 1.0; -2.0 -0.5 1.0];
        y = [1.0, 1.0, -1.0, -1.0];
        classical = learn(X, y, perceptron(); maxiter = 100);
        pocket = learn(X, y, perceptron(); maxiter = 100, pocket = true);
        @test mistakes(X, y, classical.β) == 0
        @test pocket.β == classical.β

        # XOR on the corners of a square: never separable, and the pocket never does worse -
        X = [1.0 1.0 1.0; -1.0 -1.0 1.0; 1.0 -1.0 1.0; -1.0 1.0 1.0];
        y = [-1.0, -1.0, 1.0, 1.0];
        classical = learn(X, y, perceptron(); maxiter = 50);
        pocket = learn(X, y, perceptron(); maxiter = 50, pocket = true);
        @test mistakes(X, y, classical.β) > 0
        @test mistakes(X, y, pocket.β) ≤ mistakes(X, y, classical.β)

        # the default is the classical algorithm -
        @test learn(X, y, perceptron(); maxiter = 50).β == classical.β
    end
end
