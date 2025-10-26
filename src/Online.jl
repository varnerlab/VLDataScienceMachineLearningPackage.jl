
"""
    play(model::MyBinaryWeightedMajorityAlgorithmModel, data::Array{Float64,2})

Play the Binary Weighted Majority Algorithm. This function simulates the Binary Weighted Majority Algorithm
using the given model and data. The function returns a tuple with two elements. The first element is a matrix
with the results of the simulation. The second element is the weights of the experts at the end of the simulation.

### Arguments
- `model::MyBinaryWeightedMajorityAlgorithmModel`: the model to simulate
- `world::Function`: the world function that generates the actual outcomes
- `data::Array{Float64,2}`: the data to use in the simulation

### Returns
- `Tuple{Array{Int64,2}, Array{Float64,2}}`: a tuple with the results of the simulation and the weights of the experts
"""
function play(model::MyBinaryWeightedMajorityAlgorithmModel, 
    data::Array{Float64,2})::Tuple{Array{Int64,2}, Array{Float64,2}}

    # initialize -
    n = model.n; # how many experts do we have?
    T = model.T; # how many rounds do we play?
    ϵ = model.ϵ; # learning rate
    weights = model.weights; # weights of the experts
    expert = model.expert; # expert function
    adversary = model.adversary; # adversary function
    results_array = zeros(Int64, T, 3+n); # aggregator predictions

    # main simulation loop -
    for t ∈ 1:T
        
        # query the experts -
        expert_predictions = zeros(Int64, n);
        for i ∈ 1:n
            expert_predictions[i] = expert(i, t, data); # call the expert function, returns a prediction for expert i at time t-1
        end

        # store the expert predictions -
        for i ∈ 1:n
            results_array[t, i] = expert_predictions[i];
        end

        # compute the weighted prediction -
        weight_down_vote = findall(x-> x == -1, expert_predictions) |> i-> sum(weights[t, i]);
        weight_up_vote = findall(x-> x == 1, expert_predictions) |> i-> sum(weights[t, i]);
        aggregator_prediction = (weight_up_vote > weight_down_vote) ? 1 : -1;
        results_array[t,n+1] = aggregator_prediction; # store the aggregator prediction

        # query the adversary -
        actual = adversary(t, data); # call the adversary function, returns the actual outcome at time t
        results_array[t, n+2] = actual; # store the adversary outcome

        # compute the aggregator loss -
        results_array[t, end] = (aggregator_prediction == actual) ? 0 : 1;

        # compute the loss for each expert -
        loss = zeros(Float64, n);
        for i ∈ 1:n
            loss[i] = (expert_predictions[i] == actual) ? 0.0 : 1.0; # change the sign of the loss, to update the weights
        end

        # update the weights -
        for i ∈ 1:n
            weights[t+1, i] = weights[t, i]*(1 - ϵ*loss[i]);
        end
    end

    # return -
    return (results_array, weights);
end