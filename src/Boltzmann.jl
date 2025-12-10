"""
    sample(model::MySimpleBoltzmannMachineModel, sₒ::Vector{Int}; T::Int = 100, β::Float64 = 1.0) -> Array{Int,2}

Simulate asynchronous Gibbs updates for a simple Boltzmann machine starting from an initial spin configuration.

### Arguments
- `model::MySimpleBoltzmannMachineModel`: model containing weight matrix `W` and bias vector `b`.
- `sₒ::Vector{Int}`: initial spin state vector (+1/-1) for each neuron.
- `T::Int = 100`: number of time steps to simulate; each column in the output records a step.
- `β::Float64 = 1.0`: inverse temperature controlling flip probabilities.

### Returns
- `S::Array{Int,2}`: matrix of spin states over time (neurons × time).
"""
function sample(model::MySimpleBoltzmannMachineModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)::Array{Int,2}
    
    # initialize storage -
    W = model.W; # weight matrix
    b = model.b; # bias vector

    number_of_neurons = length(sₒ);
    S = zeros(Int, number_of_neurons, T);
    is_ok_to_stop = false; # flag to stop the simulation
    h = zeros(Float64, number_of_neurons); # input to the neurons

    # package initial state -
    s = copy(sₒ); # initial state
    S[:, 1] .= s; # store the initial state in the S matrix

    # main loop -
    t = 2;
    while (is_ok_to_stop == false)

        # Compute the input to each neuron -
        for i ∈ 1:number_of_neurons
            h[i] = dot(W[i, :], s) + b[i]; # compute the input for node i -
        end

        # update the state of the neurons -
        for i ∈ 1:number_of_neurons
            pᵢ = (1 / (1 + exp(-2 * β * h[i])));  # compute the probability of flipping the i-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the i-th bit with probability pᵢ
            s[i] = flag == 1 ? 1 : -1; # flip the i-th bit for the *next* state
        end
        
        S[:, t] .= copy(s); # store the current state in the S matrix
        
        if (t == T)
            is_ok_to_stop = true; # stop the simulation
        else
            t += 1; # increment the time step, and continue
        end
    end

    # return the results -    
    return S;
end
