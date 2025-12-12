# -- PRIVATE METHODS BELOW HERE ------------------------------------------------------------------------------------------- #
function _sample(model::MyRestrictedBoltzmannMachineModel, pass::MyRBMFeedForwardPassModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)

    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector

    number_of_hidden_neurons = length(b); # number of hidden neurons
    S = zeros(Int, number_of_hidden_neurons, T);
    h = zeros(Float64, number_of_hidden_neurons); # input to the neurons
    IN = zeros(Float64, number_of_hidden_neurons); # input to the neurons
    is_ok_to_stop = false; # flag to stop the simulation

    # package initial state -
    v = copy(sₒ); # visible state is fixed, sample over the hidden state 
   
    # main loop -
    t = 1;
    while (is_ok_to_stop == false)

        # Compute the input to each neuron -
        for i ∈ 1:number_of_hidden_neurons
            IN[i] = dot(W[:, i], v) + b[i]; # compute the input for hidden node i -
        end

        # update the state of the neurons -
        for i ∈ 1:number_of_hidden_neurons
            pᵢ = (1 / (1 + exp(-2 * β * IN[i])));  # compute the probability of flipping the i-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the i-th bit with probability pᵢ
            h[i] = flag == 1 ? 1 : -1; # flip the i-th bit for the *next* state
        end
        
        S[:, t] .= copy(h); # store the current state in the S matrix
        
        if (t == T)
            is_ok_to_stop = true; # stop the simulation
        else
            t += 1; # increment the time step, and continue
        end
    end

    # return the results -    
    return S;
end

function _sample(model::MyRestrictedBoltzmannMachineModel, pass::MyRBMFeedbackPassModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)

    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector

    number_of_visible_neurons = length(a); # number of visible neurons
    v = zeros(Float64, number_of_visible_neurons); # input to the neurons
    S = zeros(Int, number_of_visible_neurons , T);
    IN = zeros(Float64, number_of_visible_neurons ); # input to the neurons
    is_ok_to_stop = false; # flag to stop the simulation

    # package initial state -
    h = copy(sₒ); # *hidden* state is fixed, sample over the visible state 

    # main loop -
    t = 1;
    while (is_ok_to_stop == false)

        # Compute the input to each neuron -
        for i ∈ 1:number_of_visible_neurons
            IN[i] = dot(W[i,:], h) + a[i]; # compute the input for visible node i -
        end

        # update the state of the neurons -
        for i ∈ 1:number_of_visible_neurons
            pᵢ = (1 / (1 + exp(-2 * β * IN[i])));  # compute the probability of flipping the i-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the i-th bit with probability pᵢ
            v[i] = flag == 1 ? 1 : -1; # flip the i-th bit for the *next* state
        end
        S[:, t] .= copy(v); # store the current state in the S matrix
        
        if (t == T)
            is_ok_to_stop = true; # stop the simulation
        else
            t += 1; # increment the time step, and continue
        end
    end

    # return the results -    
    return S;
end

# -- PRIVATE METHODS ABOVE HERE ------------------------------------------------------------------------------------------- #

# -- PUBLIC METHODS BELOW HERE -------------------------------------------------------------------------------------------- #

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

"""
    sample(model::MyRestrictedBoltzmannMachineModel, vₒ::Vector{Int}; 
        T::Int = 100, β::Float64 = 1.0)

Sample from a Restricted Boltzmann Machine (RBM) model. This does a forward pass
and a feedback pass to sample the visible and hidden states.

### Arguments
- `model::MyRestrictedBoltzmannMachineModel`: the RBM model to be simulated.
- `vₒ::Vector{Int}`: the initial visible state.
- `T::Int`: number of internal steps for sampling.
- `β::Float64`: inverse temperature parameter.

### Returns 
- `(v, h)`: a tuple containing the sampled visible state `v` and the hidden state `h`.

"""
function sample(model::MyRestrictedBoltzmannMachineModel, vₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)
    
    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector

    number_of_visible_neurons = length(a); # number of visible neurons
    number_of_hidden_neurons = length(b); # number of hidden neurons

    V = zeros(Int, number_of_visible_neurons, T); # visible trajectory
    H = zeros(Int, number_of_hidden_neurons, T); # hidden trajectory
    v = copy(vₒ); # visible state
    h = zeros(Int, number_of_hidden_neurons); # hidden state
    IN_h = zeros(Float64, number_of_hidden_neurons); # input to hidden neurons
    IN_v = zeros(Float64, number_of_visible_neurons); # input to visible neurons
    is_ok_to_stop = false; # flag to stop the simulation

    # package initial state -
    V[:, 1] .= v; # store the initial visible state
    for k ∈ 1:number_of_hidden_neurons
        IN_h[k] = dot(W[:, k], v) + b[k]; # input for hidden node k
    end
    for k ∈ 1:number_of_hidden_neurons
        pᵢ = (1 / (1 + exp(-2 * β * IN_h[k])));  # probability of flipping the k-th bit
        flag = Bernoulli(pᵢ) |> rand # flip the k-th bit with probability pᵢ
        h[k] = flag == 1 ? 1 : -1; # flip the k-th bit for the *next* state
    end
    H[:, 1] .= h; # store the initial hidden sample

    # main loop -
    t = 2;
    while (is_ok_to_stop == false)

        # Compute the input to each hidden neuron, sample h | v -
        for k ∈ 1:number_of_hidden_neurons
            IN_h[k] = dot(W[:, k], v) + b[k]; # compute the input for hidden node k -
        end
        for k ∈ 1:number_of_hidden_neurons
            pᵢ = (1 / (1 + exp(-2 * β * IN_h[k])));  # compute the probability of flipping the k-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the k-th bit with probability pᵢ
            h[k] = flag == 1 ? 1 : -1; # flip the k-th bit for the *next* state
        end

        # Compute the input to each visible neuron, sample v | h -
        for j ∈ 1:number_of_visible_neurons
            IN_v[j] = dot(W[j, :], h) + a[j]; # compute the input for visible node j -
        end
        for j ∈ 1:number_of_visible_neurons
            pᵢ = (1 / (1 + exp(-2 * β * IN_v[j])));  # compute the probability of flipping the j-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the j-th bit with probability pᵢ
            v[j] = flag == 1 ? 1 : -1; # flip the j-th bit for the *next* state
        end

        V[:, t] .= copy(v); # store the current visible state
        H[:, t] .= copy(h); # store the current hidden state
        
        if (t == T)
            is_ok_to_stop = true; # stop the simulation
        else
            t += 1; # increment the time step, and continue
        end
    end

    # return the results -
    return (V, H);
end

"""
    learn(model::MyRestrictedBoltzmannMachineModel, data::Array{Int64,2}, p::Categorical;
        maxnumberofiterations::Int = 100, T::Int = 100, β::Float64 = 1.0, batchsize::Int = 10, η::Float64 = 0.01,
            tol::Float64 = 1e-6, verbose::Bool = true) -> MyRestrictedBoltzmannMachineModel

Train a Restricted Boltzmann Machine (RBM) model using Contrastive Divergence (CD) algorithm.

### Arguments
- `model::MyRestrictedBoltzmannMachineModel`: the RBM model to be trained.
- `data::Array{Int64,2}`: the training data, a matrix of size (number_of_visible_neurons, number_of_samples).
- `p::Categorical`: a categorical distribution for sampling indices.
- `maxnumberofiterations::Int`: maximum number of iterations for training.
- `T::Int`: number of internal steps for sampling.
- `β::Float64`: inverse temperature parameter.
- `batchsize::Int`: size of the batch for training.
- `η::Float64`: learning rate.
- `tol::Float64`: relative parameter change tolerance for early stopping.
- `verbose::Bool`: whether to print progress information.

### Return 
- `MyRestrictedBoltzmannMachineModel`: the trained RBM model.

"""
function learn(model::MyRestrictedBoltzmannMachineModel, data::Array{Int64,2}, p::Categorical;
    maxnumberofiterations::Int = 100, T::Int = 100, β::Float64 = 1.0, batchsize::Int = 10, η::Float64 = 0.01,
    tol::Float64 = 1e-6, verbose::Bool = true)::MyRestrictedBoltzmannMachineModel

    # initialize -
    W = model.W # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector
    is_ok_to_stop = false; # flag to stop the simulation
    number_of_internal_steps = T; # number of internal steps that we take in the sampling step -
    counter = 1;

    # system size -
    number_of_visible_neurons = size(W, 1); # number of visible neurons
    number_of_hidden_neurons = size(W, 2); # number of hidden neurons

    # main loop -
    while (is_ok_to_stop == false)

        W_prev = copy(W); # stash current params
        b_prev = copy(b);
        a_prev = copy(a);

        # generate some training data for this round -
        idx_batch_set = Set{Int64}();
        is_batch_set_full = false;
        while (is_batch_set_full == false)
        
            idx = rand(p); # generate a random index 
            push!(idx_batch_set, idx); # add to the set - this will fail if the index is already in the set
            if (length(idx_batch_set) == batchsize)
                is_batch_set_full = true; # ok to stop    
            end
        end
        idx_batch_vector = idx_batch_set |> collect |> sort;
        
        # process each of the batch elements -
        for i ∈ eachindex(idx_batch_vector)
            idx = idx_batch_vector[i]; # get the index
            xₒ = data[:, idx]; # get initial state that we will sample from
            
            # sample - 
            (v,h) = sample(model, xₒ, T = number_of_internal_steps, β = β);

            # ok, so we have the visible and hidden states from sampling - weights
            for j ∈ 1:number_of_visible_neurons
                for k ∈ 1:number_of_hidden_neurons
                    W[j,k] += η * (xₒ[j] * h[k, 1] - v[j, end] * h[k, end]); # positive phase minus negative phase
                end
            end

            # hidden bias update
            for k ∈ 1:number_of_hidden_neurons
                b[k] += η * (h[k, 1] - h[k, end]); # positive phase minus negative phase
            end

            # visible bias update
            for j ∈ 1:number_of_visible_neurons
                a[j] += η * (xₒ[j] - v[j, end]); # positive phase minus negative phase
            end
        end

        # compute parameter change for early stopping -
        ΔW = norm(W .- W_prev) / (norm(W_prev) + eps(Float64));
        Δb = norm(b .- b_prev) / (norm(b_prev) + eps(Float64));
        Δa = norm(a .- a_prev) / (norm(a_prev) + eps(Float64));
        parameter_change = maximum((ΔW, Δb, Δa));

        if (verbose == true)
            println("Iteration: ", counter, ", max parameter change: ", parameter_change);
        end
        
        # check for convergence - should we stop?
        if ((counter ≥ maxnumberofiterations) || (parameter_change < tol))
            is_ok_to_stop = true; # stop the training 
        else
            counter += 1; # increment the counter
        end
    end

    # build a new model (that we'll return) after we estimate the W, a and b parameters -
    idmodel = build(MyRestrictedBoltzmannMachineModel, (
        W = W,
        b = b,
        a = a
    ));

    # return -
    return idmodel;
end

# -- PUBLIC METHODS ABOVE HERE -------------------------------------------------------------------------------------------- #
