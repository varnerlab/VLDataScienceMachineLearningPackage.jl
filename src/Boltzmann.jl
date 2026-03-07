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
function sample(model::MySimpleBoltzmannMachineModel, sₒ::Vector{Int64}; 
    T::Int64 = 100, β::Float64 = 1.0)::Array{Int64,2}
    
    # initialize storage -
    W = model.W; # weight matrix
    b = model.b; # bias vector

    number_of_neurons = length(sₒ);
    S = zeros(Int64, number_of_neurons, T);
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

        # Compute the input to each visible neuron, sample v | h -
        for j ∈ 1:number_of_visible_neurons
            IN_v[j] = dot(W[j, :], h) + a[j]; # compute the input for visible node j -
        end
        for j ∈ 1:number_of_visible_neurons
            pᵢ = (1 / (1 + exp(-2 * β * IN_v[j])));  # compute the probability of flipping the j-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the j-th bit with probability pᵢ
            v[j] = flag == 1 ? 1 : -1; # flip the j-th bit for the *next* state
        end

        # Compute the input to each hidden neuron, sample h | v -
        for k ∈ 1:number_of_hidden_neurons
            IN_h[k] = dot(W[:, k], v) + b[k]; # compute the input for hidden node k -
        end
        for k ∈ 1:number_of_hidden_neurons
            pᵢ = (1 / (1 + exp(-2 * β * IN_h[k])));  # compute the probability of flipping the k-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the k-th bit with probability pᵢ
            h[k] = flag == 1 ? 1 : -1; # flip the k-th bit for the *next* state
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

"""
    learn(model::MySimpleBoltzmannMachineModel, X::Matrix{Int};
          η::Float64 = 0.01, β::Float64 = 1.0,
          T::Int = 10000, N_epochs::Int = 100,
          λ::Float64 = 0.0) -> Tuple{MySimpleBoltzmannMachineModel, Vector{Float64}}

Train a Boltzmann machine using exact gradient ascent.

For each epoch, the function runs Gibbs sampling until the chain reaches its stationary
distribution, discards the first half of samples as burn-in, then updates weights and
biases by gradient ascent on the log-likelihood.

### Arguments
- `model::MySimpleBoltzmannMachineModel`: Boltzmann machine model to train (mutated in place).
- `X::Matrix{Int}`: training data matrix of size `n × m`, where each column is a binary pattern in {-1,1}.
- `η::Float64`: learning rate (default: 0.01).
- `β::Float64`: inverse temperature used for Gibbs sampling (default: 1.0).
- `T::Int`: number of Gibbs steps per epoch (default: 10000).
- `N_epochs::Int`: number of training epochs (default: 100).
- `λ::Float64`: L2 weight decay coefficient (default: 0.0). A positive value penalizes large
  weights and prevents the model distribution from collapsing to a single configuration.

### Returns
- `model::MySimpleBoltzmannMachineModel`: updated model with trained weights and biases.
- `history::Vector{Float64}`: Frobenius norm of ΔW at each epoch.
"""
function learn(model::MySimpleBoltzmannMachineModel, X::Matrix{Int};
               η::Float64 = 0.01, β::Float64 = 1.0,
               T::Int = 10000, N_epochs::Int = 100,
               λ::Float64 = 0.0)::Tuple{MySimpleBoltzmannMachineModel, Vector{Float64}}

    # initialize -
    n = size(X, 1);           # number of nodes
    m = size(X, 2);           # number of training patterns
    history = Vector{Float64}(undef, N_epochs);  # convergence history

    # precompute data expectations (done once, outside the loop) -
    C_data = (1/m) * (X * X');        # pairwise correlation matrix (n×n)
    b_data = vec(mean(X, dims=2));    # single-node averages (n-vector)

    # train -
    for e ∈ 1:N_epochs

        # pick a random initial state -
        sₒ = rand([-1, 1], n);

        # run Gibbs sampling to stationarity -
        S = VLDataScienceMachineLearningPackage.sample(model, sₒ, T = T, β = β);

        # discard burn-in (first T÷2 columns), use remaining samples -
        S_stat = S[:, (T ÷ 2 + 1):end];

        # compute model expectations from stationary samples -
        m_stat = size(S_stat, 2);
        C_model = (1/m_stat) * (S_stat * S_stat');
        b_model = vec(mean(S_stat, dims=2));

        # compute updates (with optional L2 weight decay) -
        ΔW = η * (C_data - C_model) - λ .* model.W;
        Δb = η * (b_data - b_model);

        # symmetrize and zero the diagonal of ΔW -
        ΔW = (ΔW + ΔW') / 2 - diagm(diag(ΔW));

        # apply updates -
        model.W += ΔW;
        model.b += Δb;

        # record Frobenius norm of ΔW -
        history[e] = norm(vec(ΔW));
    end

    # return -
    return (model, history);
end

# -- PUBLIC METHODS ABOVE HERE -------------------------------------------------------------------------------------------- #
