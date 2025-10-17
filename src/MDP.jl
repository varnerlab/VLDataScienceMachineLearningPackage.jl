"""
    myrandpolicy(problem::MyMDPProblemModel, world::MyRectangularGridWorldModel, s::Int)::Int

This function implements a random policy for a given MDP problem and world model.

### Arguments
- `problem::MyMDPProblemModel`: the MDP problem model
- `world::MyRectangularGridWorldModel`: the world model
- `s::Int`: the state

### Returns
- `Int`: the action we choose
"""
function myrandpolicy(problem::MyMDPProblemModel, 
    world::MyRectangularGridWorldModel, s::Int)::Int

    # initialize -
    d = Categorical([0.25,0.25,0.25,0.25]); # you specify this

    # should keep chooseing -
    should_choose_gain = true;
    a = -1; # default
    while (should_choose_gain == true)
    
        # initialize a random categorical distribution over actions -
        aᵢ = rand(d);
        
        # get the move, and the current location -
        Δ = world.moves[aᵢ];
        current_position = world.coordinates[s]
        new_position =  current_position .+ Δ
        if (haskey(world.states, new_position) == true)
            a = aᵢ
            should_choose_gain = false;
        end
    end

    return a;
end;

"""
    myrandstep(problem::MyMDPProblemModel, world::MyRectangularGridWorldModel, s::Int, a::Int)

This function implements a random step for a given MDP problem and world model.

### Arguments
- `problem::MyMDPProblemModel`: the MDP problem model
- `world::MyRectangularGridWorldModel`: the world model
- `s::Int`: the state we are in
- `a::Int`: the action we choose

### Returns
- `(s′,r)`: a tuple of the next state and the reward for being in state `s` and taking action `a`
"""
function myrandstep(problem::MyMDPProblemModel, 
    world::MyRectangularGridWorldModel, s::Int, a::Int)

    # get the reward value -
    r = problem.R[s,a];

    # get the move, and the current location -
    Δ = world.moves[a];
    current_position = world.coordinates[s]

    # propose a new position -
    new_position =  current_position .+ Δ
    s′ = s; # default, we don't do anything
    if (haskey(world.states, new_position) == true)
        s′ = world.states[new_position];
    end

    # return -
    return (s′,r)
end;

"""
    lookahead(p::MyMDPProblemModel, U::Vector{Float64}, s::Int64, a::Int64)

This function computes the lookahead value for a given state-action pair `(s,a)`. 
It uses a vector `U` to compute the value function.

### Arguments
- `p::MyMDPProblemModel`: the MDP problem model
- `U::Vector{Float64}`: the value function vector
- `s::Int64`: the state
- `a::Int64`: the action

### Returns
- `Float64`: the lookahead value for the state-action pair `(s,a)`. 
"""
function lookahead(p::MyMDPProblemModel, U::Vector{Float64}, s::Int64, a::Int64)::Float64

    # grab stuff from the problem -
    R = p.R;  # reward -
    T = p.T;    
    γ = p.γ;
    𝒮 = p.𝒮;
    
    # compute the lookahead value and return it
    return R[s,a] + γ*sum(T[s,s′,a]*U[i] for (i,s′) in enumerate(𝒮))
end

"""
    lookahead(p::MyMDPProblemModel, U::Function, s::Int64, a::Int64)::Float64

This function computes the lookahead value for a given state-action pair `(s,a)`. 
It uses a function `U` to compute the value function.
"""
function lookahead(p::MyMDPProblemModel, U::Function, s::Int64, a::Int64)

    # get data from the problem -
    𝒮, T, R, γ = p.𝒮, p.T, p.R, p.γ;
    return R[s,a] + γ*sum(T[s,s′,a]*U(s′) for s′ in 𝒮)
end


"""
    Q(p::MyMDPProblemModel, U::Array{Float64,1}) -> Array{Float64,2}

This function computes the Q-value function for a given value function `U`.

### Arguments
- `p::MyMDPProblemModel`: the MDP problem model
- `U::Array{Float64,1}`: the value function vector

### Returns
- `Array{Float64,2}`: the Q-value function
"""
function Q(p::MyMDPProblemModel, U::Array{Float64,1})::Array{Float64,2}

    # grab stuff from the problem -
    R = p.R;  # reward -
    T = p.T;    
    γ = p.γ;
    𝒮 = p.𝒮;
    𝒜 = p.𝒜

    # initialize -
    Q_array = Array{Float64,2}(undef, length(𝒮), length(𝒜))

    # compute the Q-value function -
    for i ∈ eachindex(𝒮)
        s = 𝒮[i]; # get the state s
        for j ∈ eachindex(𝒜)
            a = 𝒜[j]; # get the action a

            # compute the Q-value -
            # We get the reward for being in state s and taking action a, 
            # and then we add the discounted sum of the future value function for the next state s′.
            Q_array[s,a] = R[s,a] + γ*sum([T[s,s′,a]*U[s′] for s′ in 𝒮]);
        end
    end

    # return -
    return Q_array
end

"""
    policy(Q_array::Array{Float64,2}) -> Array{Int,1}

This function computes the policy from the Q-value function.

### Arguments
- `Q_array::Array{Float64,2}`: the Q-value function

### Returns
- `Array{Int,1}`: the policy which maps states to actions
"""
function policy(Q_array::Array{Float64,2})::Array{Int64,1}

    # get the dimension -
    (NR, _) = size(Q_array);

    # initialize some storage -
    π_array = Array{Int64,1}(undef, NR)
    for s ∈ 1:NR
        π_array[s] = argmax(Q_array[s,:]);
    end

    # return -
    return π_array;
end

"""
    backup(problem::MyMDPProblemModel, U::Array{Float64,1}, s::Int64) -> Float64

This function computes the backup value for a given state `s` and value function `U`.

### Arguments
- `problem::MyMDPProblemModel`: the MDP problem model
- `U::Array{Float64,1}`: the value function vector
- `s::Int64`: the state

### Returns
- `Float64`: the best backup value for the state `s`
"""
function backup(problem::MyMDPProblemModel, U::Array{Float64,1}, s::Int64)::Float64
    return maximum(lookahead(problem, U, s, a) for a ∈ problem.𝒜);
end

"""
    solve(model::MyValueIterationModel, problem::MyMDPProblemModel) -> MyValueFunctionPolicy

This function solves the MDP problem using value iteration.

### Arguments
- `model::MyValueIterationModel`: the value iteration model
- `problem::MyMDPProblemModel`: the MDP problem model

### Returns
- `MyValueFunctionPolicy`: the value function policy
"""
function solve(model::MyValueIterationModel, problem::MyMDPProblemModel)::MyValueFunctionPolicy
    
    # data -
    k_max = model.k_max;

    # initialize
    U = [0.0 for _ ∈ problem.𝒮];

    # main loop -
    for _ ∈ 1:k_max
        U = [backup(problem, U, s) for s ∈ problem.𝒮];
    end

    return MyValueFunctionPolicy(problem, U);
end

"""
    solve(model::MyRandomRolloutModel, problem::MyMDPProblemModel, 
        world::MyRectangularGridWorldModel, s::Int64) -> Float64

This function solves the MDP problem using random rollouts.

### Arguments
- `model::MyRandomRolloutModel`: the random rollout model
- `problem::MyMDPProblemModel`: the MDP problem model
- `world::MyRectangularGridWorldModel`: the world model
- `s::Int64`: the state

### Returns
- `Float64`: the estimated utility value of the state `s`
"""
function solve(model::MyRandomRolloutModel, problem::MyMDPProblemModel, 
        world::MyRectangularGridWorldModel, s::Int64)::Float64

    # initialize -
    ret = 0.0;
    γ = problem.γ;
    visited_states = Set{Int64}();
    is_ok_to_stop = false;
    i = 1;
    depth = model.depth;

    while (is_ok_to_stop == false)
       
        a = myrandpolicy(problem, world, s);
        s, r = myrandstep(problem, world, s, a);

        if (s ∉ visited_states)
            push!(visited_states, s);
            ret += r*γ^(i-1);
            i += 1;

            # can we stop?
            if (length(visited_states) ≥ depth)
                is_ok_to_stop = true;
            end
        end
    end    
    
    # for i ∈ 1:depth
    #     a = myrandpolicy(problem, world, s);
    #     s, r = myrandstep(problem, world, s, a);
    #     ret += r*γ^(i-1);
    # end

    # return -
    return ret;
end;

"""
    iterative_policy_evaluation(p::MyMDPProblemModel, π, k_max::Int) -> Array{Float64,1}

This function performs iterative policy evaluation for a given MDP problem and policy.

### Arguments
- `p::MyMDPProblemModel`: the MDP problem model
- `π`: the policy function
- `k_max::Int`: the maximum number of iterations

"""
function iterative_policy_evaluation(p::MyMDPProblemModel, π, k_max::Int)

    # grab stuff from the problem -
    R = p.R;  # reward -
    T = p.T;    
    γ = p.γ;
    𝒮 = p.𝒮;

    # initialize value -
    U = [0.0 for s ∈ 𝒮];

    for _ ∈ 1:k_max
        U = [lookahead(p, U, s, π(s)) for s ∈ 𝒮]
    end

    return U;
end

"""
    greedy(problem::MyMDPProblemModel, U::Array{Float64,1}, s::Int64) -> (a::Int64, u::Float64)

This function computes the greedy action and its value for a given state `s` and value function `U`.

### Arguments
- `problem::MyMDPProblemModel`: the MDP problem model
- `U::Array{Float64,1}`: the value function vector
- `s::Int64`: the state

### Returns
- `(a::Int64, u::Float64)`: a tuple of the best action and its value
"""
function greedy(problem::MyMDPProblemModel, U::Array{Float64,1}, s::Int64)
    u, a = findmax(a->lookahead(problem, U, s, a), problem.𝒜);
    return (a=a, u=u)
end

(π::MyValueFunctionPolicy)(s::Int64) = greedy(π.problem, π.U, s).a;
