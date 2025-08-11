# -- PRIVATE API BELOW HERE ------------------------------------------------------------------------ #
function _simulate(algorithm::WolframDeterministicSimulation, rulemodel::MyOneDimensionalElementaryWolframRuleModel, initial::Array{Int64,1}; 
    steps::Int64 = 240, maxnumberofmoves::Union{Int64, Nothing} = nothing, 
    parameters::Union{Nothing, Dict{Int, Float64}} = nothing,
    cooldownlength::Int64 = 0)::Dict{Int64, Array{Int64,2}}
    
    # get stuff from model -
    radius = rulemodel.radius; # how many cells am I looking at?
    number_of_colors = rulemodel.number_of_colors; # how many colors (states) can each cell have?
    width = length(initial); # how many cells are there?

    # cooldown -
    cooldown = Dict{Int64, Int64}(); # cooldown for each cell
    foreach(i -> cooldown[i] = 0, 1:width); # initialize cooldown for each cell

    # initialize -
    frames = Dict{Int64, Array{Int64,2}}();
    frame = Array{Int64,2}(undef, steps, width) |> X -> fill!(X, 0);

    # set the initial state -
    foreach(i -> frame[1,i] = initial[i], 1:width);    
    frames[1] = frame; # set the initial frame -
    
    # main loop -
    for time ∈ 2:steps

        # create the next frame -
        frame = copy(frames[time-1]);
        tmp = Array{Int64,1}(undef, radius);
        for i ∈ 1:width

            index = nothing;
            if (i == 1)
                
                tmp[1] = frame[time-1, width];      # left
                tmp[2] = frame[time-1, i];          # center
                tmp[3] = frame[time-1, i + 1];      # right

                # compute the index (this is binary, so we need to compute from left to right)
                index = parse(Int, join(tmp), base = number_of_colors);
            elseif (i == width)
                    
                tmp[1] = frame[time-1, i - 1];      # left
                tmp[2] = frame[time-1, i];          # center
                tmp[3] = frame[time-1, 1];          # right
    
                # compute the index (this is binary, so we need to compute from left to right)
                index = parse(Int, join(tmp), base = number_of_colors);
            else
                
                tmp[1] = frame[time-1, i - 1];      # left
                tmp[2] = frame[time-1, i];          # center
                tmp[3] = frame[time-1, i + 1];      # right

                # compute the index (this is binary, so we need to compute from left to right)
                index = parse(Int, join(tmp), base = number_of_colors);
            end
             
            # what is the next state value?
            frame[time,i] = rulemodel.rule[index];
        end

        # set the frame -
        frames[time] = frame;
    end
    
    # return
    return frames;
end

function _simulate(algorithm::WolframStochasticSimulation, rulemodel::MyOneDimensionalElementaryWolframRuleModel, initial::Array{Int64,1}; 
    steps::Int64 = 240, maxnumberofmoves::Union{Int64, Nothing} = nothing, 
    parameters::Union{Nothing, Dict{Int, Float64}} = nothing,
    cooldownlength::Int64 = 0)::Dict{Int64, Array{Int64,2}}

    # get stuff from model
    radius = rulemodel.radius; # how many cells am I looking at?
    number_of_colors = rulemodel.number_of_colors; # how many colors (states) can each cell have?
    width = length(initial); # how many cells are there?
    q = Queue{Int64}(); # which cells will update?

    # initialize -
    frames = Dict{Int64, Array{Int64,2}}();
    frame = Array{Int64,2}(undef, steps, width) |> X -> fill!(X, 0);

    # cooldown -
    cooldown = Dict{Int64, Int64}(); # cooldown for each cell
    foreach(i -> cooldown[i] = 0, 1:width); # initialize cooldown for each cell

    # set the initial state -
    foreach(i -> frame[1,i] = initial[i], 1:width);    
    frames[1] = frame; # set the initial frame

    # main loop -
    for time ∈ 2:steps

        # create the next frame -
        frame = copy(frames[time-1]);
        tmp = Array{Int64,1}(undef, radius);

        # generate priority of movement -
        for i ∈ 1:width
            pᵢ = parameters === nothing ? 1.0 : parameters[i]; # probability of movement of cell i
            if (cooldown[i] == 0 && rand() < pᵢ && i ∉ collect(q)) # if the cell is not cooling down
                enqueue!(q, i); # add to queue
            end
        end
      
        # which cell moves first?
        movecount = 0;
        while (isempty(q) == false && (maxnumberofmoves === nothing || movecount ≤ maxnumberofmoves))
            i = dequeue!(q); # a random cell goes -

            # ok, index i is going to make a move -
            if (cooldownlength > 0)
                cooldown[i] = cooldownlength; # set cooldown
            end

            index = nothing;
            if (i == 1)
                
                tmp[1] = frame[time-1, width];      # left
                tmp[2] = frame[time-1, i];          # center
                tmp[3] = frame[time-1, i + 1];      # right

                # compute the index (this is binary, so we need to compute from left to right)
                index = parse(Int, join(tmp), base = number_of_colors);
            elseif (i == width)
                    
                tmp[1] = frame[time-1, i - 1];      # left
                tmp[2] = frame[time-1, i];          # center
                tmp[3] = frame[time-1, 1];          # right
    
                # compute the index (this is binary, so we need to compute from left to right)
                index = parse(Int, join(tmp), base = number_of_colors);
            else
                
                tmp[1] = frame[time-1, i - 1];      # left
                tmp[2] = frame[time-1, i];          # center
                tmp[3] = frame[time-1, i + 1];      # right

                # compute the index (this is binary, so we need to compute from left to right)
                index = parse(Int, join(tmp), base = number_of_colors);
            end
             
            # what is the next state value?
            frame[time,i] = rulemodel.rule[index];

            # update move count -
            movecount += 1;
        end

        empty!(q); # clear the queue

        # tick cooldown
        foreach(i -> cooldown[i] = max(0, cooldown[i] - 1), 1:width);

        # set the frame -
        frames[time] = frame;
    end
    
    # return
    return frames;
end
# -- PRIVATE API ABOVE HERE ------------------------------------------------------------------------ #


# -- PUBLIC API BELOW HERE ------------------------------------------------------------------------ #
"""
    function simulate(rulemodel::MyOneDimensionalElementaryWolframRuleModel, initial::Array{Int64,1};
        steps::Int64 = 24, maxnumberofmoves::Union{Int64, Nothing} = nothing, 
        algorithm::AbstractWolframSimulationAlgorithm)) -> Dict{Int64, Array{Int64,2}}

The simulate function runs a Wolfram simulation based on the provided rule model and initial state.

### Arguments
- `rulemodel::MyOneDimensionalElementaryWolframRuleModel`: The rule model to use for the simulation.
- `initial::Array{Int64,1}`: The initial state of the simulation.
- `steps::Int64`: The number of steps to simulate.
- `maxnumberofmoves::Union{Int64, Nothing}`: The maximum number of moves to simulate.
- `algorithm::AbstractWolframSimulationAlgorithm`: The algorithm to use for the simulation.

### Returns
- A dictionary mapping step numbers to the state of the simulation at that step.
"""
function simulate(rulemodel::MyOneDimensionalElementaryWolframRuleModel, initial::Array{Int64,1}; 
    steps::Int64 = 24, maxnumberofmoves::Union{Int64, Nothing} = nothing, 
    cooldownlength::Int64 = 0, parameters::Union{Nothing, Dict{Int, Float64}} = nothing,
    algorithm::AbstractWolframSimulationAlgorithm)::Dict{Int64, Array{Int64,2}}

    return _simulate(algorithm, rulemodel, initial; steps=steps, 
        maxnumberofmoves=maxnumberofmoves, cooldownlength=cooldownlength, parameters=parameters);
end
# -- PUBLIC API ABOVE HERE ------------------------------------------------------------------------ #