function _solve(A::AbstractMatrix{T}, b::AbstractVector{T}, xₒ::AbstractVector{T}, algorithm::JacobiMethod;
    ϵ::Float64 = 0.01, maxiterations::Int64 = 100, ω::Float64 = 0.01) where T <: Number

    # initialize -
    (number_of_rows, number_of_columns) = size(A);
    is_ok_to_terminate = false;
    k = 0; # initialize iteration counter to 0
    archive = Dict{Int, Array{Float64,1}}(); # we store every iteration in this dictionary
   
    x = copy(xₒ); # initialize solution vector x with the initial guess -
    archive[0] = xₒ; # store the initial guess in the archive
    k += 1;  # update the iteration counter -

    # iterate -
    while (is_ok_to_terminate == false)

        # grab the current solution vector -
        x = copy(archive[k-1]); 
        y = zeros(number_of_rows); # this will hold the updated solution vector at k+1
 
        # MyJacobiMethod algorithm -
        for i in 1:number_of_rows
            σ = 0.0;
            for j ∈ 1:number_of_columns
                if (i != j)
                    σ += A[i,j]*x[j];
                end
            end
            y[i] = (b[i] - σ)/A[i,i];
        end
        
        # check the error condition -
        (maximum(A*y - b) < ϵ || k > maxiterations) ? is_ok_to_terminate = true : is_ok_to_terminate = false;

        # update the archive -
        archive[k] = y; # grab a copy of the solution vector at k+1
        k += 1;  # update the iteration counter -
    end

    # return archive -
    return archive;
end

function _solve(A::AbstractMatrix{T}, b::AbstractVector{T}, xₒ::AbstractVector{T}, algorithm::GaussSeidelMethod;
    ϵ::Float64 = 0.01, maxiterations::Int64 = 100, ω::Float64 = 0.01) where T <: Number

    # initialize -
    (number_of_rows, number_of_columns) = size(A);
    is_ok_to_terminate = false;
    k = 0; # initialize iteration counter to 0
    archive = Dict{Int, Array{Float64,1}}(); # we store every iteration in this dictionary
   
    x = copy(xₒ); # initialize solution vector x with the initial guess -
    archive[0] = xₒ; # store the initial guess in the archive
    k += 1;  # update the iteration counter -

    # iterate -
    while (is_ok_to_terminate == false)

        x = copy(archive[k-1]); # grab the current solution vector, create a copy so we don't overwrite the data in the archive
 
        # MyGaussSeidelMethod algorithm implementation. Looks alot like the Jacobi method, but we update the solution vector in place
        for i in 1:number_of_rows
            σ = 0.0;
            for j ∈ 1:number_of_columns
                if (i != j)
                    σ += A[i,j]*x[j];
                end
            end
            x[i] = (b[i] - σ)/A[i,i]; # update the solution vector x in place! 
        end
        
        # check the error condition -
        (maximum(A*x - b) < ϵ || k > maxiterations) ? is_ok_to_terminate = true : is_ok_to_terminate = false;

        # update the archive -
        archive[k] = x; # grab a copy of the solution vector at k+1 (do I need to make a copy here?)
        k += 1;  # update the iteration counter -
    end

    # return archive -
    return archive;
end

function _solve(A::AbstractMatrix{T}, b::AbstractVector{T}, xₒ::AbstractVector{T}, algorithm::SuccessiveOverRelaxationMethod;
    ϵ::Float64 = 0.01, maxiterations::Int64 = 100, ω::Float64 = 0.01) where T <: Number

    # initialize -
    (number_of_rows, number_of_columns) = size(A);
    is_ok_to_terminate = false;
    k = 0; # initialize iteration counter to 0
    archive = Dict{Int, Array{Float64,1}}(); # we store every iteration in this dictionary
   
    x = copy(xₒ); # initialize solution vector x with the initial guess -
    archive[0] = xₒ; # store the initial guess in the archive
    k += 1;  # update the iteration counter -

    # Grok: Impl me -
    while (is_ok_to_terminate == false)

        x = copy(archive[k-1]); # grab the current solution vector, create a copy so we don't overwrite the data in the archive

        # MySuccessiveOverRelaxationMethod algorithm implementation. Similar to Gauss-Seidel but with relaxation factor ω
        for i in 1:number_of_rows
            σ = 0.0;
            for j ∈ 1:number_of_columns
                if (i != j)
                    σ += A[i,j]*x[j];
                end
            end
            x_new = (b[i] - σ)/A[i,i];
            x[i] = (1 - ω)*x[i] + ω*x_new; # apply relaxation - uses updated values within iteration
        end

        # check the error condition -
        (maximum(A*x - b) < ϵ || k > maxiterations) ? is_ok_to_terminate = true : is_ok_to_terminate = false;

        # update the archive -
        archive[k] = x; # grab a copy of the solution vector at k+1 (do I need to make a copy here?)
        k += 1;  # update the iteration counter -
    end
    
    # Successive Over-Relaxation method
    return archive;
end

# -- PUBLIC METHODS BELOW HERE ---------------------------------------------------------------------------------------------------------------------------------------- #

"""
    solve(A::AbstractMatrix{T}, b::AbstractVector{T}, xₒ::AbstractVector{T}; 
    algorithm::AbstractLinearSolverAlgorithm = JacobiMethod(), ϵ::Float64 = 0.01, maxiterations::Int64 = 100) where T <: Number

The `solve` function solves the linear system of equations `Ax = b` using the specified algorithm. 
The function returns the solution vector `x` for each iteration of an iterative method. 

### Arguments
- `A::AbstractMatrix{T}`: The system matrix `A` in the linear system of equations `Ax = b`.
- `b::AbstractVector{T}`: The right-hand side vector `b` in the linear system of equations `Ax = b`.
- `xₒ::AbstractVector{T}`: The initial guess for the solution vector `x`.
- `algorithm::AbstractLinearSolverAlgorithm`: The algorithm to use to solve the linear system of equations. The default algorithm is `JacobiMethod()`.
- `ϵ::Float64`: The error tolerance for the iterative method. The default value is `0.01`.
- `maxiterations::Int64`: The maximum number of iterations for the iterative method. The default value is `100`.
- `ω::Float64`: The relaxation factor for the Successive Over-Relaxation method. The default value is `0.01`. This parameter is only used if the `SuccessiveOverRelaxationMethod` algorithm is selected.

### Returns
- `d::Dict{Int,Array{T,1}}`: The solution vector `x` for each iteration of an iterative method. The keys of the dictionary are the iteration numbers, and the values are the solution vectors at each iteration.
"""
function solve(A::AbstractMatrix{T}, b::AbstractVector{T}, xₒ::AbstractVector{T};
    algorithm::AbstractLinearSolverAlgorithm = JacobiMethod(), 
    ϵ::Float64 = 0.01, maxiterations::Int64 = 100, ω::Float64 = 0.01) where T <: Number
    
    
    # return -
    return _solve(A, b, xₒ, algorithm, ϵ = ϵ, maxiterations = maxiterations, ω = ω);
end

# -- PUBLIC METHODS ABOVE HERE ---------------------------------------------------------------------------------------------------------------------------------------- #