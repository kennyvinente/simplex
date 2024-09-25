using LinearAlgebra, Combinatorics

# Auxiliary function to check if a vector is non-negative
function is_nonnegative(x::Vector)
    return all(x .>= 0)
end

# Function to find all Basic Feasible Solutions
function search_BFS(c::Vector, A::Matrix, b::Vector)
    m, n = size(A)
    
    # Pre-condition: Matrix A must have full rank
    @assert rank(A) == m "Matrix A does not have full rank."

    opt_x = zeros(n)  # Optimal solution
    obj = Inf         # Optimal objective value

    # Iterate over all combinations of columns of A that form a basis
    for b_idx in combinations(1:n, m)
        try
            # Construct the basis matrix B and the cost vector c_B
            B = A[:, b_idx]
            c_B = c[b_idx]
            
            # Compute the basic solution associated with the basis
            x_B = inv(B) * b

            # Check if the basic solution is non-negative
            if is_nonnegative(x_B)
                z = dot(c_B, x_B)
                if z < obj
                    obj = z
                    opt_x = zeros(n)
                    opt_x[b_idx] = x_B
                end
            end

            # Print debug information (can be removed in production)
            println("Basis:", b_idx)
            println("\t x_B = ", round.(x_B, digits=5))
            println("\t Non-negative? ", is_nonnegative(x_B))
            if is_nonnegative(x_B)
                println("\t Obj = ", dot(c_B, x_B))
            end

        catch e
            # Handle singular matrix exceptions
            if isa(e, SingularException) || isa(e, LAPACKException)
                println("SingularException: Basis ", b_idx, " is non-invertible. Skipping this basis.")
            else
                rethrow(e)
            end
        end
    end

    return opt_x, obj
end
