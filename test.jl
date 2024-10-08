
## ex. from book Julia Programming for Operations Research
c = [-3; -2; -1; -5; 0; 0; 0]
A = [7 3 4 1 1 0 0 ;
     2 1 1 5 0 1 0 ;
     1 4 5 2 0 0 1 ]
b = [7; 3; 8]

## ex. from prof. Fabio D'Andreagiovanni
c = [-3; -2; -3; 0; 0; 0]
A = [2 1 1 1 0 0 ;
     1 2 3 0 1 0 ;
     2 2 1 0 0 1 ]
b = [2; 5; 6]

## ex. from prof. David P. Williamson
c = [-1; 2; -1; 0; 0; 0; 0]
A = [1 0 0 1 0 0 0 ;
     0 1 0 0 1 0 0 ;
     1 1 0 0 0 1 0; 
     -1 0 2 0 0 0 1]
b = [4; 4; 6; 4]


c = Array{Float64}(c)
A = Array{Float64}(A)
b = Array{Float64}(b)


using JuMP, HiGHS, LinearAlgebra
m = Model(HiGHS.Optimizer)
@variable(m, x[1:length(c)] >=0 )
@objective(m, Min, dot(c, x))
@constraint(m, A*x .== b)
@time JuMP.optimize!(m)
obj0 = JuMP.objective_value(m)
opt_x0 = JuMP.value.(x)

include("search_bfs.jl")
@time opt_x1, obj1 = search_BFS(c, A, b)

include("simplex_method.jl")
using Main.SimplexMethod
@time opt_x2, obj2 = simplex_method(c, A, b)


println()
println("obj by JuMP                  = ", obj0)
println("obj by search_extreme_points = ", obj1)
println("obj by simplex_method        = ", obj2)
println()
println("x*  by JuMP                  = ", opt_x0)
println("x*  by search_extreme_points = ", opt_x1)
println("x*  by simplex_method        = ", opt_x2)
println()