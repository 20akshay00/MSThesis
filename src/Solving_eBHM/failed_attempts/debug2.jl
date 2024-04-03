using Plots, LinearAlgebra, LaTeXStrings, DataStructures

abstract type BoseHubbardModel end

struct MeanField <: BoseHubbardModel
    n_max :: Int64
    a :: Matrix{Float64}
    n :: Matrix{Float64}

    MeanField(n_max) = new(n_max, diagm(1 => sqrt.(1:n_max)), diagm(0 => 0:n_max))
end

expect(op, state) = (conj.(state') * op * state)[1]

# function get_hamiltonian(model :: MeanField, t, mu, U, V, z, params)
#     ψₐ, ψᵦ, ρₐ, ρᵦ = params
#     H = -mu * model.n + 0.5 * U * model.n * (model.n - I) + 0.5 * V * z * (ρᵦ * model.n - ρₐ * ρᵦ * I) - t * z * ψᵦ * (model.a + model.a') + t * z * ψₐ * ψᵦ * I
#     return H 
# end

function get_hamiltonian(model :: MeanField, t, mu, U, V, z, params)
    ψₐ, ψᵦ, ρₐ, ρᵦ = params
    H = -mu * model.n / (z * t) + (U/(2 * z * t)) * model.n * (model.n - I) + (V/t) * (ρᵦ * model.n - ρₐ * ρᵦ * I) - ψᵦ * (model.a + model.a') + ψₐ * ψᵦ * I
    return H 
end

function get_order_parameter(model, t, mu, U, V, z, init = nothing, depth = 1)
    ground_state((ψₐ, ψᵦ, ρₐ, ρᵦ)) = (eigvecs(get_hamiltonian(model, t, mu, U, V, z, [ψₐ, ψᵦ, ρₐ, ρᵦ]))[:, 1], eigvecs(get_hamiltonian(model, t, mu, U, V, z, [ψᵦ, ψₐ, ρᵦ, ρₐ]))[:, 1])
    
    init = isnothing(init) ? rand(0:100, 4) : init
    params = CircularBuffer{Vector{Float64}}(3)
    push!(params, init)

    tol = 4
    num_iter = 0

    while(true)
        psi_gs = ground_state(params[end])
        params_new = [abs(expect(model.a, psi_gs[1])), abs(expect(model.a, psi_gs[2])), expect(model.n, psi_gs[1]), expect(model.n, psi_gs[2])]
        push!(params, params_new)

        println(params_new, num_iter)

        if((norm((params[end] .- params[end - 1])) <= 1/10^tol)) 
            return round.(params[end], digits = tol), depth
        end
        
        if(num_iter == 500) 
            if abs(params[1][3] - params[1][4]) <= 1/10^(tol - 1)
                if ((abs(params[1][3] - params[3][3]) <= 1/10^(tol - 1)) && (abs(params[1][3] - params[2][3]) >= 1/10^(tol - 1)))
                    res = get_order_parameter(model, t, mu, U, V, z, [params[1][1], params[2][2], params[1][3], params[2][4]], depth + 1)

                elseif abs(params[1][1] .- params[1][2]) <= 1/10^(tol - 1)
                    res = get_order_parameter(model, t, mu, U, V, z, [params[1][1], params[1][2], params[1][3], 0], depth + 1)
                else
                    res = get_order_parameter(model, t, mu, U, V, z, [(params[1][1] .+ params[1][2])/2, (params[1][1] .+ params[1][2])/2, params[1][3], params[1][4]], depth + 1)
                end
            else
                if abs(params[1][1] .- params[1][2]) <= 1/10^(tol - 1)
                    res = get_order_parameter(model, t, mu, U, V, z, [params[1][1], params[1][2], (params[1][3] + params[2][3])/2, (params[1][4] + params[2][4])/2], depth + 1)
                else
                    res = get_order_parameter(model, t, mu, U, V, z, (params[1] .+ params[2]) ./ 2, depth + 1)
                end
            end

            return res
        end

        num_iter += 1
    end

    return nothing
end

# find N for the ground state
# function get_num_particles(model, t, mu, V, z)
#     psi, phi = get_order_parameter(model, t, mu, V, z)
#     gs = eigvecs(get_hamiltonian(model, t, mu, V, z, psi))[:, 1]
    
#     return gs' * model.n * gs
# end

model = MeanField(6)

z, size = 4, 2
# t = range(start = 0, stop = 0.1, length = size)
V = range(start = 4, stop = 16, length = size)
mu = range(start = 0, stop = 20, length = size)
t = 1/4

# t, mu, U, V, z
# get_order_parameter(model, 0.25, 0.6060606060606061, 6.626262626262626, 0.6626262626262627, 4)
