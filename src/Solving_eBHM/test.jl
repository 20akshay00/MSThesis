using Plots, Optim, LinearAlgebra, LaTeXStrings

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

function get_order_parameter(model, t, mu, U, V, z, init = nothing)
    ground_state((ψₐ, ψᵦ, ρₐ, ρᵦ)) = (eigvecs(get_hamiltonian(model, t, mu, U, V, z, [ψₐ, ψᵦ, ρₐ, ρᵦ]))[:, 1], eigvecs(get_hamiltonian(model, t, mu, U, V, z, [ψᵦ, ψₐ, ρᵦ, ρₐ]))[:, 1])
    
    params_new = isnothing(init) ? rand(-1000:1000, 4) : init
    params_old = copy(params_new)
    
    num_iter = 0
    while(true)
        params_old = copy(params_new)
        psi_gs = ground_state(params_new)
        params_new = [abs(expect(model.a, psi_gs[1])), abs(expect(model.a, psi_gs[2])), expect(model.n, psi_gs[1]), expect(model.n, psi_gs[2])]

        println(params_new, num_iter)

        if((norm((params_old .- params_new)./params_old) <= 1e-3)) 
            return params_new
        end
        
        if(num_iter == 2000) 
            return get_order_parameter(model, t, mu, U, V, z)
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
t = 1

# t, mu, U, V, z
get_order_parameter(model, 0.25, 1.02, 4.0, 0.6, 4)
# order_param = zeros((size, size, 4))
# num_particles = zeros((size, size))

# for k1 in 1:size
#     for k2 in 1:size
#         order_param[k2, k1, :] .= abs.(get_order_parameter(model, t, mu[k2], V[k1], z))
#     # num_particles[k2, k1] = abs.(get_num_particles(model, t[k1], mu[k2], V, z))
#     end
# end