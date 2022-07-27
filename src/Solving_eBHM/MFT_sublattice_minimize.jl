using Plots, Optim, LinearAlgebra, LaTeXStrings, FiniteDiff

abstract type BoseHubbardModel end

struct MeanField <: BoseHubbardModel
    n_max :: Int64
    a :: Matrix{Float64}
    n :: Matrix{Float64}

    MeanField(n_max) = new(n_max, diagm(1 => sqrt.(1:n_max)), diagm(0 => 0:n_max))
end

function get_hamiltonian(model :: MeanField, t, mu, U, V, z, params)
    ψₐ, ψᵦ, ρₐ, ρᵦ = params
    H = -mu * model.n + 0.5 * U * model.n * (model.n - I) + 0.5 * V * z * (ρᵦ * model.n - ρₐ * ρᵦ * I) - t * z * ψᵦ * (model.a + model.a') + t * z * ψₐ * ψᵦ * I
    return H 
end

function get_order_parameter(model, t, mu, U, V, z)
    E_gs((ψₐ, ψᵦ, ρₐ, ρᵦ)) = eigvals(get_hamiltonian(model, t, mu, U, V, z, [ψₐ, ψᵦ, ρₐ, ρᵦ]))[1] + eigvals(get_hamiltonian(model, t, mu, U, V, z, [ψᵦ, ψₐ, ρᵦ, ρₐ]))[1]
    return Optim.minimizer(optimize((params) -> sum(FiniteDiff.finite_difference_gradient(E_gs, params) .^ 2), zeros(4)))
end

# find N for the ground state
# function get_num_particles(model, t, mu, V, z)
#     psi, phi = get_order_parameter(model, t, mu, V, z)
#     gs = eigvecs(get_hamiltonian(model, t, mu, V, z, psi))[:, 1]
    
#     return gs' * model.n * gs
# end

function loop(size)
    model = MeanField(6)

    z = 4
    # t = range(start = 0, stop = 0.1, length = size)
    t = range(start = 0., stop = 0.6, length = size)
    mu = range(start = 1, stop = 5, length = size)
    V, U = 1., 0.

    order_param = zeros((size, size, 4))
    num_particles = zeros((size, size))

    Threads.@threads for k1 in 1:size
        for k2 in 1:size
            order_param[k2, k1, :] .= abs.(get_order_parameter(model, t[k1], mu[k2], U, V, z))
        # num_particles[k2, k1] = abs.(get_num_particles(model, t[k1], mu[k2], V, z))
        end
    end

    return t, mu, order_param
end

function phase_diag(t, mu, order_param)
    theme(:lime)
    gr()
    p = []
    p_names = ["ψₐ", "ψᵦ", "ρₐ", "ρᵦ"]

    for i in 1:4
        p_temp = heatmap(t, mu, order_param[:, :, i])
        plot!(
            ylabel = "μ (chemical potential)",
            xlabel = "t (hopping parameter)",
            framestyle = :box, 
            title = p_names[i])

        push!(p, p_temp)
    end

    plot(p...)
end
