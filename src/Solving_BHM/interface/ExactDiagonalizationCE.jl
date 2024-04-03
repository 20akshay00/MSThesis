using Combinatorics, Plots, LaTeXStrings
using SparseArrays, KrylovKit, LinearAlgebra

hilbert_dim(N, L) = binomial(N + L - 1, L - 1)

function generate_basis(N, L)
    if L > 1
        basis = zeros(Int16, (hilbert_dim(N, L), L))
        j = 1
    
        for n in 0:N
            d = hilbert_dim(n, L - 1)
            basis[j:(j + d - 1), 1] .= (N - n)
            basis[j:(j + d - 1), 2:end] = generate_basis(n, L - 1)
            j += d
        end
        
    else
        basis = [N]
    end

    return basis
end

function hop(state, i, j)
    res_state = copy(state)
    res_state[i] += 1
    res_state[j] -= 1
    return res_state
end

# tags to label the basis states
tag(state) = sum(sqrt.(100 .* (1:length(state)) .+ 3) .* state)
unzip(arr) = map(x -> getfield.(arr, x), fieldnames(eltype(arr)))

###### Operator matrices ######

function generate_kinetic(t, basis)
    D, L = size(basis)
    basis_tags = tag.(eachslice(basis, dims = 1)) # generate tags for each basis
    inds, basis_tags = sortperm(basis_tags), sort(basis_tags) # sorting tag-list for efficient searching

    H_kin = Dict{Tuple{Int64, Int64}, Float64}() # store sparse values; (index) -> val

    for v in 1:D # iterate through basis vectors
        state = basis[v, :] # get vth basis state 

        for j in 1:L # iterate through states (hopping) 
            if state[j] > 0 
                i = mod1(j + 1, L) # Periodic BC
                u = inds[searchsortedfirst(basis_tags, tag(hop(state, i, j)))]
                H_kin[(u, v)] = get(H_kin, (u, v), 0.) - t * ((state[i] + 1) * state[j]) ^ 0.5
            
                i = mod1(j - 1, L) # Periodic BC
                u = inds[searchsortedfirst(basis_tags, tag(hop(state, i, j)))]
                H_kin[(u, v)] = get(H_kin, (u, v), 0.) - t * ((state[i] + 1) * state[j]) ^ 0.5
            end
        end
    end

    return sparse(unzip(keys(H_kin))..., collect(values(H_kin)))
end

function generate_interaction(U, basis)
    @assert(length(U) == size(basis)[2] || length(U) == 1)
    return spdiagm(0.5 .* U .* sum(basis .* (basis .- 1), dims = 2) |> vec)
end

function generate_number(mu, basis) 
    @assert(length(mu) == size(basis)[2] || length(mu) == 1)
    return spdiagm(mu .* (sum(basis, dims = 2) |> vec))
end

###### Expectation values ######

expectation(state, op) = (state' * op * state)[1]

# Single Particle Density Matrix
function SPDM(ground_state, basis)    
    D, L = size(basis)
    basis_tags = tag.(eachslice(basis, dims = 1)) # generate tags for each basis
    inds, basis_tags = sortperm(basis_tags), sort(basis_tags) # sorting tag-list to reduce complexity

    SPDM = zeros((L, L))

    for i in 1:L, j in 1:L 
        new_state = zeros(D) # ground state after action of hopping term

        for v in 1:D # iterate over components of the ground state
            state = basis[v, :]
            if state[j] > 0 
                u = inds[searchsortedfirst(basis_tags, tag(hop(state, i, j)))]

                # co-efficient from the hopping term
                if i != j 
                    new_state[u] += ((state[i] + 1) * state[j]) ^ 0.5 * ground_state[v] 
                else
                    new_state[u] += state[i] * ground_state[v] 
                end
            end
        end

        SPDM[i, j] = sum(ground_state .* new_state)
    end

    return SPDM
end

function navg(ground_state, basis, i)
    return sum(basis[:, i] .* (ground_state .^ 2))
end

function nvar(ground_state, basis, i)
    n_exp = sum(basis[:, i] .* (ground_state .^ 2))
    nsq_exp = sum((basis[:, i] .^ 2).* (ground_state .^ 2))

    return n_exp, nsq_exp, sqrt(nsq_exp - n_exp ^ 2)
end