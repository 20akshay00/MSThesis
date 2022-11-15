using Random, StatsBase, OffsetArrays

site(x, y, Lx, Ly) = y * Lx + x

function init_square_lattice(Lx, Ly)
    n_sites = Lx * Ly
    spins = 2 .* mod.(shuffle(1:n_sites), 2) .- 1
    op_string = OffsetVector(zeros(Int64, 10), 0:9)

    bonds = []

    for x0 in 0:(Lx - 1)
        for y0 in 0:(Ly - 1)
            s0 = site(x0, y0, Lx, Ly)
            s1 = site(mod(x0 + 1, Lx), y0, Lx, Ly) # bond to the right
            push!(bonds, [s0, s1])

            s2 = site(x0, mod(y0 + 1, Ly), Lx, Ly) # bond to the top
            push!(bonds, [s0, s2])
        end
    end

    return spins, op_string, vcat(bonds'...)
end

function diagonal_update!(spins, op_string, bonds, beta)
    n_bonds = first(size(bonds))
    M = first(size(op_string))

    # number of non-identity operators
    n = count(==(0), op_string)

    # ratio of acceptance probabilities for insert/remove 
    prob_ratio = 0.5 * beta * n_bonds
    for p in 0:(M - 1)
        op = op_string[p]
        if op == 0 # if identity, propose an insertion
            b = rand(1:n_bonds) # select bond
            if (spins[bonds[b, 1]] - spins[bonds[b, 2]]) == 0 # check if anti-parallel
                p_insert = prob_ratio / (M - n)
                if rand() < p_insert # metropolis sampling
                    # insert diagonal operator
                    op_string[p] = 2 * b
                    n += 1
                end
            end

        elseif mod(op, 2) == 0 # if diagonal, propose a removal
            p_remove = (M - n + 1) / prob_ratio
            if rand() < p_remove 
                # remove diagonal operator
                op_string[p] = 0
                n -= 1
            end

        else # if off-diagonal, get propagated spin state 
            b = op ÷ 2
            spins[bonds[b, 1]] == -spins[bonds[b, 1]]
            spins[bonds[b, 2]] == -spins[bonds[b, 2]]
        end
    end

    return n
end

function loop_update!(spins, op_string, bonds)
    vertex_list, first_vertex_at_site = create_linked_vertex_list(spins, op_string, bonds)
    flip_loops!(spins, op_string, vertex_list, first_vertex_at_site)
end

function create_linked_vertex_list(spins, op_string, bonds)
    n_sites = first(size(spins))
    M = first(size(op_string))

    vertex_list = OffsetVector(zeros(Int64, 4 * M), 0:(4 * M - 1))

    first_vertex_at_site = -1 * ones(Int64, n_sites)
    last_vertex_at_site = -1 * ones(Int64, n_sites)

    for p in 0:(M - 1)
        v0 = 4 * p # left incoming leg
        v1 = v0 + 1 # right incoming leg
        op = op_string[p]

        if op == 0 # if identity
            vertex_list[v0:(v0 + 3)] .= -2 
        else
            b = op ÷ 2
            s0 = bonds[b, 1]
            s1 = bonds[b, 2]
            v2 = last_vertex_at_site[s0]
            v3 = last_vertex_at_site[s1]
            
            if v2 == -1 # no operator encountered before
                first_vertex_at_site[s0] = v0 
            else
                vertex_list[v2] = v0
                vertex_list[v0] = v2
            end

            if v3 == -1 # no operator encountered before
                first_vertex_at_site[s1] = v1 
            else
                vertex_list[v3] = v1
                vertex_list[v1] = v3
            end
            
            last_vertex_at_site[s0] = v0 + 2 # left outgoing leg
            last_vertex_at_site[s1] = v0 + 3 # right outgoing leg
        end
    end

    # connect vertices between top and bottom
    for s0 in 1:n_sites
        v0 = first_vertex_at_site[s0]
        if v0 != -1 # there is an operator acting on the site
            v1 = last_vertex_at_site[s0]
            vertex_list[v1] = v0
            vertex_list[v0] = v1
        end
    end

    return vertex_list, first_vertex_at_site
end

function flip_loops!(spins, op_string, vertex_list, first_vertex_at_site)
    n_sites = first(size(spins))
    M = first(size(op_string))
    
    for v0 in 0:2:(4 * M - 1)
        if vertex_list[v0] < 0
            continue
        end

        v1 = v0 

        if rand() < 0.5
            while true
                op = v1 ÷ 4
                op_string[op] = xor(op_string[op], 1)
                vertex_list[v1] = -1
                v2 = xor(v1, 1)
                v1 = vertex_list[v2]
                vertex_list[v2] = -1

                println("v1 = $(v1)")
                if v1 == v0 break end
            end
        else 
            while true
                vertex_list[v1] = -2
                v2 = xor(v1, 1)
                v1 = vertex_list[v2]
                vertex_list[v2] = -2
                if v1 == v0 break end
            end
        end

        for s0 in 1:n_sites
            if first_vertex_at_site[s0] == -1
                if rand() < 0.5
                    spins[s0] = -spins[s0]
                end
            else
                if vertex_list[first_vertex_at_site[s0]] == -1
                    spins[s0] = -spins[s0]
                end
            end
        end
    end
end


function thermalize!(spins, op_string, bonds, beta, niter)
    if beta == 0
        throw(DomainError(beta, "Temperature must be finite."))
    end

    for _ in 1:niter
        n = diagonal_update!(spins, op_string, bonds, beta)
        loop_update!(spins, op_string, bonds)
        M_old = length(op_string)
        M_new = n + n ÷ 3

        if M_new > M_old
            op_string = OffsetVector(vcat(parent(op_string), zeros(Int64, M_new - M_old)), 0:(M_new - 1))
        end   
    end

    return op_string
end

function measure!(spins, op_string, bonds, beta, niter)
    ns = Vector{Float64}(undef, niter)
    for i in 1:niter
        n = diagonal_update!(spins, op_string, bonds, beta)
        loop_update!(spins, op_string, bonds)
        ns[i] = n      
    end

    return ns
end

function main(Lx, Ly, beta_list=[1.], n_updates_measure=10000, n_bins=10)
    spins, op_string, bonds = init_square_lattice(Lx, Ly)
    n_sites, n_bonds = length(spins), length(bonds)
    Es_Errs = []

    for beta in beta_list
        println("beta = $(round(beta, digits = 3))")
        op_string = thermalize!(spins, op_string, bonds, beta, n_updates_measure ÷ 10)
        Es = Vector{Float64}(undef, n_bins)

        for i in 1:n_bins
            ns = measure!(spins, op_string, bonds, beta, n_updates_measure)
            E = (mean(ns)/beta + 0.25 * n_bonds) / n_sites
            Es[i] = E         
        end 

        E, Eerr = mean(Es), std(Es)/sqrt(n_bins)
        push!(Es_Eerrs, (E, Eerr))
    end

    return Es_Errs
end