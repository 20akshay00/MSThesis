
begin
	using FixedPoint, Combinatorics
	using LinearAlgebra, SparseArrays, KrylovKit
end

# ╔═╡ 951cb5d0-5ac4-11ed-36b5-a98ab7fda6cb
begin
	abstract type BoseHubbardModel end
	
	struct MeanField <: BoseHubbardModel
	    n_max :: Int64
	    a :: Vector{SparseMatrixCSC{Float64, Int64}}
		n :: Vector{SparseMatrixCSC{Float64, Int64}}

		unit_cell::Matrix{Int64}
		num_sites :: Int64
		z :: Int64
		
	    function MeanField(n_max, unit_cell) 
			a_local = spdiagm(1 => sqrt.(1:n_max))
			n_local = spdiagm(0 => 0:n_max)
			id = sparse(I, n_max + 1, n_max + 1)

			@assert size(unit_cell)[1] == size(unit_cell)[2]
			num_sites = size(unit_cell)[1]

			a = unique(permutations(vcat([a_local], repeat([id], num_sites - 1)))) .|> (arr) -> kron(arr..., 1)
			n = unique(permutations(vcat([n_local], repeat([id], num_sites - 1)))) .|> (arr) -> kron(arr..., 1)
			
			return new(n_max, a, n, unit_cell, num_sites, sum(unit_cell) ÷ num_sites)
		end
	end
	
	expect(op, state) = (conj.(state') * op * state)[1]
	
	function get_hamiltonian(model :: MeanField, t, mu, U, V, z, params)
		psiU, rhoU = eachcol(model.unit_cell * params)
		psi, rho = eachcol(params)

		H = reduce(1:model.num_sites, init = zero(model.a[1])) do M, i
			M - mu * model.n[i] + (U/2) * model.n[i] * (model.n[i] - I) + V * rhoU[i] * (model.n[i] - 0.5 * rho[i] * I) - t * psiU[i] * (model.a[i] + model.a[i]') + t * psi[i] * psiU[i] * I
		end
		
	    return H 
	end
	
	function get_order_parameter(model, t, mu, U, V, z, atol = 1e-4)
		
	    function f(params)
			psi = eigsolve(get_hamiltonian(model, t, mu, U, V, z, reshape(params, :, 2)), 1, :SR)[2][1]

			return abs.(expect.([model.a..., model.n...], [psi])) 
		end
		
		return afps(f, rand(2 * model.num_sites), tol = atol)[:x]
	end
end
