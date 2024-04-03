using Combinatorics, SparseArrays, KrylovKit, LinearAlgebra

begin
	function create_lattice_op(n_max, L, op_loc, op_lat)
	    single_site_identity = spdiagm(0 => ones(n_max + 1))
	    for i in 0:(L - 1)
	        tmp = 1
	        for j in 0:(i - 1)
	            tmp = kron(tmp, single_site_identity)
	        end
	        tmp = kron(tmp, op_loc)
	        for j in 0:(L - i - 2)
	            tmp = kron(tmp, single_site_identity)
	        end
	
	        op_lat[i + 1] = tmp
	    end
	end
	
	expect(ket, op) = (ket' * op * ket)[1]
	var(ket, op) = expect(ket, op * op) - (expect(ket, op))^2
	
	function get_ground_state(model, t, mu, U = 1., V = 0.)
		mu = model.isCanonical ? 0 : mu
		N = model.isCanonical ? model.N[1][model.keep_sites, :][:, model.keep_sites] : model.N[1]
		
		E, psi = eigsolve(-t .* model.Hk .+ 0.5 * U .* model.Hu .- mu .* model.Hn .+ V .* model.Hv, 1, :SR)

		return psi[1]
	end
end

# ╔═╡ cb0dd484-1054-4cc9-9a68-5651332c498e
begin
	struct BHM
	    L :: Int64
	    n_max :: Int64
	    isCanonical :: Bool
		isPeriodic :: Bool
	    dim :: Int64
	    M :: Float64 
		keep_sites :: Vector{Bool}
		
	    # Local operators
	    a :: SparseMatrixCSC{Float64, Int64}
	    a_dag :: Adjoint{Float64, SparseMatrixCSC{Float64, Int64}}
	    n :: SparseMatrixCSC{Float64, Int64}
	
	    # lattice operators 
	    A :: Vector{SparseMatrixCSC{Float64, Int64}}
	    A_dag :: Vector{SparseMatrixCSC{Float64, Int64}}
	    N :: Vector{SparseMatrixCSC{Float64, Int64}}
	    N_tot :: SparseMatrixCSC{Float64, Int64}

		# hamiltonian pieces
		Hk :: SparseMatrixCSC{Float64, Int64}
		Hu :: SparseMatrixCSC{Float64, Int64}
		Hv :: SparseMatrixCSC{Float64, Int64}
		Hn :: SparseMatrixCSC{Float64, Int64}
	end
	
	function BHM(L, n_max, isCanonical, isPeriodic, M = 0)
		# # general stuff
	 #    dim = (n_max + 1)^L

		# construct local operators
	    a = spdiagm(1 => sqrt.(1:n_max))
	    a_dag = a'
	    n = a_dag * a

		# construct lattice operators
	    A = Vector{SparseMatrixCSC{Float64, Int64}}(undef, L)
	    A_dag = Vector{SparseMatrixCSC{Float64, Int64}}(undef, L)
		N = Vector{SparseMatrixCSC{Float64, Int64}}(undef, L)
		
	    create_lattice_op(n_max, L, a, A)
	    create_lattice_op(n_max, L, a_dag, A_dag)
	    create_lattice_op(n_max, L, n, N)

		N_tot = reduce(+, N)
		
		# construct pieces of the hamiltonian
		dim = size(A[1])[1]
		Hk, Hu, Hv = spzeros(dim, dim), spzeros(dim, dim), spzeros(dim, dim)
		Hn = copy(N_tot)
		l_max = isPeriodic ? L : (L - 1)

		for i in 1:l_max
			j = mod1(i + 1, L)
			Hk = Hk + (A_dag[i] * A[j] + A_dag[j] * A[i])
			Hv = Hv + (N[i] * N[j])
		end

		for i in 1:L
			Hu =  Hu + N[i] * (N[i] - I)
		end

		# if canonical, find restricted indices
		keep_sites = []
		
		if isCanonical
			m = LinearAlgebra.diag(N_tot)
			keep_sites = (m .> (M - 0.5)) .&& (m .< (M + 0.5))

			Hk = Hk[keep_sites, :][:, keep_sites]
			Hu = Hu[keep_sites, :][:, keep_sites]
			Hn = Hn[keep_sites, :][:, keep_sites]
		end
		
		# construct the model
	    return BHM(L, n_max, isCanonical, isPeriodic, dim, M, keep_sites, a, a_dag, n, A, A_dag, N, N_tot, Hk, Hu, Hv, Hn)
	end
end

