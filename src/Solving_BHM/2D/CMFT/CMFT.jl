using Combinatorics, SparseArrays, KrylovKit, LinearAlgebra, JLD

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

	function get_order_parameter(model, t, mu, U = 1)
		E_gs(psi) = eigsolve(-t .* model.Hk .+ U .* model.Hu .- mu .* model.Hn + t .* (-psi[1] .* model.Hmf + 4 * psi[1]^2 * I), 1, :SR)[1][1] |> real

		return Optim.minimizer(optimize(E_gs, zeros(1)))[1]
	end
end

begin
	struct BHM
	    L :: Int64
	    n_max :: Int64
	    isCanonical :: Bool
		isPeriodic :: Bool
	    dim :: Int64
	    M :: Float64 
		keep_sites :: Vector{Bool}
		
	    N :: Vector{SparseMatrixCSC{Float64, Int64}}

		# hamiltonian pieces
		Hk :: SparseMatrixCSC{Float64, Int64}
		Hu :: SparseMatrixCSC{Float64, Int64}
		Hn :: SparseMatrixCSC{Float64, Int64}
        Hmf :: SparseMatrixCSC{Float64, Int64}
	end

	to1D((i, j), L) = L * (i - 1) + j
	to2D(i, L) = ((i-1)÷L + 1, mod1(i, L))

	function BHM(L, n_max, isCanonical, isPeriodic, M = 0)
		# # general stuff
	 #    dim = (n_max + 1)^L

		# construct local operators
	    a = spdiagm(1 => sqrt.(1:n_max))
	    a_dag = copy(a')
	    n = a_dag * a

		# construct lattice operators
	    A = Vector{SparseMatrixCSC{Float64, Int64}}(undef, L * L)
	    A_dag = Vector{SparseMatrixCSC{Float64, Int64}}(undef, L * L )
		N = Vector{SparseMatrixCSC{Float64, Int64}}(undef, L * L)
		
	    create_lattice_op(n_max, L * L, a, A)
	    create_lattice_op(n_max, L * L, a_dag, A_dag)
	    create_lattice_op(n_max, L * L, n, N)

		N_tot = reduce(+, N)
		
		# construct pieces of the hamiltonian
		dim = size(A[1])[1]
		Hk, Hu = spzeros(dim, dim), spzeros(dim, dim)
		Hn = copy(N_tot)
		l_max = isPeriodic ? L : (L - 1)

		for ind_1d in 1:L^2
			i, j = to2D(ind_1d, L)
			nbs = to1D.([(mod1(i + 1, L), j), (i, mod1(j + 1, L))], L)
			for nb in nbs
				Hk += (A_dag[ind_1d] * A[nb] + A_dag[nb] * A[ind_1d])
			end
		end

		for ind_1d in 1:L^2
			Hu =  Hu + 0.5 * N[ind_1d] * (N[ind_1d] - I)
		end

		# if canonical, find restricted indices
		keep_sites = []
		
		if isCanonical
			m = LinearAlgebra.diag(N_tot)
			keep_sites = (m .> (M - 0.5)) .& (m .< (M + 0.5))

			Hk = Hk[keep_sites, :][:, keep_sites]
			Hu = Hu[keep_sites, :][:, keep_sites]
			Hn = Hn[keep_sites, :][:, keep_sites]
		end
		
        inds = to2D.([(1, 1), (1, L), (L, 1), (L, L)], L) 
        Hmf = reduce(inds, init = zero(Hk)) do M, i 
            M + A[i] + A_dag[i]
        end 

		# construct the model
	    return BHM(L, n_max, isCanonical, isPeriodic, dim, M, keep_sites, N, Hk, Hu, Hn, Hmf)
	end
end