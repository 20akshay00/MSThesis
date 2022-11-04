### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 4b0f86a0-0daf-11ed-36e0-2d007099c7b4
using Combinatorics, SparseArrays, KrylovKit, LinearAlgebra, Plots, LaTeXStrings, Optim


# ╔═╡ a0915818-c566-4963-91a9-3aad31317608
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
		E_gs(psi) = eigsolve(-t .* model.Hk .+ U .* model.Hu .- mu .* model.Hn + t .* (-psi[1] .* model.Hmf + 2 * psi[1]^2 * I), 1, :SR)[1][1] |> real

		return Optim.minimizer(optimize(E_gs, zeros(1)))[1]
	end
end

# ╔═╡ c906b6d6-85ab-463f-b676-4eb3f9b928db
begin
	struct BHM
	    L :: Int64
	    n_max :: Int64
		isPeriodic :: Bool
	    dim :: Int64
	    M :: Float64 
		
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
		Hn :: SparseMatrixCSC{Float64, Int64}
		Hmf :: SparseMatrixCSC{Float64, Int64}
	end
	
	function BHM(L, n_max, isPeriodic, M = 0)

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
		Hk, Hu = spzeros(dim, dim), spzeros(dim, dim)
		Hn = copy(N_tot)
		Hmf = A[1] + A_dag[1] + A[L] + A_dag[L]
		l_max = isPeriodic ? L : (L - 1)

		for i in 1:l_max
			j = mod1(i + 1, L)
			Hk = Hk + (A_dag[i] * A[j] + A_dag[j] * A[i])
		end

		for i in 1:L
			Hu =  Hu + 0.5 * N[i] * (N[i] - I)
		end

		
		# construct the model
	    return BHM(L, n_max, isPeriodic, dim, M, a, a_dag, n, A, A_dag, N, N_tot, Hk, Hu, Hn, Hmf)
	end
end

# ╔═╡ f34bc577-d072-4c71-907f-c08f23e1bc18
model = BHM(4, 4, false, 0);

# ╔═╡ 37ebc92f-7957-41eb-8f5e-1a453d1352ce
begin 
	num_points = 100
	t = range(start = 0, stop = 0.15, length = num_points)
	mu = range(start = 0., stop = 3., length = num_points)
	
	order_param = zeros((4, num_points, num_points))

	for num_sites in [1, 2, 3, 4]
		model = BHM(num_sites, 4, false, 0);
		Threads.@threads for k1 in 1:num_points
			for k2 in 1:num_points
		    	order_param[num_sites, k2, k1] = abs(get_order_parameter(model, t[k1], mu[k2]))
			end 
		end
	end
end

begin
	theme(:lime)
	plot()
	for num_sites in 1:4
		plot!(t[[findlast(<(0.2), order_param[num_sites, i, :]) for i in 1:size(order_param)[2]]], mu, label = "$(num_sites) sites", ls = :dash, lw = 1.25)
	end
	plot!(
	    ylabel = "μ (chemical potential)",
	    xlabel = "t (hopping parameter)",
	    title = "Cluster Mean Field Theory (1D)",  
	    colorbar_title = "\n Order Parameter",
		framestyle = :box, 
		size = (550, 500),
		xlim = (0, 0.15))
end
