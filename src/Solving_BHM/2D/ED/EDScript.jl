L = 3
nbosons = 4
model = BHM(L, nbosons, false, true, 0);

begin
	ln = 100
	t = range(start = 0, stop = 0.15, length = ln)
	mu = range(start = 0, stop = 3., length = ln)
	
	gs = Array{Vector{Float64}}(undef, ln, ln)
	
	@time Threads.@threads for k1 in 1:ln
		for k2 in 1:ln
	    	gs[k2, k1] = get_ground_state(model, t[k1], mu[k2])
		end
	end

    save("../data/ed/L=$(L)_nb=$(nbosons)_n=$(ln).jld", "t", t, "mu", mu, "ground_state", gs, "L", L, "nbosons", nbosons)
end
