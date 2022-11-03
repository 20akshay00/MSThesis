using JLD

begin
    params = [6, 4, false, true, 0]
    model = BHM(6, 4, false, true, 0);

	ln = 100
	t = range(start = 0, stop = 0.45, length = ln)
	mu = range(start = 0, stop = 3., length = ln)
	U = 1.
    V = 0.14

	gs = Array{Vector{Float64}}(undef, ln, ln)
	
	@time Threads.@threads for k1 in 1:ln
		for k2 in 1:ln
	    	gs[k2, k1] = get_ground_state(model, t[k1], mu[k2], U, V)
		end
	end

    save("../data/ed/V=$(round(V, digits=4))_n=$(ln).jld", "t", t, "mu", mu, "V", V, "ground_state", gs, "model_params", params)
end