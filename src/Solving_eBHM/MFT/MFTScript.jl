using JLD

begin
	model = MeanField(6, [0 3 3; 3 0 3; 3 3 0])
	z, num_points = 3, 100
	t = range(start = 0.001, stop = 0.04, length = num_points)
	mu = range(start = -0.2, stop = 1.2, length = num_points)
	V = 0.16
	
	order_param = zeros((num_points, num_points, model.num_sites * 2))
	
	@time Threads.@threads for k1 in 1:num_points
	    for k2 in 1:num_points
	        order_param[k2, k1, :] .= get_order_parameter(model, t[k1], mu[k2], 1., V, z, 1e-5)
	    end
	end

    save("../data/tri/V=$(round(V, digits=4))_n=$(num_points).jld", "t", t, "mu", mu, "V", V, "order_param", order_param, "n_max", model.n_max, "unit_cell", model.unit_cell)
end