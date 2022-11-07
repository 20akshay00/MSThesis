begin 
	ln = 100
	t = range(start = 0, stop = 0.15, length = ln)
	mu = range(start = 0., stop = 3., length = ln)
	
	order_param = zeros((ln, ln))

    num_sites = 4
    nbosons = 4
    model = BHM(num_sites, nbosons, false, 0);

    Threads.@threads for k1 in 1:ln
        for k2 in 1:ln
            order_param[k2, k1] = abs(get_order_parameter(model, t[k1], mu[k2]))
        end 
    end

    save("../data/cmft/L=$(num_sites)_nb=$(nbosons)_n=$(ln).jld", "t", t, "mu", mu, "order_param", order_param, "L", num_sites, "nbosons", nbosons)
end