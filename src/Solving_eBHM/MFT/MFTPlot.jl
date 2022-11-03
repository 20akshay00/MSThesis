using LaTeXStrings, Plots

function classify(order_params, err = 1e-2)
	
    function identify(params...)
        c = 4 # default

		psi, rho = eachcol(reshape(collect(params), :, 2))
		
        if all(y -> isapprox(y, rho[1], atol = err), rho) # BHM
			if isapprox(psi[1], 0, atol = err) 
				c = 0 # MI
			else
				c = 2 # SF
			end
		else
			if isapprox(psi[1], 0, atol = err) # eBHM
				c = 1 # DW
			else 
				c = 3 # SS
			end
		end

        return c
    end
	
    return mapslices((arr) -> identify(arr...), order_params, dims = 3)[:, :]
end

function phase_diagram(model, t, mu, order_param, err = 1e-2, loc = [0, 0])
    p = []
    n = model.num_sites
    p_names = latexstring.(vcat(repeat(["\\psi_"], n) .* ('@' .+ (1:n)), repeat(["\\rho_"], n) .* ('@' .+ (1:n))))
    
    for i in 1:(2 * n)
        p_temp = (i <= n) ? heatmap(t, mu, order_param[:, :, i], xticks = nothing) : heatmap(t, mu, order_param[:, :, i], c = cgrad(:thermal, [1, 2, 3]), clims = (0, 4), xticks = nothing) 
        plot!(
            ylabel = "μ/U",
            xlabel = "t/U",
            framestyle = :box, 
            title = p_names[i],
            margin = 5Plots.mm)
    
        push!(p, p_temp)
    end
    
    phases = heatmap(t, mu, classify(order_param, err), 
        # c = palette([:black, "#02ff00", "#d4d0c8", "#ff0000", :blue]), 
        c = palette(["#fbe59b", "#75a5ce", "#e86572", "#7d62a9", :transparent]),
        title = "eBHM MFT - Phase Diagram",
        ylabel = "μ/U",
        xlabel = "t/U",
        colorbar = false,
        colorbar_ticks = [0, 1, 2, 3, 4],
        clims = (0, 4),
        right_margin = 10Plots.mm,
        bottom_margin = 7Plots.mm)
    
    if !isnothing(loc)
        annotate!(loc..., text("zV/U = $(model.z * 0.15)", :black, :left, 20))
    end
    
    l = @layout [
        [grid(n, 1)] c{0.5w} [grid(n, 1)]
    ]
    plot(p[1:n]..., phases, p[(n+1):end]..., size = ((n + 1) * 500, 1.5 * 500), layout = l)
end