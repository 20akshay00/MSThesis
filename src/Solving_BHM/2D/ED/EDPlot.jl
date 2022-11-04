using Plots, LaTeXStrings

begin
	theme(:lime)
	p1 = heatmap(t, mu, order_param, clims = (0, 1))
	plot!(
	    ylabel = "μ (chemical potential)",
	    xlabel = "t (hopping parameter)",
	    framestyle = :box, 
	    title = "4-site Exact Diagonalization",  
	    colorbar_title = "\n Order Parameter",
	size = (550, 500),
	margin = 6Plots.mm)
end
