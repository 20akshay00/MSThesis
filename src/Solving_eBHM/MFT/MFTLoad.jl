using JLD

d = load("./data/tri/V=0.4_n=50.jld")

order_param = d["order_param"]
t = d["t"]
mu = d["mu"]

unit_cell = d["unit_cell"]
n_max = d["n_max"]

model = MeanField(n_max, unit_cell)