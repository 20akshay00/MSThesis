# Low-temperature phases of interacting bosons in a lattice
This repository hosts the various scripts written as part of my Master's thesis project, performed for the fulfillment of a BS-MS degree in Physics, at the [Indian Institute of Science Education and Research (IISER), Mohali](https://www.iisermohali.ac.in/). The written thesis can be found [here](https://20akshay00.github.io/files/MS18117_PRJ502.pdf). The code presented here is rather disorganized and redundant as I had not given much thought to it during my thesis, however, in the interest of reproducibility and open science, I have made this repository public. 

## What's what and what's where?

### Code
- `Calculating_WaveFunc` corresponds to the computation of BHM parameters using Wannier functions as described in Chapter 2.3.
- `Solving_BHM` contains the exact diagonalization and mean-field scripts used to study the Bose-Hubbard model in Chapter 3.
- `Solving_eBHM` contains the mean-field scripts used to study the extended Bose-Hubbard model in Chapter 4.
- `SpinBosons` contains the mean-field scripts used to study the spin-1 Bose-Hubbard model in Chapter 5.
- `PhaseBoundary` contains the code utilizing a bisection algorithm to compute phase boundaries for all the above mentioned chapters.
- `BosonMediation` contains Wolfram Language notebooks used to derive analytic expressions for the perturbative treatment performed in Chapter 6. Running these requires the VSCode extension, [WolframLanguageForJupyter](https://github.com/WolframResearch/WolframLanguageForJupyter).
- `SSE` contains a broken attempt at implementing the Stochastic Series Expansion as described in Chapter 7.
- [`BoseHubbardMachineLearning`](https://github.com/20akshay00/BoseHubbardMachineLearning) contains the neural network approach to solve the Bose-Hubbard model as described in Chapter 7.

### Presentations
The source files for some presentations are kept in `/ppt`, and can also be viewed on this [webpage](https://20akshay00.github.io/web-presentations/).

## Future prospects
This repository contains the foundations of a numerical package to compute mean-field phase diagrams of the extended Bose-Hubbard model using experimental parameters as an input. However, I have moved on to other projects at the current time and do not plan to work on this in the near future. Nevertheless, the opportunity remains to be picked up someday. 