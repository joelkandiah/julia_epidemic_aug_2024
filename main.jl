# Initialise Pacakges
using OrdinaryDiffEq
using DiffEqCallbacks
using Distributions
using DataInterpolations
using Turing
using LinearAlgebra
using StatsPlots
using Random
using Bijectors
using TOML

# Set up a model using 4 different random sets of parameters and sets of observations
for q in 1:4

    # Set seed
    if q == 1
        Random.seed!(1234)
    elseif q == 2
        Random.seed!(1357)
    elseif q == 3
        Random.seed!(2358)
    elseif q == 4
        Random.seed!(3581)
    end

    # Set up an output directory for the plots
    outdir = "Plot attempt $q/"
    if !isdir(outdir)
        mkdir(outdir)
    end
    
    # Initialise parameters
    # Length of outbreak
    tmax = 80.0
    tspan = (0.0, tmax)
    obstimes = 1.0:1.0:tmax

    # Set the number of people who are infected within the population and the population size
    N = 1_000_000.0
    I0 = 100.0
    u0 = [N-I0, I0, 0.0] # S,I,R

    # Set the mean duration of infection (1/γ)
    γ = 0.125

    # Store a parameter vector
    p = [γ, N]; # γ, N

    # Set up the simulated random walk on the β parameters
    # Initialise the parameters to draw the sample of β₀ and βₜ
    β₀σ²= 0.15
    β₀μ = 0.18

    βσ² = 0.15

    # Create store for the β values
    true_beta = repeat([NaN], 3)
    
    # Draw log(β₀) from Normal(log(β₀μ), β₀σ²)
    true_beta[1] = exp(rand(Normal(log(β₀μ), β₀σ²)))

    # Draw log(βₜ) from Normal(log(βₜ₋₁), βσ²)
    for i in 2:(length(true_beta) - 1)
        true_beta[i] = exp(log(true_beta[i-1]) + rand(Normal(0.0,βσ²)))
    end 
    true_beta[length(true_beta)] = true_beta[length(true_beta)-1]
    
    # Store times at which the random walk evolves
    knots = collect(0.0:40.0:tmax)
    K = length(knots)

    # Store all chosen parameters to write to a toml file
    params = Dict(
        "tmax" => tmax,
        "N" => N,
        "I0" => I0,
        "γ" => γ,
        "β" => true_beta,
        "knots" => knots,
        "inital β mean" => β₀μ,
        "initial β sd" => β₀σ²,
        "β sd" => βσ²
    )

    open(string(outdir, "params.toml"), "w") do io
           TOML.print(io, params)
    end

    # Construct piecewise constant function of β values
    function betat(p_, t)
        beta = ConstantInterpolation(p_, knots)
        return beta(t)
    end;

    # Write the sir model ode (note mutates du)
    function sir_tvp_ode!(du, u, p_, t)
        (S, I, R) = u
        (γ, N) = p_[1:2]
        βt = betat(p_[3:end], t)
        infection = βt*S*I/N
        recovery = γ*I

        # Use @inbounds to prevent boundchecks for speed
        @inbounds begin
            du[1] = -infection
            du[2] = infection - recovery
            du[3] = infection
        end
    end;

    # Construct ODE problem using the parameters we have chosen and ODE system for SIR model
    prob_ode = ODEProblem(sir_tvp_ode!, u0, tspan, [p..., true_beta...])
    
    # Evaluate ODE 
    # Use non-default solver and tolerances for stability
    # Use saveat to determine the output times of the solver
    # Use tstops and d_discontinuities to note when SIR might not be smooth due to the random walk
    sol_ode = solve(prob_ode,
                AutoVern7(Rodas4P()),
                maxiters = 1e6,
                abstol = 1e-8,
                reltol = 1e-5,
                # callback = cb,
                saveat = 1.0,
                tstops = knots[2:end-1],
                d_discontinuities = knots);

    # Plot the ODE dynamics
    plot(sol_ode,
        xlabel="Time",
        ylabel="Number",
        labels=["S" "I" "R"])

    # Calculate the cumulative recoveries and number of infected individuals
    R = [0; Array(sol_ode(obstimes))[3,:]] # Cumulative recoveries
    I = [I0; Array(sol_ode(obstimes))[2,:]] # Number of infected individuals

    # Write function to calculate difference between adjacent elements of an array
    function adjdiff(ary)
        ary1 = @view ary[begin:end-1]
        ary2 = @view ary[begin+1:end]
        return ary2 .- ary1
    end

    # Calculate the number of newly recovered individuals each day
    X = adjdiff(R)

    # Define NegativeBinomial distribution where for Y ~ NegativeBinomial3(μ, ϕ), E(Y) = μ, Var(Y) = μ(1-ϕ)
    function NegativeBinomial3(μ, ϕ)
        p = 1 / (1 + ϕ)
        r = μ / ϕ
        return NegativeBinomial(r, p)
    end

    # Draw the data for the simulation
    # Similar to some ascertainment of infections
    # Note: uses 0.3 as ascertainment proportion and add a small number for stability
    Y = rand.(NegativeBinomial3.(X .* 0.3 .+ 1e-3, 10));

    # Plot data (compare random data to proportion of recovereds)
    bar(obstimes, Y, legend=false)
    plot!(obstimes, X .* 0.3, legend=false)

    # Plot infectious populatioon
    plot([0; obstimes], I, legend=false)


    # Initialise ODEproblem so it can be recreated in the model
    prob_tvp = ODEProblem(sir_tvp_ode!,
            u0,
            tspan,
            true_beta);
            
    # Define turing model for the SIR model
    # Priors for betas are same as the simulated distributions
    # Prior for initial infected population log(I₀/N) ~ Normal(-9.0,0.2)
    @model function bayes_sir_tvp(
        y,
        K,
        ::Type{T}=Float64;
        γ = γ,
        N = N,
        knots = knots,
        obstimes = obstimes,
    ) where {T <: Real}
        # Set prior for initial infected
        log_I₀  ~ Normal(-9.0, 0.2)
        
        # Set initial populations in each compartment
        I = exp(log_I₀) * N
        u0 = [N-I, I, 0.0]

        # Set priors for betas
        ## Note how we clone the endpoint of βt
        p = Vector{T}(undef, K+2)
        log_β = Vector{T}(undef, K-2)
        p[1:2] .= [γ, N]
        log_β₀ ~ Normal(log(0.4), 0.2)
        p[3] = exp(log_β₀)
        for i in 4:K+1
            log_β[i-3] ~ Normal(0.0, 0.2)
            p[i] = exp(log(p[i-1]) + log_β[i-3])
        end
        p[K+2] = p[K+1]

        # Run model
        ## Remake with new initial conditions and parameter values
        prob = remake(prob_tvp,
            u0=u0,
            p=p,
            d_discontinuities = knots)
        
        ## Solve
        sol = solve(prob,
                AutoVern7(Rodas4P()),
                saveat = obstimes,
                maxiters = 1e6,
                d_discontinuities = knots[2:end-1],
                tstops = knots[2:end-1],
                abstol = 1e-7,
                reltol = 1e-4)

        # Automatically reject the proposal values if the solver returns an error
        if any(sol.retcode != :Success)
            if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
                @DynamicPPL.addlogprob! -Inf
                return
            end
        end
        
        ## Calculate new recovereds
        sol_R = [0; Array(sol(obstimes))[3,:]]
        sol_X = (adjdiff(sol_R))

        ## Automatically reject proposals if the numerical error leads to negative numbers of recovereds
        if (any(sol_X .< -(1e-3)) | any(Array(sol(obstimes))[2,:] .< -1e-3))
            if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
                # print("hit")
                @DynamicPPL.addlogprob! -Inf
                return
            end
        end

        # Evaluate the likelihood
        y ~ product_distribution(NegativeBinomial3.(sol_X .* 0.3 .+ 1e-3, 10))

        return (; sol, p)
    end;



    # Produce inference using MCMC
    using AdvancedMH

    @time chn = sample(bayes_sir_tvp(Y, K),
        MH(:log_I₀ => AdvancedMH.RandomWalkProposal(Normal(0,6e-2)),
        :log_β₀  => AdvancedMH.RandomWalkProposal(Normal(0,3e-3)),
        #    :log_β => AdvancedMH.RandomWalkProposal(product_distribution(Normal.(repeat([0],K-2), repeat([3e-3],K-2))))
        :log_β => AdvancedMH.RandomWalkProposal(Normal(0, 3e-3))
        ),
        MCMCThreads(), 100, 6, discard_initial = 100_000, thinning = 1000, save_state = true)

    plot(chn)
    savefig(string(outdir,"chn_mh.png"))

   # Transform the beta parameters 
    betas = Array(chn[chn.name_map.parameters[1:end]])
    beta_idx = [collect(2:K); K]

    betas[:,2:end] = exp.(cumsum(betas[:,2:end], dims = 2))

    # Estimate credible interval for betas
    beta_μ = [quantile(betas[:,i], 0.5) for i in beta_idx]
    betas_lci = [quantile(betas[:,i], 0.025) for i in beta_idx]
    betas_uci = [quantile(betas[:,i], 0.975) for i in beta_idx]

    # Plot betas
    plot(obstimes,
        betat(beta_μ, obstimes),
        xlabel = "Time",
        ylabel = "β",
        label="Using Random Walk MH algorithm",
        title="Estimates of β",
        color=:blue)
    plot!(obstimes,
        betat(betas_lci, obstimes),
        alpha = 0.3,
        fillrange = betat(betas_uci, obstimes),
        fillalpha = 0.3,
        color=:blue,
        label="95% credible intervals")
    plot!(obstimes,
        betat(true_beta, obstimes),
        color=:red,
        label="True β")

    savefig(string(outdir,"mh_betas.png"))

    # Generate confidence intervals for the number of infectious individuals
    function generate_confint_infec(chn, Y, K)
        chnm_res = generated_quantities(bayes_sir_tvp(Y, K), chn);

        infecs = cat(map(x ->Array(x.sol)[2,:], chnm_res)..., dims = 2)
        lowci_inf = mapslices(x -> quantile(x, 0.025), infecs, dims = 2)[:,1]
        medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 2)[:, 1]
        uppci_inf = mapslices(x -> quantile(x, 0.975), infecs, dims = 2)[:,1]
        return (; lowci_inf, medci_inf, uppci_inf)
    end

    # Generate confidence intervals for the number of recovered individuals
    function generate_confint_recov(chn, Y, K)
        chnm_res = generated_quantities(bayes_sir_tvp(Y, K), chn);

        infecs = cat(map(x ->Array(x.sol)[3,:], chnm_res)..., dims = 2)
        lowci_inf = mapslices(x -> quantile(x, 0.025), infecs, dims = 2)[:,1]
        medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 2)[:, 1]
        uppci_inf = mapslices(x -> quantile(x, 0.975), infecs, dims = 2)[:,1]
        return (; lowci_inf, medci_inf, uppci_inf)
    end



    # Plot estimates of the infectious and recovered compartments
    confint = generate_confint_infec(chn,Y,K)

    I_dat = Array(sol_ode(obstimes))[2,:] # Cumulative cases

    plot(confint.medci_inf, ribbon = (confint.medci_inf - confint.lowci_inf,  confint.uppci_inf - confint.medci_inf),  legend = false)
    plot!(I_dat, linesize = 3)

    savefig(string(outdir,"infections_mh.png"))

    confint = generate_confint_recov(chn,Y,K)

    R_dat = Array(sol_ode(obstimes))[3,:] # Cumulative cases

    plot(confint.medci_inf, ribbon = (confint.medci_inf - confint.lowci_inf, confint.uppci_inf - confint.medci_inf), legend = false)
    plot!(R_dat, linesize = 3)

    savefig(string(outdir,"recoveries_mh.png"))

    # Perform inference via NUTS
    using DynamicHMC


    @time ode_nuts_2 = sample(bayes_sir_tvp(Y, K), Turing.NUTS(1000, 0.65), MCMCThreads(), 100, 6, discard_initial = 200, thinning = 10)

    plot(ode_nuts_2)
    savefig(string(outdir,"chn_nuts.png"))

    betas = Array(ode_nuts_2[ode_nuts_2.name_map.parameters[1:end]])
    beta_idx = [collect(2:K); K]

    betas[:,2:end] = exp.(cumsum(betas[:,2:end], dims = 2))
    beta_μ = [quantile(betas[:,i], 0.5) for i in beta_idx]
    betas_lci = [quantile(betas[:,i], 0.025) for i in beta_idx]
    betas_uci = [quantile(betas[:,i], 0.975) for i in beta_idx]

    plot(obstimes,
        betat(beta_μ, obstimes),
        xlabel = "Time",
        ylabel = "β",
        label="Using the NUTS algorithm",
        title="Estimates of β",
        color=:blue)
    plot!(obstimes,
        betat(betas_lci, obstimes),
        alpha = 0.3,
        fillrange = betat(betas_uci, obstimes),
        fillalpha = 0.3,
        color=:blue,
        label="95% credible intervals")
    plot!(obstimes,
        betat(true_beta, obstimes),
        color=:red,
        label="True β")

    savefig(string(outdir,"nuts_betas.png"))


    confint = generate_confint_infec(ode_nuts_2,Y,K)
    plot(confint.medci_inf, ribbon = (confint.medci_inf - confint.lowci_inf, confint.uppci_inf - confint.medci_inf) , legend = false)
    plot!(I_dat, linesize = 3)

    savefig(string(outdir,"infections_nuts.png"))

    confint = generate_confint_recov(ode_nuts_2,Y,K)

    plot(confint.medci_inf, ribbon = (confint.medci_inf - confint.lowci_inf, confint.uppci_inf - confint.medci_inf)  , legend = false)
    plot!(R_dat, linesize = 3)

    savefig(string(outdir,"recoveries_nuts.png"))

    # advi = ADVI(10, 1000) # 10 samples, 1000 gradient iterations
    # @time ode_advi_samp = vi(bayes_sir_tvp(Y, K), advi);

    # ode_advi_postsamples = rand(ode_advi_samp, 1000);
    # beta_idx = [collect(2:K);K]
    # ode_advi_postsamples[2:end,:] = exp.(cumsum(ode_advi_postsamples[2:end,:], dims = 1))
    # betas = [quantile(ode_advi_postsamples[i,:],0.5) for i in beta_idx]
    # betas_lci = [quantile(ode_advi_postsamples[i,:], 0.025) for i in beta_idx]
    # betas_uci = [quantile(ode_advi_postsamples[i,:], 0.975) for i in beta_idx]


    # params = [ode_advi_postsamples[:,i] for i in 1:size(ode_advi_postsamples)[2]]

    # # map(x -> display(x[2:end]), params)
    # # N
    # res = map(x -> solve(ODEProblem(sir_tvp_ode!,[ N - (exp(x[1]) * N), exp(x[1]) * N, 0], tspan, [γ, N, x[2:end]...,x[end]]), AutoTsit5(Rosenbrock32()), saveat = 1).u, params)
    # # cat(cat(res..., dims = 2)..., dims = 3)

    # plot(obstimes,
    #     betat(betas, obstimes),
    #     xlabel = "Time",
    #     ylabel = "β",
    #     label="Estimated β",
    #     title="ADVI estimates",
    #     color=:green)
    # plot!(obstimes,
    #     betat(betas_lci, obstimes),
    #     alpha = 0.3,
    #     fillrange = betat(betas_uci, obstimes),
    #     fillalpha = 0.3,
    #     color=:green,
    #     label="95% credible intervals")
    # plot!(obstimes,
    #     betat(true_beta, obstimes),
    #     color=:red,
    #     label="True β")

    # savefig(string(outdir,"advi_betas.png"))

    # infecs = stack(map(x -> stack(x)[2,:], res))
    # lowci_inf = mapslices(x -> quantile(x, 0.025),infecs, dims = 2)[:,1]
    # medci_inf = mapslices(x -> quantile(x, 0.5),infecs, dims = 2)[:, 1]
    # uppci_inf = mapslices(x -> quantile(x, 0.975),infecs, dims = 2)[:,1]

    # plot(medci_inf, ribbon = (medci_inf - lowci_inf, uppci_inf - medci_inf), legend = false)
    # plot!(I_dat, linesize = 3)

    # savefig(string(outdir,"infections_advi.png"))

    # recovs = stack(map(x -> stack(x)[3,:], res))
    # lowci_inf = mapslices(x -> quantile(x, 0.025),recovs, dims = 2)[:,1]
    # medci_inf = mapslices(x -> quantile(x, 0.5),recovs, dims = 2)[:, 1]
    # uppci_inf = mapslices(x -> quantile(x, 0.975),recovs, dims = 2)[:,1]

    # plot(medci_inf, ribbon = (medci_inf - lowci_inf, uppci_inf - medci_inf), legend = false)
    # plot!(R_dat, linesize = 3)

    # savefig(string(outdir,"recovs_advi.png"))
end