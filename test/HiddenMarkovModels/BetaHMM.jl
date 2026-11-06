function test_BetaHMM_fit()
    Random.seed!(42)

    output_dim = 1
    K = 1

    α_true = 2.5
    β_true = 7.5

    emission_true = BetaEmission(; output_dim=output_dim, α=α_true, β=β_true)
    A = [1.0;;]
    πₖ = [1.0]

    true_model = HiddenMarkovModel(; K=K, B=[emission_true], A=A, πₖ=πₖ)

    n = 8000
    _, data = rand(true_model; n=n)

    emission_init = BetaEmission(; output_dim=output_dim, α=1.5, β=1.5)
    test_model = HiddenMarkovModel(; K=K, B=[emission_init], A=A, πₖ=πₖ)

    ll = StateSpaceDynamics.fit!(test_model, data)

    @test isapprox(test_model.B[1].α, α_true; atol=0.5)
    @test isapprox(test_model.B[1].β, β_true; atol=0.8)
    @test all(diff(ll) .>= -1e-6)
    @test all((0 .< data) .& (data .< 1))
end
