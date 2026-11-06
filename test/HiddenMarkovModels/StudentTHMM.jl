function test_StudentTHMM_fit()
    Random.seed!(42)

    output_dim = 1
    K = 1

    ν_true = 4.0
    μ_true = 0.6
    σ_true = 1.8

    emission_true = StudentTEmission(; output_dim=output_dim, ν=ν_true, μ=μ_true, σ=σ_true)
    A = [1.0;;]
    πₖ = [1.0]

    true_model = HiddenMarkovModel(; K=K, B=[emission_true], A=A, πₖ=πₖ)

    n = 10000
    _, data = rand(true_model; n=n)

    emission_init = StudentTEmission(; output_dim=output_dim, ν=ν_true, μ=0.0, σ=1.0)
    test_model = HiddenMarkovModel(; K=K, B=[emission_init], A=A, πₖ=πₖ)

    ll = StateSpaceDynamics.fit!(test_model, data)

    @test isapprox(test_model.B[1].ν, ν_true; atol=1e-12)
    @test isapprox(test_model.B[1].μ, μ_true; atol=0.15)
    @test isapprox(test_model.B[1].σ, σ_true; atol=0.2)
    @test all(isfinite, ll)
    @test all(diff(ll) .>= -1e-6)
end
