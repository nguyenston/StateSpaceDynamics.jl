# ACDC Interface Design for StateSpaceDynamics.jl

## Overview

This document describes the design of the ACDC (Accumulated Cutoff Discrepancy Criterion) interface for robust model selection in StateSpaceDynamics.jl. The interface supports mixture models and hidden Markov models with pluggable discrepancy measures.

## Background

ACDC addresses the problem of model selection under misspecification. Traditional methods like AIC/BIC optimize for predictive performance and tend to overfit when the model family doesn't contain the true data-generating process. ACDC instead measures how well each component's inferred distribution matches the assumed model distribution, providing robustness to model misspecification.

The core ACDC loss function is:

```
R^ρ(K) = Σₖ₌₁ᴷ max(0, D̂ₖ - ρ)
```

Where:
- `D̂ₖ` is the estimated discrepancy for component k
- `ρ` is a user-specified tolerance threshold
- K is the number of components

## Type Hierarchy

```
ComponentDiscrepancy (abstract)
├── KLDiscrepancy{T}
├── WassersteinDiscrepancy{T}
├── SquaredErrorDiscrepancy{T}
└── [User-defined discrepancies]

ACDCResult{T}
├── K::Int
├── component_discrepancies::Vector{T}
└── component_usage::Vector{T}
```

## Core Interface

### 1. Discrepancy Measures

```julia
abstract type ComponentDiscrepancy end

# Required method for all discrepancy types
function compute_discrepancy(
    d::ComponentDiscrepancy,
    samples::AbstractMatrix{T},
    reference_dist
) where {T<:Real} -> T
```

**Built-in Implementations:**

| Type | Description | Best For |
|------|-------------|----------|
| `KLDiscrepancy` | KL divergence via kNN estimation | General purpose |
| `WassersteinDiscrepancy` | Optimal transport with Sinkhorn | High dimensions |
| `SquaredErrorDiscrepancy` | MSE-based comparison | Gaussian models |

### 2. Component Discrepancies Function

The main interface function that models must implement:

```julia
function component_discrepancies(
    model,                           # Fitted model
    data::AbstractMatrix,            # Observed data
    discrepancy::ComponentDiscrepancy # Discrepancy measure
) -> ACDCResult
```

**Supported Models:**

| Model Type | Component Interpretation | Residual Type |
|------------|-------------------------|---------------|
| `GaussianMixtureModel` | Mixture components | Whitened residuals |
| `PoissonMixtureModel` | Mixture components | Standardized residuals |
| `HiddenMarkovModel` | Hidden states | Emission-specific residuals |

### 3. Loss Computation

```julia
function acdc_loss(result::ACDCResult, ρ::Real) -> Real
```

Returns the accumulated cutoff discrepancy loss for threshold ρ.

### 4. Model Selection

```julia
function acdc_select(results::Vector{ACDCResult}, ρ::Real) -> Int
```

Returns the smallest K minimizing the ACDC loss.

## Integration with StateSpaceDynamics.jl

### File Structure

```
src/
├── StateSpaceDynamics.jl  # Add: include("ACDC.jl")
├── ACDC.jl                # Module wrapper (includes Interface + Adapters)
├── ACDCInterface.jl       # Abstract types, generic functions, discrepancy measures
├── ACDCAdapters.jl        # Model-specific implementations (GMM, PMM, HMM)
├── GlobalTypes.jl         # No changes needed
├── MixtureModels.jl       # No changes needed
└── HiddenMarkovModels.jl  # No changes needed
```

### Required Changes to StateSpaceDynamics.jl

```julia
# In StateSpaceDynamics.jl, add:
include("ACDC.jl")
```

### Exports to Add

```julia
export ACDCResult, ComponentDiscrepancy
export component_discrepancies, acdc_loss, acdc_select
export KLDiscrepancy, WassersteinDiscrepancy, SquaredErrorDiscrepancy
```

## Implementation Details

### Mixture Model Residuals

For a K-component GMM with parameters (πₖ, μₖ, Σₖ), the residuals for component k are:

1. Compute posterior probabilities γₙₖ = P(zₙ = k | xₙ, θ)
2. Select observations with γₙₖ > threshold
3. Compute whitened residuals: εₙ = Lₖ⁻¹(xₙ - μₖ) where Σₖ = LₖLₖᵀ
4. Reference distribution: N(0, I)

### HMM Residuals

For HMMs, each state k has an emission model B[k]. The residual computation depends on the emission type:

| Emission Type | Residual Computation |
|--------------|---------------------|
| `GaussianEmission` | Whitened: L⁻¹(y - μ) |
| `GaussianRegressionEmission` | Whitened: L⁻¹(y - Xβ) |
| `BernoulliRegressionEmission` | Deviance residuals |
| `PoissonRegressionEmission` | Pearson: (y - λ)/√λ |

### State Posterior Computation

For HMMs, state posteriors are computed via the forward-backward algorithm, reusing the existing `class_probabilities` function.

## Usage Example

```julia
using StateSpaceDynamics

# Generate or load data
data = ...  # D × N matrix

# Fit models for different K
K_range = 1:6
results = Vector{ACDCResult{Float64}}(undef, length(K_range))

for (i, K) in enumerate(K_range)
    model = GaussianMixtureModel(K, D)
    fit!(model, data)
    results[i] = component_discrepancies(model, data, SquaredErrorDiscrepancy())
end

# Examine component-wise discrepancies
for r in results
    println("K=$(r.K): discrepancies=$(r.component_discrepancies)")
end

# Select K for a given ρ
ρ = 0.3
selected_K = acdc_select(results, ρ)

# Or examine loss curve for ρ selection
ρ_values = range(0, 2, length=100)
for r in results
    losses = [acdc_loss(r, ρ) for ρ in ρ_values]
    # Plot or analyze losses
end
```

## Custom Discrepancy Implementation

Users can define custom discrepancy measures:

```julia
struct MyDiscrepancy{T<:Real} <: ComponentDiscrepancy
    # Custom fields
end

function StateSpaceDynamics.compute_discrepancy(
    d::MyDiscrepancy{T},
    samples::AbstractMatrix{T},
    reference_dist
) where {T<:Real}
    # Custom computation
    return discrepancy_value
end
```

## Design Rationale

### Why Return Component-wise Losses?

1. **Flexibility in ρ selection**: Users can implement various strategies (domain knowledge, stability analysis, cross-validation) without re-fitting models.

2. **Interpretability**: Component-wise discrepancies reveal which components are poorly fit, aiding model diagnosis.

3. **Efficiency**: The ACDC loss is piecewise linear in ρ, so all critical points can be computed from component discrepancies.

### Why Trait-based Design?

1. **Extensibility**: New models can support ACDC by implementing one function.

2. **Separation of concerns**: Discrepancy computation is separate from model fitting and selection logic.

3. **Type stability**: Julia's multiple dispatch enables efficient, type-stable code.

### Why Pluggable Discrepancies?

1. **Domain expertise**: Different applications may require different measures of distributional fit.

2. **Computational trade-offs**: KL divergence is accurate but slow; squared error is fast but less general.

3. **Research flexibility**: Easy to experiment with novel discrepancy measures.

## Future Extensions

1. **SLDS support**: Extend to Switching Linear Dynamical Systems (treating discrete states as components).

2. **Streaming computation**: Online updates for component discrepancies.

3. **Automatic ρ selection**: Implement stability-based methods from the paper.

4. **Parallel computation**: Multi-threaded discrepancy computation for large K.

## References

Li, J., Nguyen, N., Lai, M., Paschalidis, I. Ch., & Huggins, J. H. (2024). 
Robust Model Selection for Discovery of Latent Mechanistic Processes.
