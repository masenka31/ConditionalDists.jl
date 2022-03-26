#import Distributions: probs
#import ConditionalDists: condition

"""
    ConditionalCategorical(m)

Specialization of ConditionalDistribution for `Categorical`. Can be used for
single inputs as well as batches. If the input is a batch, returns `BatchCategorical`.
Allows for a parametrized, differentiable categorical distribution.

**Important!** The mapping needs to return a probability distribution.
Easiest way to ensure this is to use softmax after the neural network
layers.

# Code example

```julia-repl
julia> m = Chain(Dense(2, 4, swish), Dense(4, 2), softmax)
julia> p = ConditionalCategorical(m)
julia> condition(p, randn(2))
julia> condition(p, randn(2, 5))
```
"""
struct ConditionalCategorical{Tm} <: AbstractConditionalDistribution
    mapping::Tm
end

Distributions.probs(p::ConditionalCategorical, z::AbstractVector) = probs(condition(p, z))

function condition(p::ConditionalCategorical, z::AbstractVector)
    α = p.mapping(z)
    DistributionsAD.Categorical(α)
end

function condition(p::ConditionalCategorical, z::AbstractMatrix)
    α = p.mapping(z)
    BatchCategorical(α)
end

# Flux.@functor ConditionalCategorical
@functor ConditionalCategorical

"""
    BatchCategorical

Implements a Categorical distribution for batches of samples, where the batch
is a matrix of size (n_classes, n_samples).
"""
struct BatchCategorical{Tm <: AbstractMatrix} <: DiscreteMatrixDistribution
    α::Tm
end

Distributions.probs(p::BatchCategorical) = p.α
Base.eltype(p::BatchCategorical) = eltype(p.α)
Distributions.params(p::BatchCategorical) = (p.α,)

function Distributions.pdf(p::BatchCategorical, y::AbstractVector{T}) where T <: Real
    α = probs(p)
    n = size(α, 2)
    xr = round.(Int, y)
    map(i -> α[xr[i], i], 1:n)
end
function Distributions.logpdf(p::BatchCategorical, y::AbstractVector{T}) where T <: Real
    log.(pdf(p, y))
end

# we do not need this to be differentiable (no reparametrization trick)
function Distributions.rand(p::BatchCategorical)
    α = probs(p)
    map(x -> rand(Categorical(collect(x))), eachcol(α))
end

function Distributions.entropy(p::BatchCategorical)
    α = probs(p)
    map(i -> entropy(α[:, i]), 1:size(α, 2))
end