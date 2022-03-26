module ConditionalDists

using LinearAlgebra
using Random
using Distributions
using DistributionsAD
using ChainRulesCore
using Functors
using Requires

export condition

export ConditionalDistribution
export ConditionalMvNormal
export ConditionalCategorical
export SplitLayer

include("cond_dist.jl")

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")
include("categorical.jl")
include("utils.jl")

function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        using .Flux: Dense
        function SplitLayer(in::Int, outs::Vector{Int}, acts::Vector)
            SplitLayer(Tuple(Dense(in,o,a) for (o,a) in zip(outs,acts)))
        end

        function SplitLayer(in::Int, outs::Vector{Int}, act=identity)
            acts = [act for _ in 1:length(outs)]
            SplitLayer(in, outs, acts)
        end
    end
end

end # module
