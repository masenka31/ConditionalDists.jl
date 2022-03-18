@testset "ConditionalCategorical" begin
    c = 3
    x = randn(2)
    y = 1
    xb = randn(2, 5)
    yb = [1,2,3,2,3]
    
    m = Chain(Dense(2, 3, swish), Dense(3,c), softmax)
    p = ConditionalCategorical(m)

    # test that the distribution is differentiable
    # and parameters are okay
    ps = Flux.params(p)
    @test !isempty(ps)
    @test length(ps) == length(Flux.params(m))
    fl() = logpdf(condition(p, x), y)
    @test_nowarn Flux.gradient(fl, ps)

    # test for vectors
    res = condition(p, x)
    @test res isa Categorical
    @test size(probs(res)) == (c, )

    # test for batches
    res = condition(p, xb)
    @test res isa ConditionalDists.BatchCategorical
    @test size(probs(res)) == (c, size(xb, 2))

    # test pdf, logpdf and entropy for single
    f(x) = [0.2,0.3,0.5]
    p = ConditionalCategorical(f)
    res = condition(p, x)

    @test entropy(res) == entropy([0.2,0.3,0.5])
    @test pdf.(res, [1,2,3]) == [0.2,0.3,0.5]
    @test logpdf.(res, [1,2,3]) == log.([0.2,0.3,0.5])

    # test pdf, logpdf, entropy for batch
    g(x) = mapreduce(xi -> f(xi), hcat, eachcol(x))
    p = ConditionalCategorical(g)
    res = condition(p, xb)

    @test length(unique(entropy(res))) == 1
    @test unique(entropy(res))[1] == entropy([0.2,0.3,0.5])
    @test length(pdf(res, yb)) == size(xb, 2)
    @test pdf(res, yb) == [0.2,0.3,0.5,0.3,0.5]
    @test logpdf(res, yb) == log.([0.2,0.3,0.5,0.3,0.5])

end