if GenGPT3.lookup_openai_api_key() == ""
    error("OPENAI_API_KEY not set, embedding tests cannot run.")
end 

@testset "Embedder" begin

    embedder = GenGPT3.Embedder()

    e = embedder("What is the tallest mountain on Mars?")
    @test length(e) == 1536

    embeddings = embedder([
        "What is the tallest mountain on Mercury?",
        "What is the tallest mountain on Venus?",
        "What is the tallest mountain on Earth?",
        "What is the tallest mountain on Mars?"
    ])
    @test length(embeddings) == 4

    sims = GenGPT3.similarity(e, embeddings)
    @test isapprox(sims[4], 1.0, atol=1e-3) 

    embeddings = reduce(hcat, embeddings)
    sims = GenGPT3.similarity(e, embeddings)
    @test isapprox(sims[4], 1.0, atol=1e-3)

end
