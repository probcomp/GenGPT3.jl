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

@testset "EmbeddingStore" begin

    store = GenGPT3.EmbeddingStore(
        GenGPT3.Embedder(model = "text-embedding-ada-002"),
        ["What is the tallest mountain on Mercury?"],
        planet = ["mercury"]
    )
    push!(store, "What is the tallest mountain on Mars?",
          planet="mars", special=true)
    append!(store,
            ["What is the tallest mountain on Venus?",
             "What is the tallest mountain on Earth?"],
            planet = ["venus", "earth"])

    @test length(store) == 4
    @test store[1].planet == "mercury" && ismissing(store[1].special)
    @test store[2].planet == "mars" && store[2].special == true
    @test store[3].planet == "venus" && ismissing(store[3].special)
    @test store[4].planet == "earth" && ismissing(store[4].special)

    entries = findsimilar(store, "What is the tallest mountain on Mars?", 3)
    @test length(entries) == 3
    @test entries[1].planet == "mars"
    @test entries[2].planet == "venus"
    @test entries[3].planet == "mercury"

    embedding = store.embedder("What is the tallest mountain on Mars?")
    entries = findsimilar(store, embedding, 3, reversed=true) do row 
        ismissing(row.special) || !row.special
    end 
    @test length(entries) == 3
    @test entries[1].planet == "earth"
    @test entries[2].planet == "mercury"
    @test entries[3].planet == "venus"

    mktemp() do path, io
        GenGPT3.save_store(path, store)
        loaded_store = GenGPT3.load_store(path)
        @test loaded_store.data.text == store.data.text
        @test loaded_store.data.embedding == store.data.embedding
        @test loaded_store.data.planet == store.data.planet
        @test isequal(loaded_store.data.special, store.data.special)
    end
end
