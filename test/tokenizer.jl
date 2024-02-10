@testset "Tokenizer" begin

    input = "What is the tallest mountain on Mars?<|endoftext|>"
    tokens = ["What", " is", " the", " tallest", " mountain", " on", " Mars", "?", "<|endoftext|>"]
    p50k_ids = [2061, 318, 262, 38760, 8598, 319, 8706, 30, 50256]
    cl100k_ids = [3923, 374, 279, 82717, 16700, 389, 21725, 30, 100257]

    @test GenGPT3.encode("p50k_base", tokens) == p50k_ids
    @test GenGPT3.decode("p50k_base", p50k_ids) == tokens
    @test GenGPT3.encode("cl100k_base", tokens) == cl100k_ids
    @test GenGPT3.decode("cl100k_base", cl100k_ids) == tokens

    @test GenGPT3.tokenize("p50k_base", input) == tokens
    @test GenGPT3.tokenize("cl100k_base", input) == tokens
    @test GenGPT3.detokenize(tokens) == input
    @test GenGPT3.detokenize("p50k_base", tokens) == input
    @test GenGPT3.detokenize("cl100k_base", tokens) == input

    @test GenGPT3.id_tokenize("p50k_base", input) == p50k_ids
    @test GenGPT3.id_tokenize("cl100k_base", input) == cl100k_ids
    @test GenGPT3.id_detokenize("p50k_base", p50k_ids) == input
    @test GenGPT3.id_detokenize("cl100k_base", cl100k_ids) == input
    
end
