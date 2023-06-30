@testset "Tokenizer" begin

    input = "What is the tallest mountain on Mars?"
    tokens = ["What", "Ġis", "Ġthe", "Ġtallest", "Ġmountain", "Ġon", "ĠMars", "?"]
    unnorm_tokens = ["What", " is", " the", " tallest", " mountain", " on", " Mars", "?"]
    ids = [2061, 318, 262, 38760, 8598, 319, 8706, 30]

    @test GenGPT3.encode(tokens) == ids
    @test GenGPT3.encode(unnorm_tokens, normalized=false) == ids
    @test GenGPT3.decode(ids) == tokens
    @test GenGPT3.decode(ids, normalized=false) == unnorm_tokens

    @test GenGPT3.tokenize(input) == tokens
    @test GenGPT3.tokenize(input, normalized=false) == unnorm_tokens
    @test GenGPT3.detokenize(tokens) == input
    @test GenGPT3.detokenize(unnorm_tokens, normalized=false) == input

    @test GenGPT3.id_tokenize(input) == ids
    @test GenGPT3.id_detokenize(ids) == input
    
end
