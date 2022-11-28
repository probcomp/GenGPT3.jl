@testset "Tokenizer" begin

    input = "What is the tallest mountain on Mars?"
    tokens = ["What", "Ġis", "Ġthe", "Ġtallest", "Ġmountain", "Ġon", "ĠMars", "?"]
    ids = [2061, 318, 262, 38760, 8598, 319, 8706, 30]

    @test GenGPT3.encode(tokens) == ids
    @test GenGPT3.decode(ids) == tokens

    @test GenGPT3.tokenize(input) == tokens
    @test GenGPT3.detokenize(tokens) == input

    @test GenGPT3.id_tokenize(input) == ids
    @test GenGPT3.id_detokenize(ids) == input
    
end
