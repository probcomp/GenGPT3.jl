if GenGPT3.lookup_openai_api_key() == ""
    error("OPENAI_API_KEY not set, GFI tests cannot run.")
end 

@testset "GPT3GenerativeFunction" begin

    gpt3 = GPT3GF(model="davinci-002", max_tokens=64, stop="\n")

    @testset "trace" begin
        prompt = "What is the tallest mountain on Mars?"
        output = "\n\nThe tallest mountain on Mars is Olympus Mons."
        tokens = GenGPT3.tokenize(gpt3.encoding, output)
        logprobs = fill(0.0, length(tokens)) ./ length(tokens)
        score = -27.153727656999997
        trace = GPT3Trace(gpt3, prompt, output, tokens, logprobs, score)

        @test get_choices(trace) == GPT3ChoiceMap(output)
        @test get_args(trace) == (prompt,)
        @test get_retval(trace) == output
        @test get_score(trace) == score
        @test get_gen_fn(trace) == gpt3

        @test trace[:prompt] == prompt
        @test trace[:output] == output
        @test trace[:tokens] == tokens
        @test trace[:token_logprobs] == logprobs
        @test trace[:score] == score
    end

    @testset "simulate" begin
        prompt = "What is the tallest mountain on Mars?"
        trace = simulate(gpt3, (prompt,))

        @test trace.prompt == prompt
        @test trace.output == (trace.tokens[end] == "\n" ? 
            join(trace.tokens[1:end-1]) : join(trace.tokens))
        @test trace.score == sum(trace.logprobs)
        @test length(trace.tokens) == length(trace.logprobs)
    end

    @testset "generate" begin
        # Unconstrained generation
        prompt = "What is the tallest mountain on Mars?"
        trace, weight = generate(gpt3, (prompt,))

        @test trace.prompt == prompt
        @test trace.output == (trace.tokens[end] == "\n" ? 
            join(trace.tokens[1:end-1]) : join(trace.tokens))
        @test trace.score == sum(trace.logprobs)
        @test length(trace.tokens) == length(trace.logprobs)
        @test weight == 0.0

        # Constrained generation
        constraints = choicemap((:output, trace.output))
        new_trace, new_weight = generate(gpt3, (prompt,), constraints)

        @test new_trace.prompt == trace.prompt
        @test new_trace.output == trace.output
        @test new_trace.tokens == trace.tokens
        @test isapprox(new_trace.score, trace.score, atol=1e-2)
        @test all(isapprox.(new_trace.logprobs, trace.logprobs, atol=1e-3))
        @test new_weight == new_trace.score
    end

    @testset "update" begin 
        # Generate initial trace
        prompt = "What is the tallest mountain on Mars?"
        output = "\n\nMount Olympus."
        constraints = choicemap((:output, output))
        trace, _ = generate(gpt3, (prompt,), constraints)

        # Updateless update
        new_trace, weight, retdiff, discard = 
            update(trace, get_args(trace), (UnknownChange(),), choicemap())
        @test new_trace == trace
        @test weight == 0.0
        @test retdiff isa NoChange
        @test isempty(discard)

        # Update output
        new_output = "\n\nOlympus Mons is the tallest mountain on Mars."
        constraints = choicemap((:output, new_output))
        new_trace, weight, retdiff, discard = 
            update(trace, get_args(trace), (NoChange(),), constraints)
        @test weight == get_score(new_trace) - get_score(trace)
        @test retdiff isa UnknownChange
        @test discard[:output] == output

        # Adjust prompt
        new_prompt = "Which mountain do the Greek gods live upon?"
        new_trace, weight, retdiff, discard = 
            update(trace, (new_prompt,), (UnknownChange(),), choicemap())
        @test weight == get_score(new_trace) - get_score(trace)
        @test retdiff isa NoChange
        @test isempty(discard)
    end 

    @testset "regenerate" begin 
        # Generate initial trace
        prompt = "What is the tallest mountain on Mars?"
        output = "\n\nMount Olympus."
        constraints = choicemap((:output, output))
        trace, _ = generate(gpt3, (prompt,), constraints)

        # Regenerate nothing
        new_trace, weight, retdiff = 
            regenerate(trace, get_args(trace), (UnknownChange(),), select())
        @test new_trace == trace
        @test weight == 0.0
        @test retdiff isa NoChange

        # Regenerate output
        new_trace, weight, retdiff = 
            regenerate(trace, get_args(trace), (NoChange(),), select(:output))
        @test weight == 0.0
        @test retdiff isa UnknownChange

        # Adjust prompt
        new_prompt = "Which mountain do the Greek gods live upon?"
        new_trace, weight, retdiff = 
            regenerate(trace, (new_prompt,), (UnknownChange(),), select())
        @test weight == get_score(new_trace) - get_score(trace)
        @test retdiff isa NoChange
    end

end
