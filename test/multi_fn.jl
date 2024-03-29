if GenGPT3.lookup_openai_api_key() == ""
    error("OPENAI_API_KEY not set, GFI tests cannot run.")
end 

@testset "MultiGPT3GenerativeFunction" begin

    multi_gpt3 = MultiGPT3GF(model="davinci-002", max_tokens=64, stop="\n")

    @testset "trace" begin
        prompts = fill("What is the tallest mountain on Mars?", 2)
        outputs = fill("\n\nThe tallest mountain on Mars is Olympus Mons.", 2)
        tokens = [GenGPT3.tokenize(multi_gpt3.encoding, o) for o in outputs]
        logprobs = [fill(0.0, length(t)) for t in tokens]
        scores = fill(-27.153727656999997, 2)
        score = sum(scores)
        trace = MultiGPT3Trace(multi_gpt3, prompts, outputs,
                               tokens, logprobs, scores, score)

        @test get_args(trace) == (prompts,)
        @test get_retval(trace) == outputs
        @test get_score(trace) == score
        @test get_gen_fn(trace) == multi_gpt3

        @test trace[:prompts] == prompts
        @test trace[:outputs] == outputs
        @test trace[:tokens] == tokens
        @test trace[:token_logprobs] == logprobs
        @test trace[:output_scores] == scores
        @test trace[:score] == score

        @test trace[1 => :prompt] == prompts[1]
        @test trace[1 => :output] == outputs[1]
        @test trace[1 => :tokens] == tokens[1]
        @test trace[1 => :token_logprobs] == logprobs[1]
        @test trace[1 => :score] == scores[1]
        @test trace[2 => :prompt] == prompts[2]
        @test trace[2 => :output] == outputs[2]
        @test trace[2 => :tokens] == tokens[2]
        @test trace[2 => :token_logprobs] == logprobs[2]
        @test trace[2 => :score] == scores[2]
    end

    @testset "simulate" begin
        prompts = fill("What is the tallest mountain on Mars?", 2)
        trace = simulate(multi_gpt3, (prompts,))

        @test trace.prompts == prompts
        @test trace.outputs == [tks[end] == "\n" ? join(tks[1:end-1]) : join(tks)
                                for tks in trace.tokens]
        @test trace.scores == [sum(logprobs) for logprobs in trace.logprobs]
        @test all(length.(trace.tokens) .== length.(trace.logprobs))
        @test trace.score == sum(trace.scores)

        trace = simulate(multi_gpt3, (2, prompts[1]))
        @test trace.prompts == prompts
        @test trace.outputs == [tks[end] == "\n" ? join(tks[1:end-1]) : join(tks)
                                for tks in trace.tokens]
        @test trace.scores == [sum(logprobs) for logprobs in trace.logprobs]
        @test all(length.(trace.tokens) .== length.(trace.logprobs))
        @test trace.score == sum(trace.scores)
    end

    @testset "generate" begin
        # Unconstrained generation
        prompts = fill("What is the tallest mountain on Mars?", 2)
        trace, weight = generate(multi_gpt3, (prompts,))

        @test trace.prompts == prompts
        @test trace.outputs == [tks[end] == "\n" ? join(tks[1:end-1]) : join(tks)
                                for tks in trace.tokens]
        @test trace.scores == [sum(logprobs) for logprobs in trace.logprobs]
        @test all(length.(trace.tokens) .== length.(trace.logprobs))
        @test trace.score == sum(trace.scores)
        @test weight == 0.0

        trace, weight = generate(multi_gpt3, (2, prompts[1]))
        @test trace.prompts == prompts
        @test trace.outputs == [tks[end] == "\n" ? join(tks[1:end-1]) : join(tks)
                                for tks in trace.tokens]
        @test trace.scores == [sum(logprobs) for logprobs in trace.logprobs]
        @test all(length.(trace.tokens) .== length.(trace.logprobs))
        @test trace.score == sum(trace.scores)
        @test weight == 0.0

        # Constrained generation
        constraints = choicemap((1 => :output, trace.outputs[1]),
                                (2 => :output, trace.outputs[2]))
        new_trace, new_weight =
            generate(multi_gpt3, (prompts,), constraints)

        @test new_trace.prompts == trace.prompts
        @test new_trace.outputs == trace.outputs
        @test new_trace.tokens == trace.tokens
        @test all(isapprox.(new_trace.scores, trace.scores, atol=1e-1))
        @test all(isapprox.(new_trace.logprobs, trace.logprobs, atol=1e-1))
        @test new_weight == new_trace.score

        new_trace, new_weight =
            generate(multi_gpt3, (2, prompts[1]), constraints)
        @test new_trace.prompts == trace.prompts
        @test new_trace.outputs == trace.outputs
        @test new_trace.tokens == trace.tokens
        @test all(isapprox.(new_trace.scores, trace.scores, atol=1e-1))
        @test all(isapprox.(new_trace.logprobs, trace.logprobs, atol=1e-1))
        @test new_weight == new_trace.score
    end

    @testset "update" begin 
        # Generate initial trace
        prompts = fill("What is the tallest mountain on Mars?", 2)
        outputs = fill("\n\nMount Olympus.", 2)
        constraints = MultiGPT3ChoiceMap(outputs)
        trace, _ = generate(multi_gpt3, (prompts,), constraints)

        # Updateless update
        new_trace, weight, retdiff, discard = 
            update(trace, get_args(trace), (UnknownChange(),), choicemap())
        @test new_trace == trace
        @test weight == 0.0
        @test retdiff isa NoChange
        @test isempty(discard)

        # Update output
        new_output = "\n\nOlympus Mons is the tallest mountain on Mars."
        constraints = choicemap((1 => :output, new_output))
        new_trace, weight, retdiff, discard = 
            update(trace, get_args(trace), (NoChange(),), constraints)
        @test weight ≈ get_score(new_trace) - get_score(trace)
        @test weight ≈ new_trace.scores[1] - trace.scores[1]
        @test retdiff isa UnknownChange
        @test discard[1 => :output] == outputs[1]

        # Adjust prompt
        new_prompt = "Which mountain do the Greek gods live upon?"
        new_prompts = [prompts[1], new_prompt] 
        new_trace, weight, retdiff, discard = 
            update(trace, (new_prompts,), (UnknownChange(),), choicemap())
        @test weight ≈ get_score(new_trace) - get_score(trace)
        @test weight ≈ new_trace.scores[2] - trace.scores[2]
        @test retdiff isa NoChange
        @test isempty(discard)
    end 

    @testset "regenerate" begin 
        # Generate initial trace
        prompts = fill("What is the tallest mountain on Mars?", 2)
        outputs = fill("\n\nMount Olympus.", 2)
        constraints = MultiGPT3ChoiceMap(outputs)
        trace, _ = generate(multi_gpt3, (prompts,), constraints)

        # Regenerate nothing
        new_trace, weight, retdiff = 
            regenerate(trace, get_args(trace), (UnknownChange(),), select())
        @test new_trace == trace
        @test weight == 0.0
        @test retdiff isa NoChange

        # Regenerate output
        selection = select(1 => :output)
        new_trace, weight, retdiff = 
            regenerate(trace, get_args(trace), (NoChange(),), selection)
        @test weight == 0.0
        @test retdiff isa UnknownChange

        # Adjust prompt
        new_prompt = "Which mountain do the Greek gods live upon?"
        new_prompts = [prompts[1], new_prompt] 
        new_trace, weight, retdiff = 
            regenerate(trace, (new_prompts,), (UnknownChange(),), select())
        @test weight ≈ get_score(new_trace) - get_score(trace)
        @test weight ≈ new_trace.scores[2] - trace.scores[2]
        @test retdiff isa NoChange
    end

end
