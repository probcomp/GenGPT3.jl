if GenGPT3.lookup_openai_api_key() == ""
    error("OPENAI_API_KEY not set, GFI tests cannot run.")
end 

@testset "MultiGPT3GenerativeFunction" begin

    multi_gpt3 = MultiGPT3GF(model="text-babbage-001", max_tokens=256)

    @testset "trace" begin
        prompts = fill("What is the tallest mountain on Mars?", 2)
        outputs = fill("\n\nThe tallest mountain on Mars is Olympus Mons.", 2)
        scores = fill(-5.61428454404, 2)
        score = sum(scores)
        trace = MultiGPT3Trace(multi_gpt3, prompts, outputs,
                               Vector{String}[], Vector{Float64}[],
                               scores, score)

        @test get_args(trace) == (prompts,)
        @test get_retval(trace) == outputs
        @test get_score(trace) == score
        @test get_gen_fn(trace) == multi_gpt3

        @test trace[1 => :output] == outputs[1]
        @test trace[2 => :output] == outputs[2]
    end

    @testset "simulate" begin
        prompts = fill("What is the tallest mountain on Mars?", 2)
        trace = Gen.simulate(multi_gpt3, (prompts,))

        @test trace.prompts == prompts
        @test trace.outputs == [join(tks[1:end-1]) for tks in trace.tokens]
        @test trace.scores == [sum(logprobs) for logprobs in trace.logprobs]
        @test all(length.(trace.tokens) .== length.(trace.logprobs))
        @test trace.score == sum(trace.scores)

        trace = Gen.simulate(multi_gpt3, (2, prompts[1]))
        @test trace.prompts == prompts
        @test trace.outputs == [join(tks[1:end-1]) for tks in trace.tokens]
        @test trace.scores == [sum(logprobs) for logprobs in trace.logprobs]
        @test all(length.(trace.tokens) .== length.(trace.logprobs))
        @test trace.score == sum(trace.scores)
    end

    @testset "generate" begin
        # Unconstrained generation
        prompts = fill("What is the tallest mountain on Mars?", 2)
        trace, weight = Gen.generate(multi_gpt3, (prompts,))

        @test trace.prompts == prompts
        @test trace.outputs == [join(tks[1:end-1]) for tks in trace.tokens]
        @test trace.scores == [sum(logprobs) for logprobs in trace.logprobs]
        @test all(length.(trace.tokens) .== length.(trace.logprobs))
        @test trace.score == sum(trace.scores)
        @test weight == 0.0

        trace, weight = Gen.generate(multi_gpt3, (2, prompts[1]))
        @test trace.prompts == prompts
        @test trace.outputs == [join(tks[1:end-1]) for tks in trace.tokens]
        @test trace.scores == [sum(logprobs) for logprobs in trace.logprobs]
        @test all(length.(trace.tokens) .== length.(trace.logprobs))
        @test trace.score == sum(trace.scores)
        @test weight == 0.0

        # Constrained generation
        constraints = choicemap((1 => :output, trace.outputs[1]),
                                (2 => :output, trace.outputs[2]))
        new_trace, new_weight =
            Gen.generate(multi_gpt3, (prompts,), constraints)

        @test new_trace.prompts == trace.prompts
        @test new_trace.outputs == trace.outputs
        @test new_trace.tokens == trace.tokens
        @test all(isapprox.(new_trace.scores, trace.scores, atol=1e-1))
        @test all(isapprox.(new_trace.logprobs, trace.logprobs, atol=1e-1))
        @test new_weight == new_trace.score

        new_trace, new_weight =
            Gen.generate(multi_gpt3, (2, prompts[1]), constraints)
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
        trace, _ = Gen.generate(multi_gpt3, (prompts,), constraints)

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
        trace, _ = Gen.generate(multi_gpt3, (prompts,), constraints)

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
