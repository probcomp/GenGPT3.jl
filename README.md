# GenGPT3.jl

[GPT-3](https://en.wikipedia.org/wiki/GPT-3) as a generative function in [Gen.jl](https://www.gen.dev/), implemented by wrapping the [OpenAI API](https://openai.com/api/) in Gen's interface.

## Usage

Install both Gen and this package via the Julia Pkg REPL:

```
add Gen
add https://github.com/probcomp/GenGPT3.jl.git
```

Add your OpenAI API key as an environment variable named `OPENAI_API_KEY`. You can follow [this guide](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety), or set `ENV["OPENAI_API_KEY"]` to the value of your API key in the Julia REPL.

Now you can construct GPT-3 as a generative function, and call GFI functions on it:

```julia
using Gen, GenGPT3

# Construct GPT3GenerativeFunction
gpt3 = GPT3GF(model="davinci-002", max_tokens=256)

# Untraced execution 
prompt = "What is the tallest mountain on Mars?"
output = gpt3(prompt)

# Traced execution
trace = simulate(gpt3, (prompt,))

# Constrained generation
constraints = choicemap((:output, "Olympus Mons."))
trace, weight = generate(gpt3, (prompt,), constraints)
```

## Configuration

The constructor for a `GPT3GenerativeFunction` (or `GPT3GF` for short), can be used to configure a variety of options, documented below:

```julia
GPT3GenerativeFunction(;
    model = "davinci-002",
    temperature = 1.0,
    max_tokens = 1024,
    stop = nothing,
    encoding = GenGPT3.MODEL_ENCODINGS[model],
    api_key_lookup = () -> ENV["OPENAI_API_KEY"],
    organization_lookup = () -> ENV["OPENAI_ORGANIZATION"]
)
```

Constructs GPT-3 as a generative function, where sampling and scoring of completions are performed via calls to the OpenAI API.

The generative function takes in a prompt as an (optional) argument, then samples and returns a completion. This represents a distribution over strings (up to `max_tokens` long) which end in a `stop` sequence. The completion is stored in the `:output` address of the resulting trace.

### Arguments
- `model::String`:
    The pretrained model to query. Defaults to `"davinci-002"`.
- `temperature::Float64 = 1.0`:
    The softmax temperature. Values between `0.0` and `2.0` are allowed.
    Higher temperatures increase randomness. Note that if this is not set
    to `1.0`, then the resulting log probabilities will no longer be normalized.
- `max_tokens::Int = 1024`:
    The maximum number of output tokens generated (including the stop sequence).
- `encoding::String = GenGPT3.MODEL_ENCODINGS[model]`:
    Tokenizer encoding for the model. For most models, this is `"cl100k_base"`.
- `stop::Union{String,Nothing} = nothing`:
    The stop sequence as a string. Defaults to the `<|endoftext|>` token if not
    specified. If specified, then the model will be prevented from generating
    any `<|endoftext|>` tokens (to avoid multiple termination possibilities).
- `api_key_lookup::Function`:
    A zero-argument function that returns the OpenAI API key. Defaults to
    looking up the `"OPENAI_API_KEY"` environment variable.
- `organization_lookup::Function`:
    A zero-argument function that returns the OpenAI organization ID to use.
    Defaults to the `"OPENAI_ORGANIZATION"` environment variable, if specified.

### Variants

GenGPT3.jl also provides support for batched LLM calls with [`MultiGPT3GF`](src/multi_fn.jl) and a "mixture-of-prompts" generative function called [`GPT3Mixture`](src/mixture.jl). `MultiGPT3GF` reduces the latency of making multiple API calls, and `GPT3Mixture` (which uses `MultiGPT3GF` under the hood) can be used to marginalize over uncertainty about the prompt.

Support for batched importance sampling of LLM generations is provided by [`GPT3ImportanceSampler`](src/importance.jl), which can be configured to have 
separate model and proposal LLM generative functions.

## Utilities

Utilities for converting between strings and tokens are also included as part of this package (using functionality provided by [BytePairEncoding.jl](https://github.com/chengchingwen/BytePairEncoding.jl) and [TextEncodeBase.jl](https://github.com/chengchingwen/TextEncodeBase.jl)):

```julia-repl
julia> tokens = GenGPT3.tokenize("cl100k_base", "What is the tallest mountain on Mars?")
["What", " is", " the", " tallest", " mountain", " on", " Mars", "?"]

julia> ids = GenGPT3.encode("cl100k_base", tokens)
[3923, 374, 279, 82717, 16700, 389, 21725, 30]

julia> text = GenGPT3.id_detokenize("cl100k_base", ids)
"What is the tallest mountain on Mars?"

julia> ids = GenGPT3.id_tokenize("cl100k_base", text)
[3923, 374, 279, 82717, 16700, 389, 21725, 30]

julia> tokens = GenGPT3.decode("cl100k_base", ids)
["What", " is", " the", " tallest", " mountain", " on", " Mars", "?"]
```

Support for calling the OpenAI Embeddings API is also provided:

```julia-repl
julia> embedder = GenGPT3.Embedder(model="text-embedding-ada-002");

julia> embedder("What is the tallest mountain on Mars?")
1536-element Vector{Float64}:
  0.009133313
 -0.0017141271
 -0.010894737
  ⋮
 -0.0018739601
 -0.013621684
 -0.037864104
```
