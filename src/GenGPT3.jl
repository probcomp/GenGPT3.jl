module GenGPT3

using Gen
using JSON3
using HTTP
using Tables
using TypedTables
using JSONTables
using Base: @kwdef

import Gen:
    get_choices, get_args, get_retval, get_score, get_gen_fn,
    simulate, generate, project, assess, update, regenerate

export GPT3ChoiceMap, GPT3Trace
export GPT3GenerativeFunction, GPT3GF
export MultiGPT3ChoiceMap, MultiGPT3Trace
export MultiGPT3GenerativeFunction, MultiGPT3GF
export GPT3MixtureTrace, GPT3Mixture
export GPT3ImportanceSamplerTrace, GPT3ISTrace
export GPT3ImportanceSampler, GPT3IS

include("tokenizer.jl")
include("web_api.jl")
include("gen_fn.jl")
include("multi_fn.jl")
include("mixture.jl")
include("importance.jl")
include("embedder.jl")

end
