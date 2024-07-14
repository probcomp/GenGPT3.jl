## GPT-3 Tokenizer ##

using TextEncodeBase
using BytePairEncoding

using TextEncodeBase: AbstractTokenization, DATLookupDict, lookup_word
using BytePairEncoding: BPETokenizer, BPETokenization, TikTokenBPE, StringView

const CL100K_BASE = BytePairEncoding.load_tiktoken("cl100k_base")
const P50K_BASE = BytePairEncoding.load_tiktoken("p50k_base")
const R50K_BASE = BytePairEncoding.load_tiktoken("r50k_base")

const TOKENIZERS = Dict(
    "cl100k_base" => CL100K_BASE,
    "p50k_base" => P50K_BASE,
    "r50k_base" => R50K_BASE
)

const ENCODERS = Dict(
    name => tkr.tokenization.base.bpe.encoder for (name, tkr) in TOKENIZERS
)

const EOT_IDS = Dict(
    "cl100k_base" => 100257,
    "p50k_base" => 50256,
    "r50k_base" => 50256
)

const SPECIAL_TOKENS_ENCODER = Dict(
    "cl100k_base" => Dict(
        "<|endoftext|>" => 100257,
        "<|fim_prefix|>" => 100258,
        "<|fim_middle|>" => 100259,
        "<|fim_suffix|>" => 100260,
        "<|endofprompt|>" => 100276
    ),
    "p50k_base" => Dict(
        "<|endoftext|>" => 50256
    ),
    "r50k_base" => Dict(
        "<|endoftext|>" => 50256
    )
)

const SPECIAL_TOKENS_DECODER = Dict(
    name => Dict(id => str for (str, id) in tokens)
    for (name, tokens) in SPECIAL_TOKENS_ENCODER
)

const MODEL_ENCODINGS = Dict(
    "gpt-4" => "cl100k_base",
    "gpt-3.5-turbo" => "cl100k_base",
    "gpt-3.5" => "cl100k_base",
    "gpt-3.5-turbo-instruct" => "cl100k_base",
    "gpt-35-turbo" => "cl100k_base",
    # Current base models
    "davinci-002" => "cl100k_base",
    "babbage-002" => "cl100k_base",
    # Embeddings
    "text-embedding-ada-002" => "cl100k_base",
    "text-embedding-3-small" => "cl100k_base",
    "text-embedding-3-large" => "cl100k_base",
    # Deprecated GPT-3.5 models
    "text-davinci-003" => "p50k_base",
    "text-davinci-002" => "p50k_base",
    "text-davinci-001" => "r50k_base",
    "text-curie-001" => "r50k_base",
    "text-babbage-001" => "r50k_base",
    "text-ada-001" => "r50k_base",
    # Deprecated GPT-3 base models
    "davinci" => "r50k_base",
    "curie" => "r50k_base",
    "babbage" => "r50k_base",
    "ada" => "r50k_base"
)

"""
    encode(encoding, tokens)

Encodes a sequence of string tokens into integer token IDs. Supported values
for `encoding` include `"cl100k_base"`, `"p50k_base"` and `"r50k_base"`. 
"""
function encode(name::AbstractString, tokens::AbstractVector{<:AbstractString})
    encoder = ENCODERS[name]
    special = SPECIAL_TOKENS_ENCODER[name]
    return map(t -> get(() -> get(encoder, t, -1) - 1, special, t), tokens)
end
encode(t::BPETokenizer, tokens::AbstractVector{<:AbstractString}) =
    encode(TextEncodeBase.tokenization(t), tokens)
encode(t::AbstractTokenization, tokens::AbstractVector{<:AbstractString}) =
    encode(TextEncodeBase.base(t), tokens)
encode(t::BPETokenization, tokens::AbstractVector{<:AbstractString}) =
    encode(t.bpe, tokens)
encode(bpe::TikTokenBPE, tokens::AbstractVector{<:AbstractString}) =
    map(t -> get(bpe.encoder, t, -1), tokens)

"""
    decode(encoding, token_ids)

Decodes a sequence of token IDs into string tokens. Supported values for
`encoding` include `"cl100k_base"`, `"p50k_base"` and `"r50k_base"`.
"""
function decode(name::AbstractString, token_ids::AbstractVector{<:Integer})
    encoder = ENCODERS[name]
    special = SPECIAL_TOKENS_DECODER[name]
    return map(token_ids) do id
        get(special, id) do 
            TextEncodeBase.lookup_word(encoder, "", id + 1)
        end
    end
end

"""
    tokenize(tokenizer, str)

Tokenizes a string into a sequence of string tokens. Supported values for
`tokenizer` include `"cl100k_base"`, `"p50k_base"` and `"r50k_base"`.
"""
function tokenize(name::AbstractString, str::AbstractString)
    tokenizer = TOKENIZERS[name]
    return tokenize(tokenizer, str)
end
tokenize(tokenizer::BPETokenizer, str::AbstractString) = tokenizer(str)

"""
    detokenize([tokenizer,] tokens)

Detokenizes a sequence of string tokens into a string.
"""
function detokenize(tokens::AbstractVector{<:AbstractString})
    return join(tokens)
end
detokenize(name::AbstractString, tokens::AbstractVector{<:AbstractString}) =
    detokenize(tokens)
detokenize(tokenizer::BPETokenizer, tokens::AbstractVector{<:AbstractString}) =
    detokenize(tokens)

"""
    id_tokenize(tokenizer, str)

Tokenizes a string into a sequence of token IDs. Supported values for
`tokenizer` include `"cl100k_base"`, `"p50k_base"` and `"r50k_base"`.
"""
function id_tokenize(name::AbstractString, str::AbstractString)
    tokenizer = TOKENIZERS[name]
    tokens = tokenize(tokenizer, str)
    return encode(name, tokens)
end
id_tokenize(tokenizer::BPETokenizer, str::AbstractString) =
    encode(tokenizer, tokenize(tokenizer, str))

"""
    id_detokenize(tokenizer, token_ids)

Detokenizes a sequence of token IDs into a string. Supported values for
`tokenizer` include `"cl100k_base"`, `"p50k_base"` and `"r50k_base"`.
"""
function id_detokenize(name::AbstractString, token_ids::AbstractVector{<:Integer})
    return detokenize(decode(name, token_ids))
end
