## GPT-3 Tokenizer ##

using BytePairEncoding
using TextEncodeBase
using JSON3

import BytePairEncoding: GPT2Tokenization, gpt2_codemap
import TextEncodeBase: FlatTokenizer, CodeNormalizer, Sentence, codeunmap

const ARTIFACT_DIR = normpath(dirname(@__DIR__), "artifacts")

const GPT_VOCAB_DICT = JSON3.read(read(joinpath(ARTIFACT_DIR, "vocab.json"), String))
const GPT_VOCAB_LIST = [string(s) for s in keys(GPT_VOCAB_DICT)]

const GPT_BPE = BPE(joinpath(ARTIFACT_DIR, "bpe.txt"))
const GPT_CODEMAP = gpt2_codemap()
const GPT_TOKENIZER = FlatTokenizer(CodeNormalizer(BPETokenization(GPT2Tokenization(), GPT_BPE), GPT_CODEMAP))

const GPT_EOT_ID = 50256

"Encodes a sequence of GPT-2 / GPT-3 tokens into integer token IDs."
function encode(tokens::AbstractVector{<:AbstractString})
    return map(t -> GPT_VOCAB_DICT[t]::Int, tokens)
end

"Decodes a sequence of GPT-2 / GPT-3 token IDs into string tokens."
function decode(token_ids::AbstractVector{<:Integer})
    return map(i -> GPT_VOCAB_LIST[i+1]::String, token_ids)
end

"Tokenizes a string into a sequence of GPT-2 / GPT-3 string tokens."
function tokenize(str::AbstractString)
    return map(t -> t.x::String, GPT_TOKENIZER(Sentence(str)))
end

"Detokenizes a sequence of GPT-2 / GPT-3 string tokens into a string."
function detokenize(tokens::AbstractVector{<:AbstractString})
    return join(map(t -> codeunmap(GPT_CODEMAP, t), tokens))
end

"Tokenizes a string into a sequence of GPT-2 / GPT-3 token IDs."
function id_tokenize(str::AbstractString)
    return encode(tokenize(str))
end

"Detokenizes a sequence of GPT-2 / GPT-3 token IDs into a string."
function id_detokenize(token_ids::AbstractVector{<:Integer})
    return detokenize(decode(token_ids))
end
