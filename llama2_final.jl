using LinearAlgebra
using Mmap

using StatsBase

struct Tokenizer
    id_to_token::Vector{String}
    token_to_id::Dict{String,Int}
    token_scores::Vector{Float32}
end

function Tokenizer(f::IOStream, vocab_size::Int)
    id_to_token = Vector{String}(undef, vocab_size)
    token_to_id = Dict{String,Int}()
    token_scores = Vector{Float32}(undef, vocab_size)
    max_token_length = read(f, Int32)
    for i in 1:vocab_size
        token_scores[i] = read(f, Float32)
        len = read(f, Int32)
        word = String(read(f, len))
        id_to_token[i] = word
        haskey(token_to_id, word) || (token_to_id[word] = i)
    end
    return Tokenizer(id_to_token, token_to_id, token_scores)
end

struct Config
    dim::Int        # transformer dimension
    hidden_dim::Int # for ffn layers
    n_layers::Int   # number of layers
    n_heads::Int    # number of query heads
    n_kv_heads::Int # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::Int # vocabulary size, usually 256 (byte-level) if vocab_size < 0 means to share weights
    seq_len::Int    # max sequence length
    Config(f::IOStream) = new([read(f, Int32) for i in 1:7]...)
end

struct TransformerWeights{T<:AbstractFloat}
    token_embedding_table::Matrix{T}
    rms_att_weight::Matrix{T} # (layer, dim)
    wq::Array{T,3}  # (layer, dim, dim)
    wk::Array{T,3}
    wv::Array{T,3}
    wo::Array{T,3}
    rms_ffn_weight::Matrix{T}
    w1::Array{T,3}  # (layer, hidden_dim, dim)
    w2::Array{T,3}  # (layer, dim, hidden_dim)
    w3::Array{T,3}  # (layer, hidden_dim, dim)
    rms_final_weight::Vector{T} # (dim,)
    freq_cis_real::Matrix{Float32} # (seq_len, dim/2)
    freq_cis_imag::Matrix{Float32} # (seq_len, dim/2)
    wcls::Matrix{T}
end

@kwdef struct RunState{T<:AbstractFloat}
    x::Vector{T}
    xb::Vector{T}
    xb2::Vector{T}
    hb::Vector{T}
    hb2::Vector{T}
    q::Vector{T}
    k::Vector{T}
    v::Vector{T}
    att::Vector{T}
    logits::Vector{T}
    key_cache::Array{T,3}
    value_cache::Array{T,3}
    pos::Int
end

function read_transformer_weights(T, f::IOStream, c::Config)::TransformerWeights{T}
    share_weights = c.vocab_size > 0
    vocab_size = abs(c.vocab_size)
    token_embedding_table = mmap(f, Matrix{T}, (c.dim, vocab_size))
    skip(f, sizeof(token_embedding_table))
    rms_att_weight = mmap(f, Matrix{Float32}, (c.dim, c.n_layers))
    skip(f, sizeof(rms_att_weight))
    wq = reshape(mmap(f, Matrix{T}, (c.dim * c.dim * c.n_layers, 1)), c.dim, c.dim, c.n_layers)
    skip(f, sizeof(wq))
    wk = reshape(mmap(f, Matrix{T}, (c.dim * c.dim * c.n_layers, 1)), c.dim, c.dim, c.n_layers)
    skip(f, sizeof(wk))
    wv = reshape(mmap(f, Matrix{T}, (c.dim * c.dim * c.n_layers, 1)), c.dim, c.dim, c.n_layers)
    skip(f, sizeof(wv))
    wo = reshape(mmap(f, Matrix{T}, (c.dim * c.dim * c.n_layers, 1)), c.dim, c.dim, c.n_layers)
    skip(f, sizeof(wo))
    rms_ffn_weight = mmap(f, Matrix{T}, (c.dim, c.n_layers))
    skip(f, sizeof(rms_ffn_weight))
    w1 = reshape(mmap(f, Matrix{T}, (c.dim * c.hidden_dim * c.n_layers, 1)), c.dim, c.hidden_dim, c.n_layers)
    skip(f, sizeof(w1))
    w2 = reshape(mmap(f, Matrix{T}, (c.hidden_dim * c.dim * c.n_layers, 1)), c.hidden_dim, c.dim, c.n_layers)
    skip(f, sizeof(w2))
    w3 = reshape(mmap(f, Matrix{T}, (c.dim * c.hidden_dim * c.n_layers, 1)), c.dim, c.hidden_dim, c.n_layers)
    skip(f, sizeof(w3))
    rms_final_weight = mmap(f, Vector{T}, (c.dim,))
    skip(f, sizeof(rms_final_weight))
    freq_cis_real = mmap(f, Matrix{Float32}, ((c.dim ÷ c.n_heads) ÷ 2, c.seq_len))
    skip(f, sizeof(freq_cis_real))
    freq_cis_imag = mmap(f, Matrix{Float32}, ((c.dim ÷ c.n_heads) ÷ 2, c.seq_len))
    skip(f, sizeof(freq_cis_imag))
    wcls = share_weights ? token_embedding_table : mmap(f, Matrix{T}, (c.dim, vocab_size))
    TransformerWeights{T}(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, freq_cis_real, freq_cis_imag, wcls)
end

RunState(T, c::Config) = RunState{T}(;
    x=zeros(T, c.dim),
    xb=zeros(T, c.dim),
    xb2=zeros(T, c.dim),
    hb=zeros(T, c.hidden_dim),
    hb2=zeros(T, c.hidden_dim),
    q=zeros(T, c.dim),
    k=zeros(T, c.dim),
    v=zeros(T, c.dim),
    att=zeros(T, c.seq_len),
    logits=zeros(T, abs(c.vocab_size)),
    key_cache=zeros(T, c.dim, c.seq_len, c.n_layers),
    value_cache=zeros(T, c.dim, c.seq_len, c.n_layers),
    pos=1
)


function rmsnorm!(o, x, weight)
    ss = 1.0f0 / √(dot(x, x) / length(x) + 1.0f-5)
    o .= weight .* (ss .* x)
    return nothing
end

function softmax!(x)
    x .= exp.(x .- maximum(x))
    x ./= sum(x)
    return nothing
end

@views function transformer!(token::Int, pos::Int, p::Config, s::RunState, w::TransformerWeights{T}) where {T<:AbstractFloat}
    x = s.x
    dim = p.dim
    hidden_dim = p.hidden_dim
    n_heads = p.n_heads
    head_size = dim ÷ n_heads

    copyto!(x, w.token_embedding_table[:, token])
    freq_cis_real_row = w.freq_cis_real[:, pos]
    freq_cis_imag_row = w.freq_cis_imag[:, pos]
    for l in 1:p.n_layers
        rmsnorm!(s.xb, x, w.rms_att_weight[:, l])
        mul!(s.q, w.wq[:, :, l]', s.xb)
        mul!(s.k, w.wk[:, :, l]', s.xb)
        mul!(s.v, w.wv[:, :, l]', s.xb)

        for h in 1:n_heads
            q = s.q[((h-1)*head_size+1):(h*head_size)]
            k = s.k[((h-1)*head_size+1):(h*head_size)]
            for i in 1:(head_size÷2)
                q0 = q[2*i-1]
                q1 = q[2*i]
                k0 = k[2*i-1]
                k1 = k[2*i]
                fcr = freq_cis_real_row[i]
                fci = freq_cis_imag_row[i]
                q[2*i-1] = q0 * fcr - q1 * fci
                q[2*i] = q0 * fci + q1 * fcr
                k[2*i-1] = k0 * fcr - k1 * fci
                k[2*i] = k0 * fci + k1 * fcr
            end
        end

        copyto!(s.key_cache[:, pos, l], s.k)
        copyto!(s.value_cache[:, pos, l], s.v)

        for h in 1:n_heads
            q = s.q[((h-1)*head_size+1):(h*head_size)]
            for t in 1:pos
                k = s.key_cache[((h-1)*head_size+1):(h*head_size), t, l]
                score = dot(q, k) / sqrt(T(head_size))
                s.att[t] = score
            end
            softmax!(s.att[1:pos])

            mul!(
                s.xb[((h-1)*head_size+1):(h*head_size)],
                s.value_cache[((h-1)*head_size+1):(h*head_size), 1:pos, l],
                s.att[1:pos],
            )
        end

        mul!(s.xb2, w.wo[:, :, l]', s.xb)
        x .+= s.xb2

        rmsnorm!(s.xb, x, w.rms_ffn_weight[:, l])

        mul!(s.hb, w.w1[:, :, l]', s.xb)
        mul!(s.hb2, w.w3[:, :, l]', s.xb)
        @simd for i in 1:hidden_dim
            s.hb[i] = s.hb[i] * (1.0f0 / (1.0f0 + exp(-s.hb[i])))
        end
        s.hb .*= s.hb2
        mul!(s.xb, w.w2[:, :, l]', s.hb)

        x .+= s.xb
    end

    rmsnorm!(x, x, w.rms_final_weight)
    mul!(s.logits, w.wcls', x)
end

function bpe_encode(text::String, tokenizer::Tokenizer)
    tokens = Int[]
    for c in text
        id = get(tokenizer.token_to_id, string(c), nothing)
        isnothing(id) && (println("character $c not in vocab"); exit(1))
        push!(tokens, id)
    end
    while true
        best_score = -Inf32
        best_id, best_idx = -1, -1
        for i in 1:(length(tokens)-1)
            id = get(tokenizer.token_to_id, tokenizer.id_to_token[tokens[i]] * tokenizer.id_to_token[tokens[i+1]], nothing)
            if !isnothing(id) && tokenizer.token_scores[id] > best_score
                best_score = tokenizer.token_scores[id]
                best_id = id
                best_idx = i
            end
        end
        best_id == -1 && break
        tokens[best_idx] = best_id
        deleteat!(tokens, best_idx + 1)
    end
    return tokens
end

function forward(weights::TransformerWeights{T}, tokenizer::Tokenizer, config::Config, prompt::String, temperature::Float32, steps::Int) where {T<:AbstractFloat}
    state = RunState(T, config)
    prompt_tokens = bpe_encode(prompt, tokenizer)
    token = 2 # beginning of sentence token id, 1 is <unk>
    next = 2
    for pos in 1:min(steps, config.seq_len)
        print(tokenizer.id_to_token[next])
        token = next
        next == 3 && pos == steps && break # end of sequence
        transformer!(token, pos, config, state, weights)
        pos <= length(prompt_tokens) && (next = prompt_tokens[pos]; continue)
        temperature == 0.0f0 && (next = argmax(state.logits); continue)
        state.logits ./= temperature
        softmax!(state.logits)
        next = wsample(1:abs(config.vocab_size), state.logits)
    end
end

function main(T, checkpoint_filenames::AbstractString, tokenizer_filename::AbstractString;)
    config, weights, tokenizer = nothing, nothing, nothing
    open(checkpoint_filenames, "r") do file
        config = Config(file)
        weights = read_transformer_weights(T, file, config)
    end
    open(tokenizer_filename, "r") do file
        tokenizer = Tokenizer(file, abs(config.vocab_size))
    end
    @show config
    forward(weights, tokenizer, config, "I want to", 0.0f0, 256)
end

main(Float32, "llama2_7b.bin", "tokenizer.bin")