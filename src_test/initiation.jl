# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.2                  # 2022-10-28
# Copyright (c) 2020-2023 Moscow Institute of Physics and Technology, Laboratory of Cognitive architectures


function initiate!(a::Adam, p::Dict, data::Vector{Tuple{Int, Vector{F}, R}} =
                                           Tuple{Int, Vector{F}, R}[])::Nothing
    a.app = p
    if (haskey(p, "step_reward"))
        psr = p["step_reward"]
        if (typeof(psr) <: Vector)
            a.step_reward = map(x -> R(x), psr)
            a.layers[1].step_reward = a.step_reward[1]
        elseif typeof(psr) <: Number
            a.layers[1].step_reward = R(psr)
            a.step_reward = [a.layers[1].step_reward]
        end
    end
    isempty(data) && return a

    count_pairs::Bool    = get(p, "init_pair_counts"    , true)
    init_counts::Int     = get(p, "init_counts"         , 1)
    init_layer_memory    = get(p, "init_layer_memory"   , false)
    init_values::Bool    = get(p, "init_values"         , true)
    keep_init_data::Bool = get(p, "keep_init_data"      , false)
    reparse::Int         = get(p, "init_reparse"        , 0)
    weights::Vector{F}   = get(p, "input_weights"       , F[])

    (a.scaler, a.rescaler, a.input_weights) = new_scaler(data, weights)
    a.d[1] = length(a.scaler)

    l = a.layers[1]
    l.episodic = new_episodic(data, a.scaler)
    implant_coder!(l, new_coders(l.episodic.x, a.coder[1], a.layers[1].K))
    catch_counter = encode_episodic!(l)
    @assert catch_counter < 1

    if (count_pairs)
        count_pairs!(l)
    end

    if (init_values)
        init_values!(l)
        if (init_counts > 0)
            init_value_update_counts!(l, init_counts)
        end
    end

    if (init_layer_memory)
        init_layer_memory!(l)
    end

    if (reparse > 0)
        [reparse!(l) for _ ∈ 1 : reparse]
    end

    if (!keep_init_data)
        a.layers[1].episodic = Episodic()
    end

    return nothing
end

""" Explore data and create scaling and rescaling vectors """
function new_scaler(d::Vector{Tuple{Int, Vector{F}, R}}, weights::Vector{F}=F[]
                    )::Tuple{Vector{F}, Vector{F}, Vector{F}}
    @assert !isempty(d)
    scaler::Vector{F}   = F[]
    rescaler::Vector{F} = F[]
    isempty(d) && return (scaler, rescaler, weights)

    d_::typeof(d) = filter(x -> ((x[1] > typemin(I)) && !isempty(x[2])), d)
    dim::Int = length(d_[1][2]) + 2
    w::Vector{F} = []
    w = length(weights) == dim - 1 ? [weights; [F(0.)]] :
        length(weights) != dim ? fill(F(1.), dim) : weights
    a = unique(map(x -> x[1], d_))
    a_max = max(maximum(a), -minimum(a))
    M = hcat(map(x -> x[2], d_)...)
    dim = size(M)[1]

    s_max = map(x -> x[1], [findmax(M[i,:]) for i=1:dim])
    s_min = map(x -> x[1], [findmin(M[i,:]) for i=1:dim])
    r = map(x -> x[3], d_)
    max_r = max(maximum(r), -minimum(r))
    abs_max = [[F(a_max)]; [max(s_max[i], -s_min[i]) for i=1:dim]; max_r]

    scaler = w ./ map(x -> (x == 0. ? 1. : x), abs_max)
    rescaler = map(x -> (x == 0. ? 0. : 1. / x), scaler)
    return (scaler, rescaler, w)
end

""" Fill exploration data in episodic memory """
function new_episodic(d::Vector{Tuple{Int, Vector{F}, R}},
                      scaler::Vector{F})::Episodic
    e = Episodic()
    e.length = length(d)
    e.r = map(x -> x[end], d)
    e.t = [T(i) for i=1:e.length]
    for i in findall(x -> x[1] == typemin(I) || isempty(x[2]), d)
        e.t[i] = T(0.)
    end
    e.x = vectorise(d, scaler)
    return e
end

function vectorise(d::Vector{Tuple{Int, Vector{F}, R}}, scaler::Vector{F}
                   )::Vector{Vector{F}}
    map(x -> vectorise(x, scaler, length(scaler) - 2), d)
end

function vectorise(tick::Tuple{Int, Vector{F}, R}, scaler::Vector{F}, dim::Int
                   )::Vector{F}
    (tick[1] > typemin(I)) && (length(tick[2]) == dim) ?
    [[F(tick[1])]; tick[2]; [tick[3]]] .* scaler : []
end

""" Encode raw sensor vector data in episodic memory """
function encode_episodic!(l::Layer)::Int
    e   = l.episodic
    e.i = zeros(I, e.length)
    e.m = zeros(I, e.length)
    resize!(e.s, e.length)
    resize!(e.w, e.length)
    catch_counter::Int = 0
    for t = 1:length(e.x)
        if (e.t[t] > T(0)) && (!isempty(e.x[t]))
            try
                s = encode(e.x[t], l)
                (i, m, w) = encode!(l, s)
                e.i[t] = i
                e.m[t] = m
                e.s[t] = s
                e.w[t] = w
            catch
                e.s[t] = M[]
                e.w[t] = M[]
                catch_counter += 1
            end
        else
            e.s[t] = M[]
            e.t[t] = T(0)
            e.w[t] = M[]
            e.x[t] = F[]
        end
    end
    return catch_counter
end

""" Initialise symbol pair counts - Head.R, Head.Rt for heads in Layer.semantic """
function count_pairs!(l::Layer)::Nothing
    w̄::Vector{Vector{M}} = l.episodic.w
    for t ∈ 1 : length(w̄) - 1
        if (!isempty(w̄[t]) && !isempty(w̄[t+1]))
            update_count!(l, w̄[t], w̄[t+1])
        end
    end
    return nothing
end

""" Initialise symbol values based on environment exploration data """
function init_values!(l::Layer)::Nothing
    e::Episodic = l.episodic
    # The last morpheme in episodic memory: update value if valid
    if (e.t[end] > 0) && !isempty(e.s[end])
        update_value!(l, e.w[end], e.r[end])
    end
    for i in length(e.t) : -1 : 2
        # The 1st tick of a new episode
        if ((e.t[i-1] < T(1)) || isempty(e.w[i-1]))
            continue
        end
        # The last tick of an episode
        if (e.t[i] < T(1)) || isempty(e.w[i])
            update_value!(l, e.w[i-1], e.r[i-1], M[])
            continue
        end
        # Within episode
        update_value!(l, e.w[i-1], e.r[i-1], e.w[i])
    end
    return nothing
end

function init_value_update_counts!(l::Layer, init_count)
    for h in l.semantic
        init::R = R(init_count)
        # Initialise morpheme value update counts
        for v in h.v
            if (v[5] > init)
                v[5] = init
            end
        end
        # Initialise mean morpheme reward update counts
        for n in h.n
            if (n > init)
                n = init
            end
        end
    end
    return nothing
end

""" Memorise layer morpheme correlations in initiation (environment exploration) data """
function init_layer_memory!(l::Layer)::Nothing
    for i ∈ 1 : l.episodic.length-1
        m, m′ = l.episodic.m[i], l.episodic.m[i+1]
        if (m > I(0)) & (m′ > I(0))
            push!(l.memory[m], m′)
        end
    end
    return nothing
end

""" Parse layer episodic memory to get data for the next layer creation """
function reparse!(l::Layer)
    wv::Vector{Morpheme} = Morpheme[]
    rv::Vector{R} = R[]
    tv::Vector{T} = T[]
    w_::Morpheme, r_::R, t_::T = M[], R(0.), T(0)
    e = l.episodic

    for t ∈ 1 : e.length
        if (e.t[t] > 0)
            update_value!(l, e.w[t], e.r[t], M[])
            (w_, r_, t_) = parse_input!(l, e.i[t], e.w[t], e.r[t], e.t[t])
            if !isempty(w_)
                push!(wv, w_)
                push!(rv, r_)
                push!(tv, e.t[t])
            end
        elseif (t > 1)
            (W_, R_, T_) = clear_parser!(l)
            append!(wv, W_)
            append!(rv, R_)
            append!(tv, T_)
        end
    end

    (W_, R_, T_) = clear_parser!(l)
    append!(wv, W_)
    append!(rv, R_)
    append!(tv, T_)

    return (wv , rv, tv)
end
