# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.3                  # 2022-12-12
# Copyright (c) 2020-2022 Oleg Baskov https://github.com/OlegBaskov


""" Add a new upper layer to ADAM structure, initiate """
function init_layer!(a::Adam, mv::Vector{I})
    t0 = Dates.now()
    @assert length(a.layers) > 0
    @assert a.L < a.H
    @assert a.L == length(a.layers)
    lp = a.layers[end]
    a.L += 1
    push!(a.layers, new_layer(a.L, a))
    @assert a.L == length(a.layers)
    l = a.layers[end]
    if (length(a.decode_num) < a.L)
        push!(a.decode_num, a.decode_num[end])
    end
    #-l.planner.decode_num = a.decode_num[a.L]
    l.Rnd = bitrand(l.d, 2 * lp.M * lp.N)

    m_c::Dict{I, Int} = countmap(mv)
    filter!(x -> x[1] > I(0), m_c)
    unique_m = collect(keys(m_c))
    #unique_m = unique(mv)
    
    unique_symbols::Vector{Vector{M}} = map(m -> lp.morphemes[m], unique_m)
    unique_head_morphemes::Vector{Vector{M}} = map(
        h -> unique(map(s -> s[h.N], unique_symbols)), lp.semantic)

    print("head_vectors: ")
    @time head_vectors = map(h -> map(
        wh -> head_vector(M(wh), h, M(2*lp.M*(h.N-1))), 1:h.D), lp.semantic)    
    flush(Base.stdout)

    print("subtotals: ")
    subtotals = map(h -> map(wh -> zeros(F, l.d), 1:h.D), lp.semantic)
    @time for h ∈ lp.semantic
        @threads for wh ∈ unique_head_morphemes[h.N]
            sparse2dense!(subtotals[h.N][wh], head_vectors[h.N][wh], l)
        end
    end
    flush(Base.stdout)

    min_counter = a.opt["min_frequency"][a.L-1]
    min_length = a.opt["min_length"][a.L-1]
    max_dict = a.opt["max_dict"][a.L-1]
    sv_min_length = get(a.opt, "sparse_vector_min_length", 7)
    
    
    len(s::Morpheme) = length(lp.semantic[1].string[s[1]])
    sv_length(s::Morpheme) = mapreduce(
        h -> length(head_vectors[h.N][s[h.N]]), +, lp.semantic)
    transition_condition(x::Pair{I, Int}) =  x[2] >= min_counter &&
        len(lp.morphemes[x[1]]) >= min_length && 
        sv_length(lp.morphemes[x[1]]) >= sv_min_length

    if max_dict > length(m_c)
        max_dict = length(m_c)
    end
    #m_counts = filter(x -> transition_condition(x), collect(m_c))
    m_counts = sort(collect(m_c), by = x -> x[2], rev = true)[1:max_dict]
    clustering_m = map(mc -> mc[1], m_counts)
    weights = map(mc -> F(mc[2]), m_counts)

    a.app["weights"] = Dict(map(mc -> lp.morphemes[mc[1]] => mc[2], m_counts))
    symbols = map(mc -> lp.morphemes[mc[1]], m_counts)
    if a.L == 2
        a.app["sv_length"] = Dict(map(s -> s[1] => sv_length(s), symbols))
    end

    empty!(head_vectors)

    print("making clustering vectors: ")
    clustering_vectors = zeros(F, l.d, length(clustering_m))
    @time for i in eachindex(clustering_m)
        dense_vector!(view(clustering_vectors, :, i), subtotals, lp.morphemes[clustering_m[i]])
    end
    flush(Base.stdout)

    empty!(subtotals)

    print("clustering: ")
    @time coder = new_coder(l, clustering_vectors, weights)
    flush(Base.stdout)

    morphemes = map(i -> map(c -> M(c[2][i]), coder), eachindex(symbols))

    implant_coder!(l, map(c -> c[1], coder))

    new_layer_morphemes!(l, map(
        i -> symbols[i] => morphemes[i], eachindex(symbols)))

    t1 = Dates.now()
    @info "New layer #"* string(a.L) *" created in "* time_interval(t0, t1) => Dates.format(t1, "yyyy-mm-dd HH:MM:SS")

    return nothing
end

function trim_R!(l::Layer, min_counts::Int)::Nothing
    for h in l.semantic
        trim_R!(h, min_counts)
    end
    return nothing
end

function trim_R!(h::Head, min_counts::Int)::Nothing
    for wh in 1:h.D
        filter!(wr -> wr[2] >= min_counts, h.R[wh])
        filter!(wr -> wr[2] >= min_counts, h.Rt[wh])
    end

    return nothing
end

function cut_context!(a::Adam)::Nothing
    ending_punctuation = a.app["ending punctuation"]
    l = a.layers[1]
    h = l.semantic[1]
    for char in ending_punctuation
        wh1 = encode(char, a)[1]
        wh2 = encode(' ', a)[1]
        wh = h.string_idx[[wh1, wh2]]
        empty!(h.R[wh])
        h.R_if -= h.R_i[wh]
        h.R_i[wh] = R(0.0)
        
        for m in 1:h.D
            if haskey(h.Rt[m], wh)
                r = pop!(h.Rt[m], wh)
                h.R_f[m] -= r
            end
        end
    end
    return nothing
end

    
