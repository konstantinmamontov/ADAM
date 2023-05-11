# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.3                  # 2022-12-12
# Copyright (c) 2020-2022 Oleg Baskov https://github.com/OlegBaskov


""" Add a new upper layer to ADAM structure, initiate """
function add_layer!(a::Adam)
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
    l.planner.decode_num = a.decode_num[a.L]
    e = l.episodic
    l.Rnd = bitrand(l.d, 2 * lp.M * lp.N)

    (es, er, et) = reparse!(lp)
    unique_symbols::Vector{Vector{M}} = unique(es)
    unique_head_morphemes::Vector{Vector{M}} = map(
        h -> unique(map(s -> s[h.N], unique_symbols)), lp.semantic)

    print("head_vectors: ")
    @time head_vectors = map(h -> map(
        wh -> head_vector(M(wh), h, M(2*l.M*(h.N-1))), 1:h.D), lp.semantic)    
    flush(Base.stdout)

    print("subtotals: ")
    subtotals = map(h -> map(wh -> zeros(F, l.d), 1:h.D), lp.semantic)
    @time for h ∈ lp.semantic
        @threads for wh ∈ unique_head_morphemes[h.N]
        #for wh ∈ unique_head_morphemes[h.N]
            #-subtotals[h.N][wh] = sparse2dense(head_vectors[h.N][wh], l)
            sparse2dense!(subtotals[h.N][wh], head_vectors[h.N][wh], l)
        end
    end
    flush(Base.stdout)

    #subtotals = map(h -> map(
    #    wh -> sparse2dense(head_vectors[h.N][wh], l), 1:h.D), lp.semantic)

    dv = Dict(map(s -> s => normalize(
        mapreduce(h -> subtotals[h.N][s[h.N]], +, lp.semantic)), unique_symbols))

    s_c = Dict{Morpheme, Int}()

    if a.L == 2
        a.app["sv_length"] = Dict{M, Int}()
    end
    sparse_vector_min_length = get(a.opt, "sparse_vector_min_length", 7)
    allocate_episodic!(e, length(es))
    for (i,s) ∈ enumerate(es)
        sv_length = mapreduce(
            h -> length(head_vectors[h.N][s[h.N]]), +, lp.semantic)
        if a.L == 2 && !haskey(a.app["sv_length"], s[1])
            a.app["sv_length"][s[1]] = sv_length
        end
        if (sv_length >= sparse_vector_min_length)
            push!(e.s, s)
            push!(e.r, er[i])
            push!(e.t, et[i])
            (et[i] > 0) && (s_c[s] = get(s_c, s, 0) + 1)
        end
    end

    s_counts = collect(s_c)
    clustering_vectors = map(sc -> dv[sc[1]], s_counts)
    weights = map(sc -> F(sc[2]), s_counts)
    #-weights /= sum(weights)

    print("clustering: ")
    @time coder = new_coder(l, clustering_vectors, weights)
    flush(Base.stdout)

    symbols = map(sc -> sc[1], s_counts)
    morphemes = map(i -> map(c -> M(c[2][i]), coder), eachindex(symbols))

    implant_coder!(l, map(c -> c[1], coder))

    new_layer_morphemes!(l, map(
        i -> symbols[i] => morphemes[i], eachindex(symbols)))

    encoded_symbols = Set(symbols)

    imw = map(s -> s ∈ encoded_symbols ? encode!(l, s) : encode!(
        l, s, vcat(map(h -> head_vectors[h.N][s[h.N]], lp.semantic)...)), e.s)
    e.i = map(x -> x[1], imw)
    e.m = map(x -> x[2], imw)
    e.w = map(x -> x[3], imw)
    e.length = length(e.t)

    for s ∈ l.symbols
        lp.next_layer_symbol[lp.index[s]] = true
    end

    @assert e.length==length(e.i)==length(e.r)==length(e.s)==length(e.w)

    new_semantic!(l)
    reparse!(l)

    t1 = Dates.now()
    @info "New layer #"* string(a.L) *" created in "* time_interval(t0, t1) => Dates.format(t1, "yyyy-mm-dd HH:MM:SS")

    return nothing
end

""" Create semantic memory in a layer with episodic memory """
function new_semantic!(l::Layer)
    e::Episodic = l.episodic
    if isempty(e.w)
        return nothing
    end

    if ((e.t[end] > 0) && !isempty(e.w[end]))
        update_value!(l, e.w[end], e.r[end], M[])
    end

    # Update counts and values for symbols in semantic memory
    for i ∈ length(e.t) : -1 : 2
        # 1st tick of a new episode
        if ((e.t[i-1] < T(1)) || isempty(e.w[i-1]))
            continue
        end
        # Last tick of an episode
        if ((e.t[i] < T(1)) || isempty(e.w[i]))
            update_value!(l, e.w[i-1], e.r[i-1], M[])
            continue
        end
        update_count!(l, e.w[i-1], e.w[i])
        update_value!(l, e.w[i-1], e.r[i-1], e.w[i])
    end
    return nothing
end

function new_morphemes!(l::Layer)
    map(h -> new_morphemes!(h), l.semantic)
end

function new_morphemes!(h::Head)
    [   new_morpheme!(h, i, j, h.R[i][j])
        for i::M ∈ M(1) : M(length(h.R))
        for j ∈ sort(collect(keys(h.R[i])))
        if  h.R[i][j] > 0   ]
end

""" Create layer morphemes from encoded symbols """
function new_layer_morphemes!(l::Layer,
    codes::Vector{Pair{Morpheme, Morpheme}})::Nothing

    for (s, w) ∈ codes

        if (isempty(s) || invalid(w))
            continue
        end

        i = I(length(l.symbols) + 1)

        m::I = add_layer_morpheme!(l, w, [i])

        if m > I(0)
            push!(l.symbols, s)
            l.symbol_index[s] = i
            push!(l.symbol_morphemes, m)
        end
    end

    return nothing
end

function allocate_episodic!(e::Episodic, len::Int)::Nothing
    
    sizehint!(e.s, len)
    sizehint!(e.r, len)
    sizehint!(e.t, len)

    return nothing
end