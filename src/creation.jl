# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.3                  # 2022-12-12
# Copyright (c) 2020-2022 Oleg Baskov https://github.com/OlegBaskov


""" Create a new (empty) ADAM structure with a single empty 1st layer """
function new_adam(ap::Dict = Dict())
    a = Adam()
    # Default parameters:
    a.H = get(ap, "layers", 1)           # max number of layers
    a.L = 1                              # number of created layers
    a.M = get(ap, "morphemes_in_heads", [1000])  # max number of morphemes in layer heads
    a.N = get(ap, "heads", [1])          # number of heads in layers
    a.K = get(ap, "clusters", [[100]])   # number of clusters in layer modules
    a.O = get(ap, "parser_order", [2])   # parsing stack length (number of morpheme pairs)
    a.W = get(ap, "morphemes_in_layers", [100000]) # max number of layer morphemes
    a.application        = get(ap, "application", "")
    a.adaptive_clusters  = get(ap, "adaptive_clusters", [false])
    a.cluster_mass       = get(ap, "cluster_mass", [[R(0)]])
    a.clustering         = get(ap, "clustering", ["kmeans"])
    a.coder              = get(ap, "coder", [1])
    a.d = get(ap, "clustering_space", [4, 100])
    a.decode_fast        = get(ap, "decode_fast", [false])
    a.decode_num         = get(ap, "decode_num" , [100])
    a.decode_slow        = get(ap, "decode_slow", [true])
    a.discount           = get(ap, "discount", [[0.9]])
    a.episodic_threshold = get(ap, "episodic_threshold", [10000])
    a.fitness            = get(ap, "fitness", [2])
    a.global_value_update = get(ap, "global_value_update", [false])
    a.initial_value      = get(ap, "initial_value", R(0.0))
    a.input_weights      = get(ap, "input_weights", [])
    a.internal_reward    = get(ap, "internal_rewards", [R(1.0)])
    a.layers = Layer[]
    a.learning_rate      = get(ap, "learning_rate", [[0.5]])
    a.l                  = get(ap, "morpheme_max_lengths", [3])  # ex.m
    a.planner_order      = get(ap, "planner_order", [2])
    a.planning_fast      = get(ap, "planning_fast", [false])
    a.planning_slow      = get(ap, "planning_slow", [true])
    a.plans_max          = get(ap, "plans_max", [[100]])
    a.plans_min          = get(ap, "plans_min", [1])
    a.ppmi_threshold     = get(ap, "ppmi_threshold", [[R(0.0)]])
    a.quorum             = get(ap, "quorum", [0.5])
    a.sampling_dispersion = get(ap, "sampling_dispersion", [R(0.33)])
    a.scaler = F[]
    a.semantic_threshold = get(ap, "semantic_threshold", [[R(3)]])
    a.step_reward        = get(ap, "step_reward", [R(0.0)])
    a.sync_output        = get(ap, "sync_output", [true, false])
    a.thompson_sampling  = get(ap, "thompson_sampling", [""])
    a.threading          = get(ap, "threading",      [true])
    a.update_rewards     = get(ap, "update_rewards", [true])
    a.update_values      = get(ap, "update_values",  [true])
    a.validate_plans     = get(ap, "validate_plans", [true, false])
    a.value_threshold    = get(ap, "value_threshold", [R(0.)])
    a.values             = get(ap, "values", [[2]])
    a.winner_pair_count  = get(ap, "winner_pair_count", [R(1)])
    a.working_memory     = get(ap, "working_memory", [7])

    # Development, exploratory, temporary options
    if (haskey(ap, "options"))
        a.opt = deepcopy(ap["options"])
    end

    # Add the 1st layer
    push!(a.layers, new_layer(1, a))

    return a
end

""" Create a new (empty) layer structure, L - number of the layer, 1-based """
function new_layer(L::Int, a::Adam)
    l = Layer()
    for field in intersect(fieldnames(Layer), fieldnames(Adam))
        inherit!(l, field, a, L)
    end
    @assert l.L == L

    # Update Parser
    l.parser = new_parser(inherit(a.O, L), inherit(a.working_memory, L), l.N)

    # Update Planner
    for field in intersect(fieldnames(Planner), fieldnames(Adam))
        inherit!(l.planner, field, a, L)
    end
    l.planner.order = inherit(a.planner_order, L)
    l.planner.max   = inherit(a.plans_max, L, l.N)
    l.planner.min   = inherit(a.plans_min, L)

    # Update semantic memory (created empty)
    l.semantic = [new_head(l, n) for n ∈ 1 : l.N]

    # Update layer memory (created empty)
    l.memory =[Set(I[]) for _ ∈ 1 : l.W]

    # Layer morpheme vectors
    l.morphemes  = [M[] for _ ∈ 1 : l.W]
    l.morpheme_symbols = [I[] for _ ∈ 1 : l.W]
    l.sequences = [Set(Vector{I}[]) for _ ∈ 1 : l.W]
    l.next_layer_symbol = [false for _ ∈ 1 : l.W]

    # Development, exploratory, temporary options
    l.opt = deepcopy(a.opt)

    return l
end

""" Create a new parser::Parser (pure, no changes anywhere) """
function new_parser(order::Int=2, memory_length::Int=7, morpheme_length::Int=1)::Parser
    p = Parser()
    p.O = order
    p.memory_length = memory_length
    p.stack = fill((M[], R(0.), T(0), I[]), p.O+1)
    p.r = fill(typemin(R), p.O+1)
    if (p.memory_length > 0)
        p.memory = zeros(M, memory_length, morpheme_length)
        p.reward = fill(typemin(R), memory_length)
    else
        p.memorise = false
    end
    return p
end

""" Inherit parent structure field values """
function inherit!(sibling, s::Symbol, parent, n::Int)
    if (!hasfield(typeof(sibling), s) || !hasfield(typeof(parent), s))
        return s => nothing
    end
    field = getfield(parent, s)
    if (typeof(field) <: Dict)
        return s => nothing
    end
    if (typeof(field) <: Vector)
        if (isempty(field))
            return s => getfield(sibling, s)
        elseif (typeof(field[1]) <: Vector)
            setfield!(sibling, s, inherit(field, n, sibling.N))
        else
            setfield!(sibling, s, inherit(field, n))
        end
    else
        setfield!(sibling, s, getfield(parent, s))
    end
    return s => getfield(sibling, s)
end

function inherit(vv::Vector, n::Int, N::Int)  # vv::Vector{Vector}
    isempty(vv) && return nothing
    !(typeof(vv) <: Vector) && return nothing
    v = length(vv) > n ? vv[n] : vv[end]
    (length(v) > N) && return v[1:N]
    values = [v[end] for _∈1:N]
    values[1:length(v)] = v[:]
    return values
end

inherit(v::Vector, n::Int) = length(v) < n ? v[end] : v[n]

""" Create a new (empty) head (semantic memory module) """
function new_head(l::Layer, N::Int)
    h = Head()
    h.N = N  # Head number
    for field in filter(x -> x ∉ [:m̄, :N, :n, :K, :k, :D, :coder, :encoder],
                        intersect(fieldnames(Head), fieldnames(Layer)))
        inherit!(h, field, l, 1)
    end
    h.k = l.K[N]  # Number of clusters
    h.D = l.K[N]  # Number of morphemes
    h.max_plans = l.planner.max[N]
    h.n   = fill(0, h.M)  # Number of morpheme mean reward updates
    h.R   = [Dict{M,R}() for _ ∈ 1 : h.M]
    h.Rt  = [Dict{M,R}() for _ ∈ 1 : h.M]
    h.R_i = zeros(h.M)
    h.R_f = zeros(h.M)
    h.r   = fill(l.step_reward, h.M)  # Morpheme mean (collected) reward
    h.string = [M[] for _ ∈ 1 : h.M]
    h.string[1:h.k] = [[x] for x ∈ 1 : h.k]
    h.string_idx = Dict([x] => x for x ∈ 1 : h.k)
    v::R = l.initial_value
    h.v = [[v, v, R(0.), R(0.), R(0.)] for _ ∈ 1 : h.M]
    h.m̄ = [Set() for _ ∈ 1 : h.M]  # head morphemes ⇒ layer morpheme indices
    return h
end

