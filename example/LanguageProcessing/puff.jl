

function update_parameters!(a::Adam, ap::Dict)::Nothing

    a.H = get(ap, "layers", a.H)            # max number of layers
    a.M = get(ap, "morphemes_in_heads", a.M)# max number of morphemes in layer heads
    a.N = get(ap, "heads", a.N)             # number of heads in layers
    a.K = get(ap, "clusters", a.K)          # number of clusters in layer modules
    a.O = get(ap, "parser_order", a.O)      # parsing stack length (number of morpheme pairs)
    a.W = get(ap, "morphemes_in_layers", a.W) # max number of layer morphemes
    a.d = get(ap, "clustering_space", a.d)
    a.decode_fast        = get(ap, "decode_fast", a.decode_fast)
    a.decode_num         = get(ap, "decode_num" , a.decode_num)
    a.decode_slow        = get(ap, "decode_slow", a.decode_slow)
    a.discount           = get(ap, "discount", a.discount)
    a.episodic_threshold = get(ap, "episodic_threshold", a.episodic_threshold)
    a.fitness            = get(ap, "fitness", a.fitness)
    a.global_value_update = get(ap, "global_value_update", a.global_value_update)
    a.initial_value      = get(ap, "initial_value", a.initial_value)
    a.internal_reward    = get(ap, "internal_rewards", a.internal_reward)
    a.learning_rate      = get(ap, "learning_rate", a.learning_rate)
    a.l                  = get(ap, "morpheme_max_lengths", a.l)
    a.planner_order      = get(ap, "planner_order", a.planner_order)
    a.planning_fast      = get(ap, "planning_fast", a.planning_fast)
    a.planning_slow      = get(ap, "planning_slow", a.planning_slow)
    a.plans_max          = get(ap, "plans_max", a.plans_max)
    a.plans_min          = get(ap, "plans_min", a.plans_min)
    a.ppmi_threshold     = get(ap, "ppmi_threshold", a.ppmi_threshold)
    a.quorum             = get(ap, "quorum", a.quorum)
    a.sampling_dispersion = get(ap, "sampling_dispersion", a.sampling_dispersion)
    a.semantic_threshold = get(ap, "semantic_threshold", a.semantic_threshold)
    a.step_reward        = get(ap, "step_reward", a.step_reward)
    a.sync_output        = get(ap, "sync_output", a.sync_output)
    a.thompson_sampling  = get(ap, "thompson_sampling", a.thompson_sampling)
    a.threading          = get(ap, "threading",      a.threading)
    a.update_rewards     = get(ap, "update_rewards", a.update_rewards)
    a.update_values      = get(ap, "update_values",  a.update_values)
    a.validate_plans     = get(ap, "validate_plans", a.validate_plans)
    a.value_threshold    = get(ap, "value_threshold", a.value_threshold)
    a.values             = get(ap, "values", a.values)
    a.winner_pair_count  = get(ap, "winner_pair_count", a.winner_pair_count)
    a.working_memory     = get(ap, "working_memory", a.working_memory)

    for l in a.layers
        l.ppmi_threshold = inherit(a.ppmi_threshold, l.L)
        for h in l.semantic
            h.ppmi_threshold = inherit(l.ppmi_threshold, h.N)
        end

        l.fitness = inherit(a.fitness, l.L)
        l.planner.decode_num = inherit(a.decode_num, l.L)
    end

    if (haskey(ap, "options"))
        a.opt = deepcopy(ap["options"])
    end
    return nothing
end

function save_empty_adam!(a::Adam, save_dir::String)::Nothing

    layers = a.layers

    a.layers = Layer[]

    save_object(joinpath(save_dir, "a$(a.L)_empty.jld"), a)

    a.layers = layers

    return nothing
end

function load_adam(load_dir::String, L::Int, trained::Bool)::Adam

    a = load_object(joinpath(load_dir, "a$(L)_empty.jld"))
    for ln in 1:(L-1)
        l = load_object(joinpath(load_dir, "L$(ln)_trained.jld"))
        push!(a.layers, l)
    end
    if trained
        l = load_object(joinpath(load_dir, "L$(L)_trained.jld"))
        push!(a.layers, l)
    else
        l = load_object(joinpath(load_dir, "L$(L)_empty.jld"))
        push!(a.layers, l)
    end
    return a
end

function expand_options!(a::Adam)::Nothing
    fields = ["min_frequency", "min_counts", "min_length", "max_dict"]
    for field in fields
        field_v = get(a.opt, field, [0])
        append!(field_v, map(x -> field_v[end], length(field_v):a.H - 1))
        a.opt[field] = field_v
    end
    return nothing
end

function cut_text!(text::String, a::Adam, volumes::Vector{Int} = Int[])::Vector{String}
    texts = String[]
    ending_punctuation = Set(a.app["ending punctuation"])
    for volume in volumes
        i = iterate(text, volume)[2]
        if i === nothing
            break
        end
        i = findnext(x -> x in ending_punctuation, text, i)
        if i === nothing
            break
        end
        i = findnext(x -> x == ' ', text, i)
        if i === nothing
            break
        end

        push!(texts, text[1:i])
        i = iterate(text, i)[2]
        text = text[i:end]
    end
    push!(texts, text)

    return texts
end

function cut_history(history::Vector{I}, parts::Vector{F} = F[])::Vector{Vector{I}}
    histories_v = Vector{I}[]
    len = length(history)
    last_i = 0

    for part in parts

        i = Int(round(part * len))
        i = findnext(x -> x <= I(0), history, i)
        if i === nothing
            break
        end

        push!(histories_v, history[(last_i + 1):i])
        last_i = i
    end
    push!(histories_v, history[(last_i+1):end])

    return histories_v
end

function stop_growth!(l::Layer)::Nothing

    l.W = l.n
    l.memory = l.memory[1:l.n]
    l.morphemes = l.morphemes[1:l.n]
    l.sequences = l.sequences[1:l.n]
    l.morpheme_symbols = l.morpheme_symbols[1:l.n]
    l.next_layer_symbol = l.next_layer_symbol[1:l.n]

    l.stop_growth = true

    return nothing
end






