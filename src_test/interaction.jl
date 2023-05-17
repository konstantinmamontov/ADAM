# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.1                  # 2022-10-18
# Copyright (c) 2020-2023 Moscow Institute of Physics and Technology, Laboratory of Cognitive architectures


function query!(a::Adam, o::Tuple{Int, Vector{F}, R})
    decode_plans(get_plans!(a, o), a)
end

""" Decode the 1st layer output to the application, policies: (1) best value (2) optimism """
function decode_plans(plans::Vector{Plan}, a::Adam)
    isempty(plans) ? rand(a.app["actions"]) : decode_plan(plans[1], a)
end

function decode_plan(plan::Plan, a::Adam)
    actions::Vector{Int} = get(a.app, "actions", [0, 2])
    optimism::Bool = get(a.app, "optimism", false)
    t = a.layers[1].episodic.t[end]
    if (length(plan[1]) < t - plan[end] + 1 + a.layers[1].min_plan_length)
        @debug "short plan" => (plan, t)
        return rand(actions)
    end

    w = plan[end] > t ? plan[1][1] : plan[1][t - plan[end] + 2]

    factions = map(h -> h.coder[1, h.string[w[h.N]][1]], a.layers[1].semantic)
    faction  = mean(factions) * a.rescaler[1]
    action = fit_action(faction, actions)
    !optimism && return action

    h = a.layers[1].semantic[1]
    successors = collect(keys(h.R[w[1]]))
    coder_actions = map(x -> h.coder[:, h.string[x][1]][1], successors)
    scaled_actions = map(x -> x * a.rescaler[1], coder_actions)
    options = map(x -> fit_action(x, actions), unique(scaled_actions))
    (length(actions) > length(options)) && (
        action = rand([a for a ∈ actions if a ∉ options]))
    return action
end

""" Choose the best fit action to the mean head suggestion """
function fit_action(x::F, actions::Vector{Int})::Int
    actions[findmin(map(y -> abs(x-y), actions))[2]]
end

""" Preprocess input, forward_pass to ADAM planning process """
function get_plans!(a::Adam, o::Tuple{Int, Vector{F}, R})::Vector{Plan}
    (o[1] == typemin(Int)) && return forward_pass!(a)
    x::Vector{F} = [F(o[1]), o[2]..., o[3]] .* a.scaler
    w::Morpheme = encode!(a.layers[1], x)
    t = T(a.layers[1].episodic.length) + T(1)
    return forward_pass!(a, w, o[end], t)
end

""" Adaptive clusters encoding """
function encode!(l::Layer, x::Vector{F})::Vector{M}
    #[encode!(h, x) for h in l.semantic]
    map(h -> encode!(h, x), l.semantic)
end

function encode!(h::Head, x::Vector{F})::M
    c::M = encode(x, h)  # cluster_id
    if (!h.adaptive_clusters)
        return c
    end
    was = h.coder[:,c]
    mass::R = h.R_i[c] + h.R_f[c] + h.cluster_mass
    isnan(x[1]) && (x[1] = was[1,c])
    h.coder[:,c] = (mass * h.coder[:, c] + x) / (mass + R(1.))
    h.coder[1,c] = was[1]
    return c
end

""" Check plan validity for the current episodic memory state, trim older steps """
function sync(plans::Vector{Plan}, t::T)::Vector{Plan}
    filter(p -> p[end] > 0, map(p -> sync(p,t), plans))
end

function sync(p::Plan, t::T)::Plan                          #FIXME
    length(p[1]) < 2 ? (Morpheme[], R(0.), R(0), T(0)) :
    p[end] >= t ? p :
    (p[end] + length(p[1])) < (t + 2) ? (Morpheme[], R(0.), R(0), T(0)) :
    (p[1][(t - p[end] + 1):end], p[2], p[3], t)
end

""" Clear episodic memory of all the ADAM structure layers """
function clear_episodic!(a::Adam)::Nothing
    for l in a.layers
        clear_episodic!(l)
    end
end

function clear_episodic!(l::Layer)::Nothing
    l.episodic = Episodic()
    clear_planner!(l)
    return nothing
end

function clear_planner!(l::Layer)::Nothing
    l.planner.plans = Plan[]
    l.planner.instant_plans = Plan[]
    return nothing
end

""" End ADAM training epoch: clear layer parsers and planners, add new layer """
function end_epoch!(a::Adam, v::R=R(0.))
    wvt::Tuple{Vector{Morpheme}, Vector{R}, Vector{T}} = ([],[],[])
    for l in a.layers
        clear_planner!(l)
        wvt = clear_parser!(l)
        if (l.L < a.L) && !isempty(wvt) && !isempty(wvt[1])
            for (i,w) in enumerate(wvt[1])
                forward_pass!(a, w, wvt[2][i], wvt[3][i], l.L+1,
                              sparse_vector(w, l))
            end
        end
        empty!(l.sampled_values)
    end
    # Add a new layer
    if (a.L < a.H) && (
        a.layers[end].episodic.length > a.layers[end].episodic_threshold)
        add_layer!(a)
    end
    return nothing
end

