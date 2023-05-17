# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.3                  # 2022-12-12
# Copyright (c) 2020-2023 Moscow Institute of Physics and Technology, Laboratory of Cognitive architectures

""" Forward pass through a layer #L of Adam structure """
function forward_pass!(a::Adam, s::Morpheme=M[], r::R=R(0.), t::T=T(0),
                       L::Int=1, x::Vector{Tuple{M, R}}=Tuple{M, R}[])
    l = a.layers[L]
    m::I = I(0)                 # layer morpheme index
    m̄::Vector{Plan}  = Plan[]   # m\bar[TAB]    ⇐ {m'}
    m″::Vector{Plan} = Plan[]   # m\pprime[TAB] ⇐ {m"}
    s̄::Vector{Plan}  = Plan[]   # plan in terms of layer symbols
    w::Vector{M} = M[]
    w′::Vector{M}, r′::R, t′::T = M[], R(0.0), T(0)

    # Encode symbol & vector → symbol index, morpheme index, morpheme
    (i, m, w) = encode!(l, s, x)
    #update_episodic!(l, i, m, r, s, t, w)

    if ((L > 1) && (i != I(0)))
        lp = a.layers[L-1]
        lp.next_layer_symbol[lp.index[s]] = true
    end

    (w′, r′, t′) = parse_input!(l, i, w, r, t)
    l.planner.last_morpheme = w′

    l.opt["stack"] = deepcopy(l.parser.stack)

    # Validate plan stashes 
    l.planner.instant_plans = validate(l.planner.instant_plans, l, s, t)
    executed_plans = validate(l.planner.plans, s, t)
    l.planner.plans = filter(p -> length(p[1]) >= l.min_plan_length, executed_plans)
    
    # Request plans from the upper layer(s)
    if (L < a.L) && (!isempty(w′) || isempty(l.planner.plans))
        wrtv = Tuple{Vector{M}, R, T}[]
        # Natural morpheme (parser.out)
        if !isempty(w′)
            push!(wrtv, (w′, r′, t′))
        end
        if (isempty(l.planner.plans) && (!isempty(executed_plans) && l.min_plan_length == 1))
            # Full synchronization
            wv, rv, tv = clear_parser!(l)
            append!(wrtv, map(i -> (wv[i], rv[i], tv[i]), eachindex(wv)))
        else
            # Synchronization
            append!(wrtv, actual_morphemes!(a, L))
        end
        if !isempty(wrtv)
            for (w_, r_, t_) ∈ wrtv
                m″ = forward_pass!(a, w_, r_, t_, L+1, sparse_vector(w_, l))
            end
            s̄ = validate_sync_decode(m″, l)
            if (!isempty(s̄))
                l.planner.plans = s̄
                if (length(l.planner.plans) > l.planner.decode_num)
                    return l.planner.plans[1:l.planner.decode_num]
                else
                    return l.planner.plans
                end
            end
        end
    end

    # Try to use the upper layer plans stash
    if (length(l.planner.plans) >= l.planner.min)
        if (length(l.planner.plans) > l.planner.decode_num)
            return l.planner.plans[1:l.planner.decode_num]
        end
        return l.planner.plans
    end

    # Suggest plans based on the last merged morpheme
    if (l.parser.last_merge > 0) && (l.planner.order > 0)
        s̄ = get_stack_plans(l)
    end
    if (!isempty(l.thompson_sampling))
        s̄ = sample_values(s̄, l)
    end

    # Suggest instant (layer-level) plans
    append!(s̄, l.planner.instant_plans)
    if (!isempty(w))
        m̄ = guess_plans(l)
        if (!isempty(m̄))
            s̄′ = filter(x -> length(x[1]) >= l.min_plan_length, decode(m̄, l))
            if (!isempty(s̄′) && !isempty(l.thompson_sampling))
                s̄′ = sample_values(s̄′, l)
            end
            append!(s̄, s̄′)
        end
        # Suggest short symbol strings based plans
        if (length(s̄) < l.planner.min)
            m̄ = guess_plans(l, true)
            if (!isempty(m̄))
                s̄′ = filter(x -> length(x[1]) >= l.min_plan_length, decode(m̄, l))
                if (!isempty(s̄′) && !isempty(l.thompson_sampling))
                    s̄′ = sample_values(s̄′, l)
                end
                append!(s̄, s̄′)
            end
        end
    end

    if (isempty(s̄))
        return Plan[]
    end

    # Update instant (layer) plan stash
    fitness = fitness_function(l) 
    l.planner.instant_plans = sort(s̄, by = p -> fitness(p), rev=true)
    if (length(l.planner.instant_plans) > l.planner.decode_num)
        return l.planner.instant_plans[1:l.planner.decode_num]
    end

    return l.planner.instant_plans

end

""" Update episodic memory """
function update_episodic!(l::Layer, i::I, m::I, r::R, s::Morpheme, t::T, w::Morpheme)
    e = l.episodic
    push!(e.i, i)
    push!(e.m, m)
    push!(e.r, r)
    push!(e.s, s)
    push!(e.t, t)
    push!(e.w, w)
    e.length += 1
    return nothing
end

""" Validate, strip, decode, sync multi-morpheme plans, starting in the past """
function validate_sync_decode(plans::Vector{Plan}, l::Layer)::Vector{Plan}
    if isempty(plans)
        return Plan[]
    end
    # Filter by length
    stack_i_string = vcat(map(x -> x[4], l.parser.stack)...)
    start = findfirst(i -> i != I(0), stack_i_string)
    if isnothing(start)
        stack_i_string = I[]
    else
        stack_i_string = stack_i_string[start:end]
    end
    if I(0) ∈ stack_i_string
        return Plan[]
    end
    stack_length = length(stack_i_string)
    t::T = isempty(stack_i_string) ? plans[1][4] : l.parser.stack[end][3] + T(1)
    w″::Vector{Plan} = decompose(plans, l)
    min_plan_length = stack_length + l.min_plan_length
    valid = map(x -> length(x[1]) >= min_plan_length, w″)
    if (!any(valid))
        return Plan[]
    end
    plans = plans[valid]
    w″ = w″[valid]

    # Choose plans relevant to the past
    stack_w_string = map(i -> l.morphemes[l.symbol_morphemes[i]], stack_i_string)
    valid = map(x -> x[1][1:stack_length] == stack_w_string, w″)
    if (!any(valid))
        return Plan[]
    end
    plans = plans[valid]

    # Process plans: strip morphemes, decode, strip past symbols
    ī′::Vector{Vector{I}} = Vector{I}[]
    ī″::Vector{Vector{I}} = Vector{I}[]
    s″::Vector{Vector{Morpheme}} = Vector{Morpheme}[]
    s_plans::Vector{Plan} = Plan[]  # symbol-based plans
    for plan in plans
        (w̄, v, p, tp) = plan
        ends = cumsum(map(w -> length(l.semantic[1].string[w[1]]), w̄))
        # Strip left (past) morphemes
        first = findfirst(x -> x > stack_length, ends)
        w̄ = w̄[first:end]
        w = w̄[1]
        ī′ = collect(l.sequences[l.index[w]])
        filter!(x -> !isempty(x), ī′)
        if (isempty(ī′))
            continue
        end
        # Validate past part of ī′, strip past symbols
        past = (first == 1 ? stack_length : stack_length - ends[first-1])
        if (past > 0)
            filter!(ī -> ī[1:past] == stack_i_string[end-past+1 : end], ī′)
            if (isempty(ī′))
                continue
            end
            ī′ = map(ī -> ī[past+1 : end], ī′)
        end
        ī″ = ī′
        if (length(w̄) > 1)
            futures = map(w -> collect(l.sequences[l.index[w]]), w̄[2:end])
            ī″ = mix_i_strings([[ī′]; futures])
        end
        if (isempty(ī″))
            continue
        end
        s″ = map(ī -> map(i -> l.symbols[i], ī), ī″)
        filter!(x -> !isempty(x), s″)
        if (!isempty(s″))
            append!(s_plans, map(s̄ -> (s̄, v, p, t), s″))
        end
    end
    return s_plans
end

function mix_i_strings(iii::Vector{Vector{Vector{I}}})::Vector{Vector{I}}
    (isempty(iii) || any(x -> isempty(x), iii)) ? Vector{I}[] :
    length(iii) == 1 ? iii[1] :
    [[x; y] for x ∈ iii[1] for y ∈ mix_i_strings(iii[2:end])]
end

function validate(plans::Vector{Plan}, s::Morpheme, t::T)::Vector{Plan}
    filter!(p -> p[1][1] == s, plans)
    plans = map(p -> (p[1][2:end], p[2], p[3], t + T(1)), plans)
    return plans
end

function validate(plans::Vector{Plan}, l::Layer, s::Morpheme, t::T)::Vector{Plan}
    filter!(p -> ((length(p[1]) > l.min_plan_length) && (p[1][1] == s)), plans)
    plans = map(p -> (p[1][2:end], p[2], p[3], t + T(1)), plans)
    return plans
end

function get_stack_plans(l::Layer)::Vector{Plan}
    (w′, r′, t′) = l.parser.stack[l.parser.last_merge]
    # Check past debth
    i_past = vcat(map(x -> x[4], l.parser.stack[l.parser.last_merge+1 : end])...)
    if ((length(i_past) > (l.l - l.min_plan_length)) || (I(0) ∈ i_past))
        return Plan[]
    end
    s_past = map(i -> l.symbols[i], i_past)
    w_plans = predict(w′, l, t′)  # List of w-based plans
    if (isempty(w_plans))
        return Plan[]
    end
    s_plans = decode(w_plans, l)  # List of s-based plans
    if (isempty(s_plans))
        return Plan[]
    end
    t_past::Int = length(i_past)

    # Choose long enough plans
    min_plan_length::Int = t_past + l.min_plan_length
    filter!(x -> length(x[1]) >= min_plan_length, s_plans)
    if (isempty(s_plans))
        return Plan[]
    end

    # Choose plans, consistent with current parser stack
    filter!(x -> x[1][1:t_past] == s_past, s_plans)
    if isempty(s_plans)
        return Plan[]
    end
    t::T = l.parser.stack[end][3] + T(1)
    return map(p -> (p[1][t_past+1:end], p[2], p[3], t), s_plans)
end

""" Suggest "instant" plans based on (hypothetically possible) morphemes created by merging last symbols """
function guess_plans(l::Layer, shorter_morphemes::Bool=false)::Vector{Plan}
    if isempty(l.parser.stack[end][1])
        return Plan[]
    end
    m̄::Vector{Plan} = Plan[]
    from::Int = 0
    to::Int = 0
    s̄ = l.parser.stack[end][4]
    i = l.parser.O
    while ((length(s̄) < l.l) && (i > 0) && (I(0) ∉ l.parser.stack[i][4]))
        s̄ = [l.parser.stack[i][4]; s̄]
        i -= 1
    end
    w̄ = map(x -> l.morphemes[l.symbol_morphemes[x]], s̄)
    w̄_length = length(w̄)

    if (!shorter_morphemes)  # plans based on morphemes not shorter than planner.order
        if ((l.planner.order < 2) || (w̄_length < 2))
            return predict(w̄[end], l, l.parser.stack[end][3])
        end
        if ((l.planner.order > l.l) || (l.planner.order > w̄_length))
            return Plan[]
        end
        from = max(1, w̄_length - l.l + 1)
        to   = max(from, w̄_length - l.planner.order + 1)
    else  # plans based on morphemes shorter than planner.order
        if ((l.planner.order < 2) || (w̄_length < 2))
            return Plan[]
        end
        from = max(1, w̄_length - min(l.planner.order, l.l) + 2)
        to   = w̄_length
    end
    for i ∈ from : to
        w = guess_morpheme(w̄[i:end], l)
        if (invalid(w))
            continue
        end
        append!(m̄, predict(w, l, l.parser.stack[end][3]))
        if (length(m̄) >= l.planner.min)
            return m̄
        end
    end
    return m̄
end

""" Find a layer morpheme formed by a string of layer symbols """
function guess_morpheme(w̄::Vector{Vector{M}}, l::Layer)::Vector{M}
    if (length(w̄) == 1)
        return w̄[1]

    elseif (!isempty(w̄) && all(w -> (!isempty(w) && (M(0) ∉ w)), w̄))
        w::Vector{M} = check(map(n -> get(l.semantic[n].string_idx,
                             map(w -> w[n], w̄), M(0)), 1:l.N))
        if (haskey(l.index, w))
            return w
        end
        return M[]
    end
end

""" Suggest layer plans following the current layer symbol w::Vector{M} """
function predict(w::Vector{M}, l::Layer, t::T)::Vector{Plan}
    m̄::Vector{Plan} = Plan[]
    if (invalid(w))
        return m̄
    end

    # 1 head layer
    if (l.N == 1)
        return singlehead(w, l, t, 1)
    end

    # Fast planning
    if (l.planner.planning_fast)
        if (haskey(l.index, w))
            m̄′::Vector{I} = collect(l.memory[l.index[w]])
            if !isempty(m̄′)
                w″::Vector{Vector{M}} = map(m -> l.morphemes[m], m̄′)
                m̄ = map(w′ -> ([w′], get_value(w′, l),
                               get_probability(w, w′, l), t+T(1)), w″)
            end
        end
        if (length(m̄) >= l.planner.min)
            return m̄
        end
    end
    if (!l.planner.planning_slow)
        return m̄
    end

    # Slow planning
    paths::Vector{Tuple{Vector{Morpheme},R,R}} = Tuple{Vector{Morpheme},R,R}[]
    quorum::Int = max(min(Int(round(l.planner.quorum * l.N)), l.N), 2)
    quo::Int = max(l.N-1, quorum)
    while (length(paths) < l.planner.min) && (quo >= quorum)
        append!(paths, vote(suggest(w, l), quo, l))
        quo -= 1
    end
    if (isempty(paths))
        return m̄
    end
    return [m̄ ; map(x -> (x[1], x[3], x[2], t+T(1)), paths)]
end

""" Suggest 1-head plans following the current layer symbol w::Vector{M} """
function singlehead(w::Vector{M}, l::Layer, t::T, head::Int=1)::Vector{Plan}
    paths::Vector{Tuple{Vector{Morpheme},R,R}} =
        valuate(suggest(w[head], l.semantic[head]), l)
    if (isempty(paths))
        return Plan[]
    end
    return map(x -> (x[1], x[3], x[2], t+T(1)), paths)
end

""" Decompose and valuate 1-head layer plan suggestion """
function valuate(suggestions::Vector{Pair{I,R}}, l::Layer
                 )::Vector{Tuple{Vector{Morpheme},R,R}}
    isempty(suggestions) ? Tuple{Morpheme,R,R}[] :
    map(x -> ([l.morphemes[x[1]]], x[2], get_value(l.morphemes[x[1]], l)), suggestions)
end

function suggest(w::M, h::Head)::Vector{Pair{I,R}}
    vcat(map(x -> map(y -> y => R(x[2]/h.R_i[w]), collect(h.m̄[x[1]])),
             collect(h.R[w]))...)
end

""" Find head consensus on a number of head suggestions """
function vote(suggestions::Vector{Dict{I,R}}, quorum::Int, l::Layer
              )::Vector{Tuple{Vector{Morpheme},R,R}}  # (Layer morpheme, probability, value)
    counter::Dict{I,Int} = Dict()
    for key in vcat(map(x -> collect(keys(x)), suggestions)...)
        counter[key] = haskey(counter, key) ? counter[key] + 1 : 1
    end
    m̄::Vector{I} = collect(keys(filter(x -> x[2] >= quorum, counter)))
    isempty(m̄) && return Tuple{Morpheme,R,R}[]
    probability(m) = mean(filter(x -> x > R(0.), map(x -> get(x, m, R(0.)), suggestions)))
    return map(m -> ([l.morphemes[m]], probability(m), get_value(l.morphemes[m], l)), m̄)
end

""" Find indices of layer morphemes following the given layer morpheme m and their probabilities """
function suggest(w::Morpheme, l::Layer)::Vector{Dict{I,R}}
    map(x -> Dict(x), map(h -> suggest(w[h.N], h), l.semantic))
end

""" Thompson sampling """
function sample_values(plans::Vector{Plan}, l::Layer)::Vector{Plan}
    map(plan -> sample_value(plan, l), plans)
end

function sample_value(p::Plan, l::Layer)::Plan
    sv = l.sampled_values
    haskey(sv, p[1]) && return ((p[1]), sv[p[1]], p[3], p[4])
    v = thompson_sampling(p[2], l.thompson_sampling, l.sampling_dispersion)
    sv[p[1]] = v
    return ((p[1]), v, p[3], p[4])
end

function thompson_sampling(x::R, algorithm::String, dispersion::R = R(0.33))::R
    isempty(algorithm) ? x :
    lowercase(algorithm[1:4]) == "pois" ?  # Poisson
        R(sign(x) * (rand(Poisson(abs(x))) + rand() - 0.5)) :
    lowercase(algorithm[1:4]) == "gaus" ?  # Gauss
        R(x * (1. + randn() * dispersion)) : x
end

""" Plan fitness function: plan[2] - value, plan[3] ~ probability (kind of) """
function fitness_function(l::Layer)
    if l.fitness ∈ [2, 3]
        return plan -> plan[l.fitness]
    elseif l.fitness == 1
        return plan -> plan[2] * plan[3]
    elseif l.fitness == 4
        return plan -> length(plan[1])
    else
        return plan -> plan[2]
    end
end

