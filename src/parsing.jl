# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.3                  2022-12-12
# Copyright (c) 2020-2022 Oleg Baskov https://github.com/OlegBaskov

function parse_input!(l::Layer, i::I, w::Morpheme, r::R, t::T=T(0)
                      )::Tuple{Vector{M}, R, T}
    p = l.parser
    s = l.parser.stack
    update_reward!(l, w, r)
    update_values!(l, w)

    # Find the winner: morpheme pair with max probability
    winners = sortperm(p.r, rev = true)
    p.r[end] = get_count(s[end][1], w, l)
    i_max = winners[1]
    r_max = p.r[i_max]
    for k ∈ 2:p.O
        if mergeable(l, i_max) || r_max == R(0.)
            break
        end
        i_max = winners[k]
        r_max = p.r[i_max]
    end
    if (isempty(s[i_max+1][1]) || isempty(s[i_max][1]))
        update_memory!(l, w, r, t, [i])
        return p.out
    end

    # Try to merge the winner morpheme pair
    s̄::Vector{I} = [s[i_max][4]; s[i_max+1][4]]  # layer symbol index string
    (w_, r_, v_) = merge_morphemes!(l, s[i_max][1], s[i_max+1][1],
                                       s[i_max][2], s[i_max+1][2], s̄)
    if (isempty(w_))  # Merge failed
        p.r[i_max] = update_count!(l, s[i_max][1], s[i_max+1][1], l.winner_pair_count)
        update_memory!(l, w, r, t, [i])
        p.last_merge = 0
        p.last_merged_morpheme = (M[], R(0.), T(0))
        return p.out
    end

    # Replace the merged morphemes in the stack
    s[i_max] = (w_, r_, s[i_max+1][3], s̄)
    p.last_merge = i_max
    p.last_merged_morpheme = s[i_max][1:3]
    update_stack!(l, i_max+1, w, r, t, [i])
    p.r[i_max] = update_count!(l, s[i_max][1], s[i_max+1][1])

    # Update R matrix for the predecessor of the merged morpheme
    (i_max > 1) ? p.r[i_max-1] = update_count!(l, s[i_max-1][1], s[i_max][1]) :
    !isempty(p.out[1]) && update_count!(l, p.out[1], s[1][1])

    # Update the parser stack & memory morpheme values: backprop through time
    update_values!(l, i_max)

    return (M[], R(0.), T(0))
end

""" Find morpheme mean reward """
function get_reward(w::Morpheme, l::Layer)::R
    mapreduce(h -> h.r[w[h.N]], +, l.semantic) / l.N
end

""" Update layer morpheme mean reward """
function update_reward!(l::Layer, w::Morpheme, r::R,)::R
    if (isempty(w) || !l.update_rewards)
        return R(0.)
    elseif (l.N == 1)
        return update_reward!(l.semantic[1], w[1], r)
    end
    return mapreduce(n -> update_reward!(l.semantic[n], w[n], r), +, 1:l.N) / l.N
end

function update_reward!(h::Head, w::M, r::R)::R
    h.n[w] += 1
    h.r[w] += (r - h.r[w]) / h.n[w]
    return h.r[w]
end

""" Update parser stack & memory morpheme values """
function update_values!(l::Layer, w::Morpheme)::Nothing
    p = l.parser
    if (isempty(p.stack[end][1]))
        return nothing
    end
    update_value!(l, p.stack[end][1], p.stack[end][2], w)
    update_values!(l, l.parser.O)
    return nothing
end

function update_values!(l::Layer, i::Int)::Nothing
    update_stack_values!(l, i)
    p = l.parser
    if (!p.memorise || (p.recent < 1) || isempty(p.stack[1][1]))
        return nothing
    end
    update_value!(l, p.memory[p.recent, :], p.reward[p.recent], p.stack[1][1])
    update_memory_values!(l)
    return nothing
end

function update_stack_values!(l::Layer, i::Int=l.parser.O)::Nothing
    @assert (1 <= i <= l.parser.O)
    s = l.parser.stack
    if (i == 1 && !isempty(s[i][1]) && !isempty(s[i+1][1]))
        update_value!(l, s[i][1], s[i][2], s[i+1][1])
    elseif (i <= l.parser.O) && !isempty(s[i+1][1])
        for j ∈ reverse(1:i)
            isempty(s[i][1]) && break
            update_value!(l, s[j][1], s[j][2], s[j+1][1])
        end
    end
    return nothing
end

""" Update parser memory morpheme values """
function update_memory_values!(l::Layer)::Nothing
    p = l.parser
    if (!p.memorise)
        return nothing
    end
    if (p.recent > 1)
        for i ∈ reverse(1 : p.recent-1)
            update_value!(l, p.memory[i,:], p.reward[i], p.memory[i+1,:])
        end
    end
    if (p.memory_full) && (p.recent != p.memory_length)
        update_value!(l, p.memory[end,:], p.reward[end], p.memory[1,:])
        if (p.recent+1) < p.memory_length
            for i ∈ reverse(p.recent+1 : p.memory_length-1)
                update_value!(l, p.memory[i,:], p.reward[i], p.memory[i+1,:])
            end
        end
    end
    return nothing
end

""" Update morpheme value : local or global """
function update_value!(l::Layer, w::Morpheme, r::R, w_::Morpheme)::R
    if (!l.update_values)
        return R(0.)
    end
    if (l.global_value_update)
        return update_value_globally!(l, w, r, w_)
    end
    return update_value_locally!(l, w, r, w_)
end

""" Local value update v.0.4.1 """
function update_value_locally!(l::Layer, w::Morpheme, r::R, w_::Morpheme)::R
    isempty(w) ? R(0.) : isempty(w_) ?
    mapreduce(h -> update_value!(h, w[h.N], r, M(0)   ), +, l.semantic) / l.N :
    mapreduce(h -> update_value!(h, w[h.N], r, w_[h.N]), +, l.semantic) / l.N
end

function update_value!(h::Head, w::M, r::R, w_::M)::R
    v_ = (w_< 1) ? r : h.v[w_][2] * h.discount + r
    h.v[w][5] += R(1.)
    if (h.v[w][5] == R(1.))
        h.v[w][2] = v_
    else
        h.v[w][2] += h.learning_rate * (v_ - h.v[w][2]) / h.v[w][5]
    end
    (h.v[w][2] > h.v[w][1]) && (h.v[w][1] = h.v[w][2])
    return h.v[w][h.values]
end

""" Global value update v.0.4.2 """
function update_value_globally!(l::Layer, w::Morpheme, r::R, w_::Morpheme)::R
    isempty(w)  ? R(0.) :
    isempty(w_) ? update_value!(l, w, (r - get_value(w,l))) :
    mapreduce(h -> update_value!(h, w[h.N], Δv(w, r, w_, l)), +, l.semantic)
end

function Δv(w::Morpheme, r::R, w_::Morpheme, l::Layer)
    l.learning_rate[1] * (r + l.discount[1] * get_value(w_,l)) - get_value(w,l)
    # α[1], γ[1] - to keep global & local v updates compatible #v042
end

function update_value!(l::Layer, w::Morpheme, Δv::R)::R
    isempty(w) ? R(0.) :
    mapreduce(h -> update_value!(h, w[h.N], Δv), +, l.semantic)
end

function update_value!(h::Head, w::M, Δv::R)::R
    (w < M(1)) && return R(0.)
    h.v[w][5] += R(1)  # count
    h.v[w][2] += Δv / h.v[w][5]  # TODO? α/time for global value function ⇑ ?
    (h.v[w][2] > h.v[w][1]) && (h.v[w][1] = h.v[w][2])  # maх
    return h.v[w][h.values]
end

function mergeable(l::Layer, i::Int)::Bool
    p = l.parser

    w = p.stack[i][1]
    w′ = p.stack[i+1][1]

    if isempty(w) || isempty(w′)
        return false
    end

    ww::Morpheme = map(h -> get(h.w, (w[h.N], w′[h.N]), M(0)), l.semantic)
    if l.stop_growth && !haskey(l.index, ww)
        return false
    end

    h = l.semantic[1]
    if length(h.string[w[1]]) + length(h.string[w′[1]]) > l.l
        return false
    end

    if (p.r[i] < p.r[i+1]) || (i > 1 && p.r[i] < p.r[i - 1])
        return false
    end
    
    return true 
end

""" Update parser stack & memory: push new morpheme, update pair counts & values """
function update_memory!(l::Layer, w::Morpheme, r::R, t::T, s̄)::Nothing
    update_stack!(l, 1, w, r, t, s̄)
    p = l.parser
    if (p.memorise && !isempty(p.out))
        update_memory!(p)
        # Update morpheme pair count
        if (p.recent > 1)
            update_count!(l, p.memory[p.recent-1,:], p.memory[p.recent,:])
        elseif (p.memory_full && (p.recent == 1))
            update_count!(l, p.memory[end,:], p.memory[1,:])
        end
    end
    return nothing
end

""" Update parser stack: push new morpheme, update pair count """
function update_stack!(l::Layer, i::Int, w::Morpheme, r::R, t::T, s̄::Vector{I})::Nothing
    p = l.parser
    s = l.parser.stack
    p.out = (i==1) ? s[1][1:3] : (M[], R(0.), T(0))
    i < l.parser.O ? (
        s[i:(end-1)] = s[(i+1):end];
        p.r[i:(end-1)] = p.r[(i+1):end]
    ) : (i == l.parser.O) && (s[end-1] = s[end])
    s[end] = (w, r, t, s̄)
    p.r[end] = typemin(R)
    p.r[end-1] = isempty(w) ? R(0.) : update_count!(l, s[end-1][1], w)
    return nothing
end

""" Update morpheme correlation matrices and layer (next morphemes) memory """
function update_count!(l::Layer, w::Morpheme, w′::Morpheme, r::R=R(1.))::R
    if (invalid(w) || invalid(w′))
        return R(0.)
    end
    # Update layer memory: next morphemes
    push!(l.memory[l.index[w]], l.index[w′])
    # Update head morpheme correlation matrices
    if (l.N == 1)
        return log(update_count!(l.semantic[1], w[1], w′[1], r))
    end
    return mapreduce(h -> log(update_count!(h, w[h.N], w′[h.N], r)), +, l.semantic) / l.N
end

function update_count!(h::Head, i::M, j::M, r::R=R(1.))::R
    (i < 1 || j < 1) && return R(0.)
    r_ = get(h.R[i], j, R(0.)) + r
    ((r == R(0.)) && (r_ > R(0.))) && return r_
    h.R[i][j]  = r_
    h.Rt[j][i] = r_
    (r == R(0.)) && return r_
    h.R_i[i]  += C(r)
    h.R_f[j]  += C(r)
    h.R_if    += C(r)
    return r_
end

""" Update parser memory: push morpheme & reward from parser.out """
function update_memory!(p::Parser)::Nothing
    if (!p.memorise || isempty(p.out[1]))
        return nothing
    end
    if (p.recent < p.memory_length)
        p.recent += 1
    else
        p.recent = 1
        p.memory_full = true
    end
    p.memory[p.recent, :] = p.out[1]
    p.reward[p.recent   ] = p.out[2]
    return nothing
end

""" Create a new morpheme in a layer merging two layer morphemes """
function merge_morphemes!(l::Layer, w1::Morpheme, w2::Morpheme, r1::R, r2::R,
                          s̄::Vector{I})::Tuple{Morpheme, R, R, R}  # w, reward, value, probability
    # Check if the merged morpheme ww ← w1&w2 already exists
    ww::Morpheme = map(h -> get(h.w, (w1[h.N], w2[h.N]), M(0)), l.semantic)
    if (haskey(l.index, ww))
        m = l.index[ww]
        if (s̄ ∉ l.sequences[m])
            push!(l.sequences[m], s̄)
        end
        return (ww, r1+r2, get_value(ww, l), get_count(w1, w2, l))
    end

    # Check layer & heads have capacity to add morphemes
    l.stop_growth && return (M[], R(0.), R(0.), R(0.))
    if (l.n >= l.W) || (all(map(h -> (h.D >= l.M), l.semantic)))
        l.stop_growth = true
        return (M[], R(0.), R(0.), R(0.))
    end

    # Check all the heads are ready to merge morphemes
    if any(map(h -> h.R[w1[h.N]][w2[h.N]] < h.semantic_threshold, l.semantic))
        return (M[], R(0.), R(0.), R(0.))
    end

    # Create a new morpheme ww merging w1 & w2
    head_merge_results::Vector{Tuple{M, R, R, R}} =
        map(n -> merge_morphemes!(l.semantic[n], w1[n], w2[n], r1, r2, R(1)), 1:l.N)
    ww = map(x -> x[1], head_merge_results)
    if (M(0) ∈ ww)
        return (M[], R(0.), R(0.), R(0.))
    end

    # Check morphemes is next layer symbols
    if ((l.next_layer_symbol[l.index[w1]]) && (l.next_layer_symbol[l.index[w2]]))
        return (M[], R(0.), R(0.), R(0.))
    end

    m = add_layer_morpheme!(l, ww, s̄)
    if (m < 1)
        return (M[], R(0.), R(0.), R(0.))
    end

    reward::R = r1 + r2
    value::R = mapreduce(x -> x[3], +, head_merge_results)/l.N
    count::R = mapreduce(x -> x[4], +, head_merge_results)/l.N

    return (ww, reward, value, count)
end

function get_value(w::Morpheme, l::Layer)::R
    invalid(w) ? R(0.) : l.values[1] ∉ [1,2,3] ? R(0.) :
    mapreduce(h -> h.v[w[h.N]][l.values[h.N]], +, l.semantic) / l.N
end

function get_count(w::Morpheme, w′::Morpheme, l::Layer)::R
    invalid(w) || invalid(w′) ? typemin(R) :
    mapreduce(h -> log(get(h.R[w[h.N]], w′[h.N], R(0.))), +, l.semantic) / l.N
end

function get_probability(w::Morpheme, w′::Morpheme, l::Layer)::R
    if (invalid(w) || invalid(w′))
        return R(0.)
    end
    if (!all(map(h -> haskey(h.R[w[h.N]], w′[h.N]), l.semantic)))
        return R(0.)
    end
    return exp(mapreduce(h -> log(h.R[w[h.N]][w′[h.N]] / h.R_i[w[h.N]]), +, l.semantic) / l.N)
end

invalid(w::Vector{M})::Bool = (isempty(w) || (M(0) ∈ w))
  valid(w::Vector{M})::Bool = (!isempty(w) & (M(0) ∉ w))

""" Create a new morpheme in a head merging two head morphemes """
function merge_morphemes!(h::Head, w::M, w_::M, r::R=R(0.), r_::R=R(0.),
                          count::R=R(1.))::Tuple{M, R, R, R}
    i::M = (w,w_) in keys(h.w) ? h.w[(w,w_)] :
        (length(h.w) < h.M) ? new_morpheme!(h, w, w_, r+r_) : M(0)
    return i == M(0) ? (i, R(0.), R(0.), R(0.)) :
        (i, r+r_, update_value!(h, i, r+r_), update_count!(h, w, w_, count))
end

function new_morpheme!(l::Layer, w::Morpheme, w_::Morpheme, r::R=R(0.))::Morpheme
    check(map(h -> new_morpheme!(h, w[h.N], w_[h.N], r), l.semantic))
end

function check(w::Vector{M})::Vector{M}
    isempty(w) || M(0) in w ? M[] : w
end

function new_morpheme!(h::Head, w::M, w_::M, r::R=R(0.))::M
    haskey(h.w, (w,w_)) && return h.w[(w, w_)]
    (h.D >= h.M)        && return M(0)
    h.R[w][w_] < h.semantic_threshold && return M(0)

    w_string = [h.string[w]; h.string[w_]]
    (length(w_string) > h.l) && return M(0)

    i::M = M(0)
    if (haskey(h.string_idx, w_string))
        i = h.string_idx[w_string]
        h.w[(w,w_)] = i
        return i
    end

    h.D += 1
    i = M(h.D)
    h.w[(w,w_)] = i
    h.string[i] = w_string
    h.string_idx[w_string] = i
    h.R_i[i] = C(0.)
    h.R_f[i] = C(0.)
    h.n[i] = 1
    h.r[i] = r
    v = h.v[w][2]   # Shumsky: V(lr) = V(l)
    h.v[i] = [v, v, R(0.), R(0.), R(1.)]  # max, avg, 2*custom, count

    return i
end

""" Clear the parser stack by pushing empty morphemes, return stack morphemes merged """
function clear_parser!(l::Layer)::Tuple{Vector{Morpheme}, Vector{R}, Vector{T}}
    wv::Vector{Morpheme} = Morpheme[]
    rv::Vector{R} = R[]
    tv::Vector{T} = T[]
    for _ ∈ 1 : l.parser.O + 2
        (w, r, t) = parse_input!(l, I(0), M[], R(0.), T(0))
        (!isempty(w) && ∉(M(0), w)) && (
            push!(wv, w); push!(rv, r); push!(tv, t))
    end
    l.parser.last_merge = 0
    l.parser.last_merged_morpheme = (M[], R(0.), T(0))
    clear_parser_memory!(l.parser)
    return (wv, rv, tv)
end

function clear_parser_memory!(p::Parser)::Nothing
    p.memory_full = false
    p.recent = 0
    if (p.memory_length > 0)
        p.memory[:,:] .= M(0)
        p.reward[:  ] .= typemin(R)
    end
    return nothing
end

function actual_morphemes!(a::Adam, L::Int)::Vector{Tuple{Vector{M}, R, T}}
    l = a.layers[L]
    next_l = a.layers[L+1]

    p = l.parser
    out = Tuple{Vector{M}, R, T}[]
    for i ∈ 1:p.O-1
        if p.stack[i] == (M[], R(0.), T(0), I[])
            continue
        end
        w = p.stack[i][1]
        if !haskey(next_l.symbol_index, w)
            break
        end
        p.out = p.stack[i][1:3]
        push!(out, p.out)
        p.r[i] = typemin(R)
        p.stack[i] = (M[], R(0.), T(0), I[])
        if (p.memorise)
            update_memory!(p)
        end
    end

    if !isempty(out) 
        p.last_merge = 0
        p.last_merged_morpheme = (M[], R(0.), T(0))
    end

    return out
end