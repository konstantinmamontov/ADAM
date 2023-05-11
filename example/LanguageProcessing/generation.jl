

random_wl(l::Layer)::Morpheme =
    rand(l.symbols)

function choose_plan(plans::Vector{Plan})::Plan
    probabilities = map(p -> p[3], plans)
    probabilities = probabilities ./ sum(probabilities)

    j = findfirst(cumsum(probabilities) .> rand())
    if j === nothing
        j = 1
    end

    return plans[j]
end

active_plan(plan::Plan, t::T, w::Morpheme)::Bool =
    (length(plan[1]) > t - plan[end] + 1) && (
        plan[1][t - plan[end] + 1] == w)

function choose_morphemes!(plans::Vector{Plan}, t::T)::Vector{Morpheme}

    isempty(plans) && return Morpheme[]

    plan = choose_plan(plans)
    w = plan[1][t - plan[end] + 1]

    filter!(plan -> active_plan(plan, t, w), plans)

    return Morpheme[[w]; choose_morphemes!(plans, t + 1)]
end

function send_morphemes!(a::Adam, L::Int, morphemes::Vector{Morpheme},
    t::T, r::R=R(0.0))::Tuple{Vector{Plan}, T}

    plans = Plan[]

    for w ∈ morphemes

        plans = forward_pass!(a, w, r, t, L)
        t += T(1)

    end

    return plans, t
end

function last_layer_generation!(a::Adam, L::Int, iterations::Int,
    starting_morphemes::Vector{Morpheme}, logs = [],
    detailed::Bool=true)::String

    t_0 = Dates.now()
    msg_to_logs(string("layer #$L generation started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    l = a.layers[L]

    aL = a.L
    a.L = L
    stop_growth = l.stop_growth
    (L == 1) && (l.stop_growth = true)

    t = T(1)
    r = R(0.0)

    generated_morphemes = Vector{Morpheme}[]
    plans_nums = Int[]

    clear_parser!(l)
    clear_planner!(l)
    (plans, t) = send_morphemes!(a, L, starting_morphemes, t, r)
    push!(plans_nums, length(plans))
    
    for _ ∈ 1:iterations

        new_morphemes = choose_morphemes!(plans, t)
        isempty(new_morphemes) && (new_morphemes = [random_wl(l)])

        (plans, t) = send_morphemes!(a, L, new_morphemes, t, r)
        push!(plans_nums, length(plans))
        push!(generated_morphemes, new_morphemes)         
    end

    msg_to_logs("numbers of plans: $plans_nums\n", logs)

    t_1 = Dates.now()
    msg_to_logs(string("layer #$L generation completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    if detailed

        text::String = ""

        for morphemes ∈ generated_morphemes

            for w ∈ morphemes
                text = text * decode(w, L, a) * "|"
            end
            
            text = text * "|"
        end

        return text
    end

    l.stop_growth = stop_growth
    a.L = aL

    return decode(vcat(generated_morphemes...), L, a)
end

function make_seed_morphemes!(a::Adam, L::Int, text::String)::Vector{Morpheme}

    L == 1 && return filter(s -> !isempty(s), encode(text, a))

    l = a.layers[1]
    clear_parser!(l)

    mv = text_reparsing_history!(a, text)

    for n ∈ 2:(L-1)
        l = a.layers[n]
        clear_parser!(l)

        mv = reparse_morphemes!(a, n, mv)
    end

    return map(m -> a.layers[L-1].morphemes[m], filter!(m -> m > I(0), mv))
end

function last_layer_generation_telegram!(a::Adam, L::Int, iterations::Int,
    starting_morphemes::Vector{Morpheme}, logs = [],
    detailed::Bool=true)::String

    t_0 = Dates.now()
    msg_to_logs(string("layer #$L generation started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    l = a.layers[L]

    aL = a.L
    a.L = L
    stop_growth = l.stop_growth
    (L == 1) && (l.stop_growth = true)

    t = T(1)
    r = R(0.0)

    generated_morphemes = Vector{Morpheme}[]
    plans_nums = Int[]

    clear_parser!(l)
    clear_planner!(l)
    (plans, t) = send_morphemes!(a, L, starting_morphemes, t, r)
    push!(plans_nums, length(plans))
    
    break_symbols = ['.', '?', '!']

    for _ ∈ 1:iterations

        new_morphemes = choose_morphemes!(plans, t)
        isempty(new_morphemes) && (new_morphemes = [random_wl(l)])

        push!(generated_morphemes, new_morphemes)

        new_text = decode(new_morphemes, L, a)

        if any(in.(break_symbols, (new_text,)) .> 0)
            break
        end

        (plans, t) = send_morphemes!(a, L, new_morphemes, t, r)
        push!(plans_nums, length(plans))
    end

    msg_to_logs("numbers of plans: $plans_nums\n", logs)

    t_1 = Dates.now()
    msg_to_logs(string("layer #$L generation completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    if detailed

        text::String = ""

        for morphemes ∈ generated_morphemes

            for w ∈ morphemes
                text = text * decode(w, L, a) * "|"
            end
            
            text = text * "|"
        end

        return text
    end

    l.stop_growth = stop_growth
    a.L = aL

    return decode(vcat(generated_morphemes...), L, a)
end

function all_layers_generation!(a::Adam, iterations::Int,
    starting_morphemes::Vector{Morpheme}, logs = [], interaction_debug::Bool = false)::String

    t_0 = Dates.now()
    msg_to_logs(string("$(a.L) layers generation started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    stop_growth = a.layers[1].stop_growth
    (a.L == 1) && (a.layers[1].stop_growth = true)
    
    for l in a.layers
        clear_parser!(l)
        clear_planner!(l)
        l.opt["validation_report"] = Dict()
    end

    t = T(1)
    plans = Plan[]
    generated_symbols = M[]
    plans_nums = Int[]

    for s ∈ starting_morphemes
        plans = forward_pass!(a, s, R(0.0), t)
        t += 1

        if interaction_debug && a.L > 1
            #print_validation_log!(a, logs)
            print_stacks(a, logs)
        end
    end

    for _ ∈ 1:iterations

        push!(plans_nums, length(plans))

        s = Morpheme()
        if !isempty(plans)
            plan = choose_plan(plans)
            s = plan[1][t - plan[end] + 1]
        else
            s = random_wl(a.layers[1])
        end

        push!(generated_symbols, s[1])

        plans = forward_pass!(a, s, R(0.0), t)
        t += 1

        if interaction_debug && a.L > 1
            #print_validation_log!(a, logs)
            print_stacks(a, logs)
        end
    end
    
    msg_to_logs("numbers of plans: $plans_nums\n", logs)

    t_1 = Dates.now()
    msg_to_logs(string("$(a.L) layers generation completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    a.layers[1].stop_growth = stop_growth

    return decode(generated_symbols, a)
end

function adam_perplexity!(a::Adam,
    text_morphemes::Vector{Morpheme}, logs = [])::Tuple{String, Float32}

    t_0 = Dates.now()
    msg_to_logs(string("$(a.L) layers perplexity started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)
    
    for l in a.layers
        clear_parser!(l)
        clear_planner!(l)
        l.opt["validation_report"] = Dict()
    end

    t = T(1)
    plans = Plan[]
    generated_symbols = M[]

    s = Morpheme()
    s_ = Morpheme()

    match = 0

    for s ∈ text_morphemes

        if s == s_
            match += 1
        end        

        plans = forward_pass!(a, s, R(0.0), t)
        t += 1
        if !isempty(plans)
            plan = choose_plan(plans)
            s_ = plan[1][t - plan[end] + 1]
        else
            s_ = random_wl(a.layers[1])
        end

        push!(generated_symbols, s_[1])

        print_validation_log!(a, logs)
    end

    perplexity = (length(text_morphemes)) / match

    t_1 = Dates.now()
    msg_to_logs(string("$(a.L) layers perplexity completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    return string(decode(generated_symbols, a)), float(perplexity)
end

function adam_perplexity_word!(a::Adam, L::Int,
    text::String, logs = [])::Tuple{String, Float32, String}

    t_0 = Dates.now()
    msg_to_logs(string("$(a.L) layers perplexity_word started: ",
    Dates.format(t_0, "yyyy-mm-dd HH:MM:SS"), "\n"), logs)

    l = a.layers[L]

    aL = a.L
    a.L = L
    stop_growth = l.stop_growth
    (L == 1) && (l.stop_growth = true)

    clear_parser!(l)
    clear_planner!(l)

    text = lowercase(text)
 
    array_text = split(text, " ")
    if array_text[1] == ""
        array_text = array_text[2:end]
    end

    text_for_adam = string(array_text[1]) * " "

    match = 0

    generated_string = ""

    csv_text ="Затравка| продолжение| догадка| совпадение\n"

    parser_order = a.layers[a.L].parser.O

    status = "-"

    for word in array_text[2:end-1]

        t = T(1)
        r = R(0.0)
    
        starting_morphemes = make_seed_morphemes!(a, a.L, text_for_adam)

        (plans, t) = send_morphemes!(a, L, starting_morphemes, t, r)
        new_morphemes = choose_morphemes!(plans, t)
        isempty(new_morphemes) && (new_morphemes = [random_wl(l)])
        
        new_text = decode(new_morphemes, L, a)

        i = findfirst('_', new_text)
        if !(i == nothing)
            new_text = new_text[1:i]
            if new_text == word * "_"
                match += 1
                status = "+"
            end
        end

        generated_string = generated_string * new_text
        text_for_adam = word * " "


        parser_w = ""
        for i in (parser_order+1):-1:1
            if !isempty(a.layers[a.L].parser.stack[i][1])
                parser_w = decode(a.layers[a.L].parser.stack[i][1], a.L + 1, a) * parser_w 
            else
                break
            end
        end

        csv_text = csv_text * parser_w * "| " * word * "_|" * new_text * " |" * status * "\n"
        status = "-"

    end

    l.stop_growth = stop_growth
    a.L = aL

    t_1 = Dates.now()
    msg_to_logs(string("$(a.L) layers perplexity_word completed: ",
    Dates.format(t_1, "yyyy-mm-dd HH:MM:SS"),
    " ($(time_interval(t_0, t_1)))\n\n"), logs)

    perplexity = 0.0
    if match != 0
        perplexity = (length(array_text)-2) / match
    end
    return generated_string, perplexity, csv_text
end

function print_validation_log!(a::Adam, logs = [])::Nothing

    l = a.layers[1]
    if l.opt["validation_report"] == Dict()
        return nothing
    end

    out = l.parser.out
    plans = l.opt["validation_report"]["plans"]
    final_plans = l.opt["validation_report"]["final_plans"]
    plans_num = l.opt["validation_report"]["plans_num"] = length(plans)
    stack = l.opt["validation_report"]["parser_stack"]
    stack_w_string = l.opt["validation_report"]["stack_w_string"]
    t = l.opt["validation_report"]["t"]
    enough_length = l.opt["validation_report"]["enough_length"]
    actual = l.opt["validation_report"]["actual_morphemes"]

    """
    if actual == 0
        return nothing
    end
    """

    msg_to_logs("t: $t\n", logs)

    msg_to_logs("stack:\n", logs)
    msg_to_logs(stack_to_string(stack, a), logs)
    msg_to_logs("stack string: $stack_w_string\n" , logs)
    msg_to_logs("parser.out: $(string("(", isempty(out[1]) ? "" : decode(out[1][1], a), ", ", out[2], ", ", out[3], ")"))\n", logs)

    msg_to_logs("plans number: $plans_num\n", logs)
    msg_to_logs("plans with enough length: $enough_length\n", logs)
    msg_to_logs("plans validated: $actual\n", logs)
    msg_to_logs("plans:\n", logs)

    for p ∈ plans
        msg_to_logs("($(string(map(s -> decode(s[1], a), p[1]))), $(p[2]), $(p[3]), $(p[4]))\n", logs)
    end

    msg_to_logs("\n", logs)

    msg_to_logs("final plans:\n", logs)

    for p ∈ final_plans
        msg_to_logs("($(string(join(map(s -> decode(s[1], a), p[1])))), $(p[2]), $(p[3]), $(p[4]))\n", logs)
    end

    msg_to_logs("\n", logs)

    l.opt["validation_report"] = Dict()

    return nothing
end

function stack_to_string(stack::Vector{Tuple{Morpheme, R, T, Vector{I}}}, a::Adam)
    return join(map(se -> stack_element_to_string(se, a), stack))
end

function stack_element_to_string(stack_element::Tuple{Morpheme, R, T, Vector{I}}, a::Adam)::String
    (w, r, t, iv) = stack_element
    return string("(", isempty(w) ? "" : decode(w[1], a), ", ", r, ", ", t, ", ", iv, ")\n")
end

function print_plans(a::Adam, logs = [])::Nothing

    n = 10
    a.L == 1 && return nothing
    l = a.layers[1]

    out = l.parser.out

    msg_to_logs("stack:\n", logs)
    msg_to_logs(stack_to_string(l.parser.stack, a), logs)
    msg_to_logs("parser.out: $(string("(", isempty(out[1]) ? "" : decode(out[1][1], a), ", ", out[2], ", ", out[3], ")"))\n", logs)

    source = l.opt["plans_source"]
    
    msg_to_logs("plans source: $source\n", logs)
    msg_to_logs("plans:\n", logs)
    for p ∈ l.opt["plans"]
        msg_to_logs("($(string(map(s -> decode(s[1], a), p[1]))), $(p[2]), $(p[3]), $(p[4]))\n", logs)
    end

    if source == "instant plans"
        msg_to_logs("old plans:\n", logs)
        n_ = min(length(l.opt["old_plans"]), n)
        for p ∈ l.opt["old_plans"][1:n_]
            msg_to_logs("($(string(map(s -> decode(s[1], a), p[1]))), $(p[2]), $(p[3]), $(p[4]))\n", logs)
        end

        msg_to_logs("new plans:\n", logs)
        n_ = min(length(l.opt["new_plans"]), n)
        for p ∈ l.opt["new_plans"][1:n_]
            msg_to_logs("($(string(map(s -> decode(s[1], a), p[1]))), $(p[2]), $(p[3]), $(p[4]))\n", logs)
        end
    end

    return nothing
end

function print_stacks(a::Adam, logs = [])::Nothing

    l = a.layers[1]

    stack = get(l.opt, "stack", Tuple{Morpheme, R, T, Vector{I}}[])
    if !isempty(stack)
        msg_to_logs("stack L$(l.L):\n", logs)
        msg_to_logs(stack_to_string(stack, a), logs)
        
        l.opt["stack"] = Tuple{Morpheme, R, T, Vector{I}}[]
    end

    return nothing
end