

function run_chess_dialogue!(a::Adam)::Vector{String}

    stop_growth = a.layers[1].stop_growth
    (a.L == 1) && (a.layers[1].stop_growth = true)
        
    for l in a.layers
        clear_parser!(l)
        clear_planner!(l)
    end

    msg_to_logs("User: ", [Base.stdout])
    words = readline(Base.stdin)
    t = T(1)

    plans = Plan[]

    dialogue = String[]

    while words != "end"
        push!(dialogue, join(["User: ", words]))
        morphemes = filter(s -> !isempty(s), encode(words, a))

        for s ∈ morphemes
            plans = forward_pass!(a, s, R(0.0), t)
            t += 1
        end
    
        answer = ""
        s = Morpheme()
        if !isempty(plans)
            plan = choose_plan(plans)
            s = plan[1][t - plan[end] + 1]
            t += 1
        else
            break
        end
    
        letter = decode(s[1], a)
        answer = join([answer, decode(s[1], a)])
    
        while letter != "_"
            plans = forward_pass!(a, s, R(0.0), t)
            t += 1
            if !isempty(plans)
                plan = choose_plan(plans)
                s = plan[1][t - plan[end] + 1]
                letter = decode(s[1], a)
                answer = join([answer, decode(s[1], a)])
            else
                break
            end
        end

        plans = forward_pass!(a, s, R(0.0), t)
        t += 1

        push!(dialogue, join(["ADAM: ", answer]))
        msg_to_logs(join([dialogue[end], '\n']), [Base.stdout])
        msg_to_logs("User: ", [Base.stdout])
        words = readline(Base.stdin)
    end
    
    a.layers[1].stop_growth = stop_growth
    
    return dialogue
end

