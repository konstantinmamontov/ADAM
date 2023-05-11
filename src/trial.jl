#=# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.0                  # 2022-09-12
# Copyright (c) 2020-2022 Oleg Baskov https://github.com/OlegBaskov

using PyCall

""" Perform tests of an Open AI Gym Classic Control environment based example """
function run_tests!(env::PyCall.PyObject, parameters::Dict, exploration_strategy,
                    finish_reward, score_function, success_criteria)
    start = Dates.now()
    println("\nТест " * Dates.format(start, "yyyy-mm-dd HH:MM:SS"))

    # Parameters

    adam_parameters = get(parameters, "ADAM_parameters"         , Dict())
    app_parameters  = get(parameters, "application_parameters"  , Dict())
    application     = get(app_parameters, "application", "undefined")

    ## Test parameters

    test_parameters = get(parameters, "test_parameters"         , Dict())

    check_stub      = get(test_parameters, "check_stub"         , false )
    check_training  = get(test_parameters, "check_training"     , false )
    data_dir        = get(test_parameters, "data_dir"           , "data")
    explore_env     = get(test_parameters, "explore_environment", true  )
    load_stub       = get(test_parameters, "load_agent"         , ""    )
    render          = get(test_parameters, "test_loop_rendering", true  )
    save_initiated  = get(test_parameters, "save_initiated_agent" ,
                      get(test_parameters, "save_initiated"     , "")   )
    save_trained    = get(test_parameters, "save_trained_agent" ,
                      get(test_parameters, "save_trained"       , "")   )
    save_data       = get(test_parameters, "save_data"          , false )
    save_env_exp    = get(test_parameters, "save_exploration_data", false )
    save_parameters = get(test_parameters, "save_parameters"    , true  )
    save_report     = get(test_parameters, "save_report"        , true  )
    scatter_plot    = get(test_parameters, "save_scatter_plot"  , true  )
    sma_sum_plot    = get(test_parameters, "save_sma_sum_plot"  , true  )
    sma_plots       = get(test_parameters, "save_sma_plots"     , true  )
    sampling_width  = get(test_parameters, "sma_sampling_width" , 10    )
    training_set    = get(test_parameters, "exploration_data"   ,
                                           "environment_exploration_data.jld")
    parameters_path = get(test_parameters, "parameters_path", "test.json")
    training_set_path = abspath(joinpath(dirname(parameters_path), data_dir, training_set))

    ## Update debug mode

    verbose = get(test_parameters, "verbose", 2)
    log_level = [Logging.Error, Logging.Warn, Logging.Info, Logging.Debug]
    logger = global_logger(SimpleLogger(stdout, log_level[verbose]))
    (verbose > 2) && (print_dict(parameters); @show nthreads())
    @info parameters_path
    @info training_set_path

    ## Report title, path

    title = split((basename(parameters_path)), ".")[1] * "_" *
            Dates.format(start, "yymmdd-HHMMSS")
    test_parameters["title"] = title
    out_dir = joinpath(dirname(abspath(parameters_path)),
                       get(test_parameters, "output_dir", "output"))
    !isdir(out_dir) && mkdir(out_dir)
    report_path = abspath(joinpath(out_dir, title * "_report.json"))
    report = Dict()
    report["start"]  = Dates.format(start,  "yyyy-mm-dd HH:MM:SS")
    report["parameters_arg"] = parameters_path  # script argument

    ## Save a copy of parameters.json

    if (save_parameters)
        para_path = abspath(joinpath(out_dir, title * "_parameters.json"))
        report["parameters_path"] = cp(parameters_path, para_path)
    else
        report["parameters"]  = parameters
    end
    if (save_report)
        save_json(Dict(["test_report" => report]), report_path)
    end

    # Create or load agent

    if !isempty(load_stub)  ## Load agent from "data/load_stub .jld file

        data_dir = joinpath(dirname(abspath(parameters_path)),
                            get(test_parameters, "data_dir", "data"))
        @assert isdir(data_dir)
        path = joinpath(data_dir, load_stub)
        @info "load agent path" => path
        @info "abspath" => abspath(path)
        @assert isfile(path)
        adam = load_agent(path)
        @assert typeof(adam) == Adam

    else  ## Explore or load environment

        t01 = Dates.now()
        pv = test_parameters["verbose"]
        test_parameters["verbose"] = 2

        if (explore_env)  ## Explore environment
            @assert haskey(test_parameters, "exploration_episodes")
            @info "exploration episodes" => test_parameters["exploration_episodes"]
            println("\nИсследование окружения в ",
                    test_parameters["exploration_episodes"][1],
                    " эпизодах ", test_parameters["exploration_episodes"],
                    " начато ", Dates.format(t01, "yyyy-mm-dd HH:MM:SS"), ",")
            data = explore_environment(env, exploration_strategy,
                                       finish_reward, test_parameters)
            t02 = Dates.now()
            println("завершено ", Dates.format(t02, "yyyy-mm-dd HH:MM:SS"),
                    " за " * time_interval(t01, t02))
            @info "Track_length: " => length(data)
            if (save_env_exp)
                @info "saving" => training_set_path
                d = Dict()
                d["environment_exploration_data"] = data
                d["info"] = "CartPoleV1 environment exploration"
                d["path"] = abspath(joinpath(out_dir, training_set))
                d["parameters_path"] = parameters_path
                re_ = save_jld(d)
                @debug "save_jld_response" => re_
                report["training_set"] = re_["path"]
                println("Запись исследования окружения сохранена в файле\n",
                        report["training_set"])
            end
        else  ## Load environment exploration archive
            @info "training_set_path" => training_set_path
            if (!isfile(training_set_path))
                println("\nФайл записи исследования окружения не найден:\n",
                        training_set_path, '\n')
                return nothing
            end
            d::Dict = load_jld(training_set_path)
            @info "dataset_fields" => collect(keys(d))
            data = d["environment_exploration_data"]
            t02 = Dates.now()
            println("\nИсследование окружения не проводится, используется файл записи\n",
                    training_set_path)
        end
        test_parameters["verbose"]  = pv

        ## Create a new agent

        quit::String = !haskey(parameters, "ADAM_parameters") ?
            "В файле параметров отсутствует раздел \"ADAM_parameters\"" :
            isempty(app_parameters) ? "Параметры приложения не заданы"  : ""
        if !isempty(quit)
            println("\n" * quit * ", агент не создан.\n")
            if (save_report)
                report["finish"] = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
                re_ = save_json(report, report_path)
                println("Отчёт о тесте сохранен в файл\n", re_["path"], '\n')
            end
            return nothing
        end

        t03 = Dates.now()
        if (verbose > 2)
            println("\nСоздание и инициация агента начаты ",
                    Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
                    ", обучающая выборка - ", length(data), " шагов.")
            @time adam = new_adam(adam_parameters)
            @time initiate!(adam, app_parameters, data)
        else
            adam = new_adam(adam_parameters)
            initiate!(adam, app_parameters, data)
        end
        t04 = Dates.now()
        println("\nСоздание ", adam_parameters["heads"][1], "-голового агента ",
                adam_parameters["clusters"],
                " завершено ", Dates.format(t03, "yyyy-mm-dd HH:MM:SS"),
                " за " * time_interval(t03, t04))

        ## Save initiated agent

        if !isempty(save_initiated)
            t05 = Dates.now()
            print("Сохранение инициированного агента начато ",
                    Dates.format(t05, "yyyy-mm-dd HH:MM:SS"))
            path = joinpath(out_dir, title * "_" * save_initiated * "_agent.jld")
            report["initiated_agent"] = save_agent(adam, path)["path"]
            t06 = Dates.now()
            println(", завершено ", Dates.format(t06, "yyyy-mm-dd HH:MM:SS"),
                    " за ", time_interval(t05, t06))
            println("→ ", report["initiated_agent"])
        end

    end

    # Learning curves

    t11 = Dates.now()
    agents::Vector{Adam} = []
    if (verbose > 2)
        println("\nПостроение кривых обучения начато ",
                Dates.format(t11, "yyyy-mm-dd HH:MM:SS"), " : ",
                test_parameters["test_learning_curve"][1], " кривых по ",
                test_parameters["test_learning_curve"][2], " эпизодов.")
        @time (agents, scores, wins, periods) = test_learning_curves(adam, env,
            test_parameters, score_function, success_criteria, finish_reward)
    else
        println()
        (agents, scores, wins, periods) = test_learning_curves(adam, env,
            test_parameters, score_function, success_criteria, finish_reward)
    end

    report["scores"]  = map(x -> Int(round(mean(x))), scores)
    report["wins"]    = wins
    report["periods"] = map(x -> Int(round(x.value/1000)), periods)

    t12 = Dates.now()
    println("Построение кривых обучения завершено ",
            Dates.format(t12, "yyyy-mm-dd HH:MM:SS"),
            " за ", time_interval(t11, t12))


    # Save results

    ## Save plots

    plot_paths = []
    plot_title = "ADAM " * application * " learning curve\n" * title

    if (scatter_plot)
        p = plot(scores, seriestype = :scatter,
                title = plot_title, legend = false)
        plot_path = joinpath(out_dir, title * "_fig_01.png")
        savefig(plot_path)
        push!(plot_paths, plot_path)
    end

    if (sma_plots)
        p = plot([sma(x, sampling_width) for x in scores],
                 title = plot_title, legend=:outertopright,
                 label=permutedims(collect(1:length(scores))))
        plot_path = joinpath(out_dir, title * "_fig_02.png")
        savefig(plot_path)
        push!(plot_paths, plot_path)
    end

    if (sma_sum_plot && (length(scores) > 1))
        p = plot(sum([sma(x, sampling_width) for x in scores]) / length(scores),
                title = plot_title, legend = false)
        plot_path = joinpath(out_dir, title * "_fig_03.png")
        savefig(plot_path)
        push!(plot_paths, plot_path)
    end

    if !isempty(plot_paths)
        report["plot_paths"]  = plot_paths
    end

    ## Save data .jld

    if (save_data)
        data = Dict()
        data["report_path"] = report_path
        data["parameters_path"] = parameters_path
        data["data"] = scores
        data["path"] = joinpath(out_dir, title * "_data" * ".jld")
        re_ = save_jld(data)
        @info "save_jld(data) → response" => re_
        report["data_path"] = re_["path"]
    end

    ## Backup the trained agent → .jld file

    if !isempty(save_trained)
        t13 = Dates.now()
        (verbose > 1) && print("\nСохранение обученных агентов начато ",
                                Dates.format(t13, "yyyy-mm-dd HH:MM:SS"))
        path = joinpath(out_dir, title * "_" * save_trained * "_agent_")
        report["trained_agents"] = isempty(agents) ? [] :
            map(i -> save_agent(agents[i], path * string(i) * ".jld")["path"],
                                1:length(agents))

        t14 = Dates.now()
        if (verbose > 1)
            println(", завершено ", Dates.format(t14, "yyyy-mm-dd HH:MM:SS"),
                    " за ", time_interval(t13, t14))
            if (0 < length(report["trained_agents"]) < 2)
                println("Архив обученного агента (JLD2): ", report["trained_agents"][1])
            else
                println("Архивные файлы обученных агентов (JLD2):")
                for (i,path) in enumerate(report["trained_agents"])
                    println(string(i), ". ", report["trained_agents"][i])
                end
            end
        end
    end

    ## Save report .json

    finish = Dates.now()
    if (save_report)
        report["finish"] = Dates.format(finish, "yyyy-mm-dd HH:MM:SS")
        re_ = save_json(Dict(["test_report" => report]), report_path)
        @info  "Report save_json response" => re_
        println("\nОтчёт о тесте сохранен в файл " * re_["path"])
    else
        println("\nРезультаты теста сохранены в директории ", out_dir)
    end

    # Finish

    t15 = Dates.now()
    println("\nТест кривых обучения ADAM завершён ",
            Dates.format(t15, "yyyy-mm-dd HH:MM:SS"),
            ", общее время выполнения " * time_interval(start, t15), '\n')

    return nothing
end


""" Explore OpenAI Gym Classic Control environment using argument functions:
    - exploration_strategy(observation::Vector{F})::)::Int - returns action
    - finish_reward(time::Int)::R - returns reward upon done dependent on time
"""
function explore_environment(env::PyCall.PyObject, exploration_strategy,
                             finish_reward, parameters::Dict=Dict()
                            )::Vector{Tuple{Int, Vector{F}, R}}
    ntracks, tmin, tmax, tlimit = get(parameters, "exploration_episodes",
        [10, 0, 200, 200])
    step_reward::R = get(parameters, "step_reward", R(0.))
    explore_env_statistics = get(parameters, "explore_env_statistics", false)
    title = get(parameters, "title", "")
    out_dir = joinpath(dirname(get(parameters, "parameters_path", "test.json")), "output")
    !isdir(out_dir) && mkdir(out_dir)
    record::Vector{Tuple{Int, Vector{F}, R}} = []
    track_lengths::Vector{Int} = []
    n::Int = 1
    steps_done::Int = 1
    while n <= ntracks
        track = explore_track(env, exploration_strategy, finish_reward,
                              step_reward, tlimit)
        if (tmin <= length(track)-1 <= tmax)
            append!(record, track)
            n += 1
            push!(track_lengths, length(track)-1)
        end
        steps_done += 1
    end
    if (explore_env_statistics) 
        N_bins_hist = 25
        plot_title = "Explore environment track lengths"
        p = histogram(track_lengths, bins = N_bins_hist, 
                title = plot_title, legend = false)
        plot_path = joinpath(out_dir, title * "_expl_env.png")
        savefig(plot_path)
        println("Построена гистограмма распределения длин треков в исследовании окружения.")
        println("Сохранена в ", plot_path)
        """
        println("Границы квантилей распределения длин треков: ")
        quantiles = quantile(track_lengths, [0.1*i for i in 1:10])
        for i in 1:10
            println(i*10, "% - ", quantiles[i])
        end
        """
        # Выводим в консоль краткую сводку о длинах треков
        N_bins_4_print = 10
        max_len = maximum(track_lengths)
        min_len = minimum(track_lengths)
        step_len = (max_len - min_len)/N_bins_4_print
        lower_bound = 0
        step_num = 0
        upper_bound = round(min_len + step_len*step_num)
        while upper_bound <= max_len + 1
            println(lower_bound, "-", upper_bound, " шагов => ",
                  length(filter(x -> (x < upper_bound) && (x >= lower_bound), track_lengths)),
                  " треков")
            step_num += 1
            lower_bound = upper_bound
            upper_bound = round(min_len + step_len*step_num)
            (abs(max_len - upper_bound) < 2) && (upper_bound = max_len + 1)
        end
        println("Создано ", steps_done - 1, " треков. ", n - 1, 
                " использовано (находятся в диапазоне ", tmin, "-", tmax, ")")
    end
    return record
end

""" Explore a single episode of OpenAI Gym Classic Control environment """
function explore_track(env::PyCall.PyObject, exploration_strategy,
                       finish_reward, step_reward::R = R(0.), Δt::Int = 200
                      )::Vector{Tuple{Int, Vector{F}, R}}
    track::Vector{Tuple{Int, Vector{F}, R}} = []
    o::Vector{F} = map(x -> F(x), env.reset())
    reward::R = R(0.)
    terminated::Bool = false
    action::Int = typemin(Int)
    push!(track, (action, o, reward))
    action = exploration_strategy(o)
    for t ∈ 1 : Δt
        (observation, reward, terminated, truncated, info) = env.step(action)
        o = map(x -> F(x), observation)
        if (terminated)
            push!(track, (action, o, finish_reward(t)))
            break
        end
        push!(track, (action, o, step_reward))
        action = exploration_strategy(o)
    end
    return track
end

""" Test several learning curves, return all (n = n curves) trained agents """
function test_learning_curves(a::Adam, env::PyCall.PyObject, parameters::Dict,
                              score_function, success_criteria, finish_reward
                              )::Tuple{Vector{Adam}, Vector{Vector{Int}},
                                       Vector{Int},  Vector{Millisecond}}
    p_ = deepcopy(parameters)
    curves, eps, tmax = get(parameters, "test_learning_curve", [3, 100, 200])
    p_["test_learning_curve"][1] = 1  # test 1 curve
    scores::Vector{Vector{Int}} = [Int[] for _∈1:curves]
    wins::Vector{Int} = zeros(Int, 3)
    Δt::Vector{Millisecond} = zeros(Millisecond, 3)
    agents::Vector{Adam} = []
    for i ∈ 1 : curves
        a_ = deepcopy(a)
        scores[i], wins[i], Δt[i] = test_learning!(
            a_, env, p_, score_function, success_criteria, finish_reward)
        push!(agents, deepcopy(a_))
    end
    return agents, scores, wins, Δt
end

""" Test a single learning curve with `eps` episodes, mutate `a::Adam`! """
function test_learning!(a::Adam, env::PyCall.PyObject, p::Dict,
                        score_function, success_criteria, finish_reward
                        )::Tuple{Vector{Int}, Int, Millisecond}
    episodes, Δt =
        haskey(p, "test_learning_curve") ? p["test_learning_curve"][2:3] :
        haskey(p, "test_loop_episodes" ) ? p["test_loop_episodes"] : [100, 200]
    verbose::Int = get(p, "verbose", 2)
    start = Dates.now()
    loop = [test_episode!(a, env, finish_reward, Δt, verbose)
            for _ ∈ 1 : episodes]
    finish = Dates.now()
    score::Vector{Int} = map(x -> score_function(x), loop)
    wins::Int = count(x -> success_criteria(x), score)
    Δt = finish - start
    (verbose > 1) && println(
        wins, " побед в ", length(score), " эпизодах",
        ", средний балл ", round(mean(score), digits=1),
        "; Δt = ", time_interval(start, finish))
    return (score, wins, Δt)
end

""" Test a single learning episode (epoch) no longer than Δt """
function test_episode!(adam::Adam, env::PyCall.PyObject, finish_reward, Δt=200,
                       verbose=1)::Vector{Tuple{T, Vector{Plan}, Int}}
    actions::Vector{Int} = get(adam.app, "actions", [0, 2])
    sampling_p::F   = deepcopy(get(adam.app, "sampling_p0", 0.0))
    sampling::Bool  = sampling_p > 0.0
    sampling_t0::T  = get(adam.app, "sampling_t0", 1000)
    step_reward::R  = get(adam.app, "step_reward", [R(0.0)])[1]
    reward::R  = R(0.)
    terminated::Bool = false
    episode = []
    action::Int = typemin(Int)  # undefined action
    o::Vector{F} = map(x -> F(x), env.reset())
    plans  = get_plans!(adam, (action, o, R(0.))) # episode boundary
    action = decode_plans(plans, adam)  # empty plans, undef action?
    t::Int = 0
    push!(episode, (t, plans, action))

    for outer t ∈ 1 : Δt
        (observation, reward, terminated, truncated, info) = env.step(action)
        o = map(x -> F(x), observation)
        if (terminated)
            query!(adam, (action, o, finish_reward(t)))
            break
        end

        plans = get_plans!(adam, (action, o, step_reward))

        # Sampling
        if (sampling && (rand() < sampling_p))
            sampling_p = sampling_p / (1. + adam.layers[1].episodic.length / sampling_t0)
            action = rand(actions)
        else
            action = decode_plans(plans, adam)
        end

        push!(episode, (t, plans, action))
    end

    end_epoch!(adam)
    (verbose > 2) && print("test_episode!() finish: ", t, " ticks\n")
    return episode
end


# Legacy (v < 0.4.4) compatibility functions

""" Test multiple learning curves (v<0.4.4 legacy compatible, mutate `a::Adam` → last curve) """
function test_learning_curve!(a::Adam, env::PyCall.PyObject, parameters::Dict,
                              score_function, success_criteria, finish_reward
                              )::Vector{Vector{Int}}
    n = parameters["test_learning_curve"][1]  # number of curves
    if (n < 2)
        return [test_learning!(a, env, parameters, score_function,
                               success_criteria, finish_reward)]
    end

    scores::Vector{Vector{Int}} = [Int[] for _∈1:n]
    for i ∈ 1:n-1
        a_ = deepcopy(a)
        scores[i] = test_learning!(a_, env, parameters, score_function,
                                   success_criteria, finish_reward)[1]
    end
    scores[n] = test_learning!(a, env, parameters, score_function,
                               success_criteria, finish_reward)[1]
    return scores
end

""" Test agent `a` in OpenAI Gym environment `env` using `finish_reward` function(t) """
function test_loop!(a::Adam, env::PyCall.PyObject, finish_reward, parameters::Dict)
    n, tmax = get(parameters, "test_loop_episodes", [1, 200])
    v::Int  = get(parameters, "verbose", 2)
    return [test_episode!(a, env, finish_reward, tmax, v) for _∈1:n]
end=#
