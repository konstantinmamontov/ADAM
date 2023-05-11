# Adaptive Deep Autonomous Machine (ADAM)  v.0.5.3                  # 2022-12-12
# Copyright (c) 2020-2022 Oleg Baskov https://github.com/OlegBaskov

module ADAM

import Base.Threads: @spawn, @threads, nthreads, threadid
import Clustering: kmeans, KmeansResult
import Dates: Dates, Millisecond, now
import Distributions: Poisson
import LinearAlgebra: dot, norm, axpy!, normalize, normalize!
import Logging: Logging, SimpleLogger, current_logger, global_logger, min_enabled_level
import Random: Random, randperm, shuffle!, bitrand
import Plots: plot, savefig, histogram

include("types.jl")
export  C, F, I, M, P, R, T, V,
        Float, Morpheme, Path, Plan, Prediction,
        Adam, Head, Episodic, Layer, Parser, Planner

include("codec.jl")     ;   export sparse_vector, sparse2dense, implant_coder!, new_coder, encode!, dense_vector!
include("creation.jl")  ;   export new_adam, new_layer
include("growth.jl")    ;   export add_layer!
include("initiation.jl");   export initiate!, reparse!
include("parsing.jl")   ;   export parse_input!, clear_parser!
include("planning.jl")  ;   export forward_pass!

include("interaction.jl")
export  clear_episodic!, decode_plans, end_epoch!, get_plans!, query!

include("storage.jl")
export  load_agent, load_jld, load_json, save_agent, save_jld, save_json

include("trial.jl")
export  explore_environment, run_tests!, test_learning!, test_learning_curves,
        test_loop!, test_learning_curve!  # v < 0.4.4 legacy

include("utils.jl")
export  countmap, mean, option, print_dict, print_fields, print_vec,
        save_report, sma, time_interval, tweak

end
