lib_path = joinpath(dirname(dirname(@__DIR__)),"src")
if (lib_path ∉ LOAD_PATH) push!(LOAD_PATH, lib_path) end
app_path = abspath(@__DIR__)
(app_path ∉ LOAD_PATH) && push!(LOAD_PATH, app_path)

using ADAM
using NLP_app

import Dates: Dates, now, format
import JLD2: save_object

ENV["GKSwstype"] = "100"

run_learning(app_path)