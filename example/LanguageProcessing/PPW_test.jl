lib_path = joinpath(dirname(dirname(@__DIR__)),"src")
if (lib_path ∉ LOAD_PATH) push!(LOAD_PATH, lib_path) end
app_path = abspath(@__DIR__)
(app_path ∉ LOAD_PATH) && push!(LOAD_PATH, app_path)

using ADAM
using NLP_app

import Dates: Dates, now, format
import JLD2: save_object, load_object

ENV["GKSwstype"] = "100"

start_time = Dates.now()

parameters = load_json(joinpath(app_path, "PPW_test_parameters.json"))
test_parameters = parameters["test_parameters"]

start = Dates.format(start_time, "yyyy_mm_dd: H_M_S")

today = split(start, ':')[1]
time = split(start, ' ')[2]
title = parameters["Test"] * "_" * time

report_path = mkpath(joinpath(app_path, "reports", today, title))

log_file = open(joinpath(report_path, "log.txt"), "w")
logs = [Base.stdout, log_file]

msg_to_logs(string("program execution started: ",
Dates.format(start_time, "yyyy-mm-dd HH:MM:SS"), "\n\n"), logs)

report_file = open(joinpath(report_path, "report.txt"), "w")

println(report_file, "Time: ", string(Dates.format(start_time, "yyyy-mm-dd HH:MM:SS")))
println(report_file, "Parameters:")
print_dict(parameters, report_file)

layer = get(test_parameters, "layer", 1)
trained = true
load_dir_date = get(test_parameters, "load_dir_date", "")
load_dir_time = get(test_parameters, "load_dir_time", "")

load_dir = joinpath(app_path, "saved_objects", load_dir_date, load_dir_time)
@time a = load_adam(load_dir, layer, trained)

text_file = joinpath(app_path, "articles",
    test_parameters["text_file"])
text = read(text_file, String)
text = clean_text!(text, a)

PPL = PPW(a, text, report_path)

end_time = Dates.now()
msg_to_logs(string("program execution completed: ",
Dates.format(end_time, "yyyy-mm-dd HH:MM:SS"),
" ($(time_interval(start_time, end_time)))"), logs);