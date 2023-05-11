

module NLP_app

using ADAM

import ADAM: encode, encode!, add_layer_morpheme!, clear_planner!, countmap
import Random: MersenneTwister, randn, bitrand
import LinearAlgebra: norm
import Clustering: kmeans
import Dates: Dates, now, format
import Printf: @sprintf
import Base.Threads: @threads
import Plots: savefig, histogram, title!, xlabel!, ylabel!, plot
import JLD2: save_object, load_object

include("init.jl")
export initialize, update!, clean_text!

include("coder.jl")
export encode, decode

include("exploration.jl")
export explore_text!, simplify_text!

include("text_parsing.jl")
export parse_text!, reparse_text!, training_reparse!

include("text_reparsing_separated.jl")
export text_reparsing_history!, text_reparsing_stats!

include("parsing_analytics.jl")
export text_parsing_results

include("layer_training.jl")
export train_layer!, reparse_morphemes!, expand_layer_morphemes!, filter_sample!

include("generation.jl")
export last_layer_generation!, make_seed_morphemes!,
all_layers_generation!, adam_perplexity!, adam_perplexity_word!,
last_layer_generation_telegram!

include("output.jl")
export msg_to_logs, log_msg, pretty_size,
show_morphemes, show_learned_words, show_learned_popular_words

include("synonyms.jl")
export find_synonyms, find_full_synonyms, find_same_phrases,
find_s_vector_synonyms, find_d_vector_synonyms

include("brain.jl")
export print_adam_statistics, R_matrix_distribution, frequent_morphemes, frequency_distribution

include("dialogue.jl")
export run_chess_dialogue!

import ADAM: normalize, sparse2dense, sparse2dense!, head_vector, new_coder,
reparse!, new_semantic!, new_morphemes!, invalid, new_layer_morphemes!,
allocate_episodic!

include("layer_creation_separated.jl")
export init_layer!, trim_R!, cut_context!

import ADAM: inherit
include("puff.jl")
export update_parameters!, save_empty_adam!, load_adam, expand_options!, cut_text!, cut_history, stop_growth!

import ADAM: invalid, get_count, mergeable, get_value, clear_parser_memory!
include("perplexity.jl")
export PPW

include("run.jl")
export run_learning

end