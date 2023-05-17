This repository is designed to reproduce and verify the results published in the article "Neuro-symbolic conversational intelligence for a fraction of the cost".

The reproduction of the results consists of two consecutive stages:

1. Creating a language model
2. Testing the model using perplexity measurement

Prerequisites:

1. Hardware: 3.00 GHz Intel Xeon Gold 6154 processor (or similar) with 1 TB RAM, no GPU required
2. Software: Centos 9 operating system, Julia language version 1.6.7 with installed packages listed in the file Project.toml and dependencies listed in the file Manifest.toml

The reproduction of the results:

Step 1: preliminary preparation of the source code files.
After cloning the repo go to the src directory and run from the command line: unzip package_src.zip

Step 2: place the text corpus as a single utf-8 encoded text file in the directory ../example/LanguageProcessing/articles, specify the file name in the train parameters file "Learning_full_parameters.json" 

Step 3: to create a language model, run the following command: "julia --project=. ./example/LanguageProcessing/Learning_full.jl" from the root directory  

Step 4: to measure the perplexity, run the following command: "julia --project=. ./example/LanguageProcessing/PPW_test.jl" from the root directory 

"text_file"