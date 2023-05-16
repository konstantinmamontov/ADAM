This repository is designed to reproduce and verify the results published in the article "Neuro-symbolic conversational intelligence for a fraction of the cost".

The reproduction of the results consists of two consecutive stages:

1. Creating a language model
2. Testing the model using perplexity measurement

Prerequisites:

1. Hardware: 3.00 GHz Intel Xeon Gold 6154 processor with 1 TB RAM, no GPU
2. Software: Julia language version 1.6.7 with installed packages listed in the file Project.toml and dependencies listed in the file Manifest.toml

To create a language model, first place the text corpus as a single file in the directory ../LanguageProcessing/articles and then run the script Learning_full.jl

To measure the perplexity, run the script PPW_test.jl