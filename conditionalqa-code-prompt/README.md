# ConditionalQA Code Prompt

This repository includes the code and prompts used for ConditionalQA dataset in 2024 arXiv paper "Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs." The link to the original repo is [here](https://github.com/UKPLab/arxiv2024-conditional-reasoning-llms).

## Project structure
### Scripts
* `conditionalqa_code_prompt.ipynb` -- This notebook runs `code prompts` on `ConditionalQA`
* `conditionalqa_text_prompt.ipynb` -- This notebook runs `text prompts` on `ConditionalQA`
  
### Backend
* `src` -- This folder contain the classes that define `text prompts` and `code prompts` for `ConditionalQA`.
* `data` -- This folder contains the training, dev, and ICL demonstrations used in the experiments (including ablations).
* `outputs` -- This folder contains all the prompts (inputs and outputs). It also includes the evaluation results of each prompt. 

## Requirements
* openai
* langchain
* scikit-learn
* vllm

You also need an Azure OpenAI or OpenAI API account and put your key in the notebook to run them.

## Installation
```
conda create --name code_prompting python=3.9
conda activate code_prompting
pip install -r requirements.txt
```

## Running the experiments 
Run these notebooks:
* `conditionalqa_code_prompt.ipynb`
* `conditionalqa_text_prompt.ipynb`

To reproduce results for OpenAI model, simply add the OpenAI API keys to the notebooks and run the notebook. 
