# Think-On-Graph on ConditionalQA dataset

This repository includes the code for paper "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph" The link to the original repo is [here](https://github.com/IDEA-FinAI/ToG).

ConditionalQA dataset is a question answering dataset featuring complex questions with conditional answers where answers are only applicable if certain conditions apply. More information about the dataset can be found [here](https://haitian-sun.github.io/conditionalqa/).

## Project structure
* `data/` -- This folder contains all datasets used in the paper. See `data/README.md` for details.
* `eval/` -- This folder contains evaluation scripts.
* `ToG/` -- This folder contains source codes for Think on Graph approach. See `ToG/README.md` for details.
    - `kg_utils_condqa.py`: All the functions for querying KG used in `main_think_on_graph_condqa.py`.
    - `main_think_on_graph_condqa.py`: The main ToG algorithm using ConditionalQA dataset.
    - `prompt_list_condqa.py`: The prompts for LLMs to pruning, reasoning and generating.
    - `utils_condqa.py`: All the functions used in ToG with ConditionalQA dataset. 
* `requirements.txt` -- Pip environment file.

## Requirement installation
```
conda create --name tog_env python=3.9
conda activate tog_env
pip install -r requirements.txt
```
Make sure you have OpenAI API subscription to run any OpenAI models.
## Get started
Before running ToG on ConditionalQA dataset, please ensure that you have successfully setup a Knowledge Graph (KG) based on documents used in ConditionalQA dataset on your local machine. 

### Setting up ConditionalQA KG
To set up the ConditionalQA KG, you must run the LLM_KG pipeline to extract and load the Knowledge Graph in Neo4j.

#### Requirements
* **Neo4j Community Edition** (version 3.2 or higher)
* **APOC Plugin** for Neo4j

#### Configure KG for Think on Graph
Ensure that Neo4j is installed, and the server is running. You can configure the connection parameters in the `kg_utils_condqa.py` file inside the `ToG/` folder:

```
from neo4j import GraphDatabase

uri = "bolt://localhost:7688" # Default Neo4j Bolt URL 
user = <Replace with your Neo4j username>
password = <Replace with your Neo4j password>

driver = GraphDatabase.driver(uri, auth=(user, password))
session = driver.session()
```

## Running the experiments 

Navigate to the `ToG/` folder and then run below command:
```sh
python main_freebase_condqa.py \  # if you wanna use Wikidata as KG source, run main_wiki.py
--dataset conditionalqa \ # dataset your wanna test, see ToG/data/README.md
--max_length 256 \ 
--temperature_exploration 0.4 \ # the temperature in exploration stage.
--temperature_reasoning 0.1 \ # the temperature in reasoning stage.
--width 3 \ # choose the search width of ToG, 3 is the default setting.
--depth 3 \ # choose the search depth of ToG, 3 is the default setting.
--remove_unnecessary_rel True \ # whether removing unnecessary relations.
--LLM_type gpt-3.5-turbo \ # the LLM you choose
--opeani_api_keys sk-xxxx \ # your own api keys, if LLM_type == llama, this parameter would be rendered ineffective.
--num_retain_entity 5 \ # Number of entities retained during entities search.
--prune_tools llm \ # prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.
```
For example:
```sh
python main_freebase_condqa.py --dataset conditionalqa --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0.1 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type gpt-3.5-turbo --opeani_api_keys <ADD_OPENAI_API_KEY> --num_retain_entity 5  --prune_tools llm
```

LLM Models used: 
* Mixtral-8X7B-Instruct-v0.1 model
* Mistral-7B-Instruct-v0.2 model model
* GPT 3.5 Turbo
* GPT 4o

We use vLLM for Mistral and Mixtral models, so make sure you initialise the required vllm server:

For example:
```
python -m vllm.entrypoints.openai.api_server --model mistralai/Mixtral-8X7B-Instruct-v0.1 --tensor-parallel-size 4 --api-key token-abc123 
```

## Evaluation
The results of the predictions are evaluated against the reference answers using metrics like:

* EM (Exact Match): Measures if the predicted answer is exactly the same as the reference.
* Conditional EM: Evaluates the exact match under certain conditions.
* F1 Score: Measures the overlap between the predicted and reference answers.
* Conditional F1: Evaluates F1 score under specific conditions.

To run the evaluation script, run below code:
```
python evaluation_conditionalqa.py --pred_file <path_to_file>/ComplexQA/think-on-graph/output/ToG_conditionalqa.jsonl --ref_file <path_to_file>/ComplexQA/think-on-graph/data/conditional_qa_dev_entities_v3_gpt3.json
```