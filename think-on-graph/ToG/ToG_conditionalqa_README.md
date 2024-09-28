# Think-On-Graph on ConditionalQA dataset

This repository includes the code for paper "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph" The link to the original repo is [here](https://github.com/IDEA-FinAI/ToG).

ConditionalQA dataset is a question answering dataset featuring complex questions with conditional answers where answers are only applicable if certain conditions apply. More information about the dataset can be found [here](https://haitian-sun.github.io/conditionalqa/).

## Project structure
* `CoT/` -- This folder contains experiments that correspond to the CoT and IO prompt. See `CoT/README.md` for details.
* `data/` -- This folder contains all datasets used in the paper. See `data/README.md` for details.
* `eval/` -- This folder contains evaluation scripts. See `eval/README.md` for details.
* `Freebase/` -- Freebase environment setting as mentioned in Think on Graph repo. See `Freebase/README.md` for details.
* `Wikidata/` -- Wikidata environment setting as mentioned in Think on Graph repo. See `Wikidata/README.md` for details.
* `ToG/` -- This folder contains source codes for Think on Graph approach. See `ToG/README.md` for details.
    - `client.py`: Pre-defined Wikidata APIs, copy from `Wikidata/`.
    - `freebase_func.py`: All the functions for querying Freebase KG used in `main_freebase.py`.
    - `kg_utils_condqa.py`: All the functions for querying KG used in `main_freebase_condqa.py`.
    - `main_freebase_condqa.py`: The main ToG algorithm using ConditionalQA dataset. See `ToG/README.md` for details.
    - `main_freebase.py`: The main file of ToG where Freebase as KG source. See `ToG/README.md` for details.
    - `prompt_list_condqa.py`: The prompts for the ToG with ConditionalQA dataset to pruning, reasoning and generating.
    - `prompt_list.py`: The prompts for the ToG to pruning, reasoning and generating.
    - `utils_condqa.py`: All the functions used in ToG with ConditionalQA dataset. 
    - `utils.py`: All the functions used in ToG.
* `virtuoso/` -- This folder contains the code for managing virtoso service.
* `requirements.txt` -- Pip environment file.

## Requirement installation
```
conda create --name tog_env python=3.9
conda activate tog_env
pip install -r requirements.txt
```

## Get started
Before running ToG on ConditionalQA dataset, please ensure that you have successfully setup a Knowledge Graph (KG) based on documents used in ConditionalQA dataset on your local machine. 

### Setting up ConditionalQA KG
To set up the ConditionalQA KG, you must run the LLM_KG pipeline to extract and load the Knowledge Graph in Neo4j.

#### Requirements
* **Neo4j Community Edition** (version 3.2 or higher)
* **APOC Plugin** for Neo4j

#### Configure Neo4j
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

We use vLLM for below models:

* **Mixtral-8x7B model**

    Initialise the model by running below command on a seperate terminal:
    ```
    python -m vllm.entrypoints.openai.api_server --model mistralai/Mixtral-8X7B-Instruct-v0.1 --tensor-parallel-size 4 --api-key token-abc123 
    ```
    Navigate to the `ToG/` folder and then run below command for Think on Graph with Mixtral model:
    ```
    python main_freebase_condqa.py --dataset conditionalqa --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0.1 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type mixtral --num_retain_entity 5  --prune_tools llm
    ```

* **Mistral model**
    
    For Mistral-7B-Instruct-v0.2 model, run this on a seperate terminal:
    ```
   python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --dtype float16 --api-key token-abc123 
    ```
    For running the code, navigate to `ToG/` folder and then run below command:
    ```
    python main_freebase_condqa.py --dataset conditionalqa --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0.1 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type mistral --num_retain_entity 5  --prune_tools llm
    ```

For OpenAI models, make sure you have OpenAI API key:

* **GPT 3.5 Turbo**
    
    Navigate to `ToG/` folder and then run below command:
    ```
    python main_freebase_condqa.py --dataset conditionalqa --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0.1 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type gpt-3.5-turbo --opeani_api_keys <ADD_OPENAI_API_KEY> --num_retain_entity 5  --prune_tools llm
    ```

* **GPT 4o**
    
    Navigate to `ToG/` folder and then run below command:
    ```
    python main_freebase_condqa.py --dataset conditionalqa --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0.1 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type gpt-4o --opeani_api_keys <ADD_OPENAI_API_KEY> --num_retain_entity 5  --prune_tools llm
    ```

See `ToG/README.md` for details about the parameters for `main_freebase_condqa.py`.

## Experiment results

| **Model**                                     | **ConditionalQA** |
|-----------------------------------------------|-------------------|
| GPT 3.5                                       | 16.29             |
| GPT 4o                                        | 20.40             |
| Mistral                                       | 16.24             |
| Mixtral                                       | 17.40             |
