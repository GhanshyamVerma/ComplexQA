import json
import time
import openai
import re
from prompt_list_condqa import *
from langchain import PromptTemplate #,  LLMChain
import torch
from langchain_community.llms import VLLM
from langchain.chat_models import ChatOpenAI

llm = None
def initialise_llama3(temperature, max_tokens):
    print("--initialising llama3")
    global llm
    llm = VLLM(
            model = "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ",
            trust_remote_code=True,  
            max_new_tokens=max_tokens,
            top_k=-1,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.1,
            return_full_text=False
        )

def initialise_llama2(temperature, max_tokens):
    print("--initialising llama2")
    global llm
    llm = VLLM(
            model = "TheBloke/Llama-2-70B-Chat-GPTQ",
            trust_remote_code=True,  
            max_new_tokens=max_tokens,
            top_k=-1,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.1,
            return_full_text=False
        )

def initialise_mixtral(temperature, max_tokens):
    print("--initialising Mixtral")
    global llm
    # llm = VLLM(
    #         # model = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    #         model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    #         trust_remote_code=True,  
    #         dtype="float16",
    #         max_new_tokens=max_tokens,
    #         top_k=-1,
    #         top_p=0.95,
    #         temperature=temperature,
    #         repetition_penalty=1.1,
    #         return_full_text=False
    #     )
    inference_server_url = "http://localhost:8000/v1"
    llm = ChatOpenAI(
        model="mistralai/Mixtral-8X7B-Instruct-v0.1",
        openai_api_key="token-abc123",
        openai_api_base=inference_server_url,
        max_tokens=256,
        temperature=0,
    )

def initialise_mistral(temperature, max_tokens):
    print("--initialising Mistral")
    global llm
    llm = VLLM(
            model = "mistralai/Mistral-7B-Instruct-v0.2",
            trust_remote_code=True,  
            dtype="float16",
            max_new_tokens=max_tokens,
            top_k=-1,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.1,
            return_full_text=False
        )

def initialise_gpt(temperature, max_tokens, opeani_api_keys, model_name):
    print("--initialising gpt")
    global llm
    llm = ChatOpenAI(
    model=model_name,
    api_key=opeani_api_keys,
    temperature=temperature,
    max_tokens=max_tokens,
    request_timeout=30,
    max_retries=3,
    timeout=60 * 3,
)

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores

def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
    else:
        openai.api_key = opeani_api_keys

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    f = 0
    while(f == 0):
        try:
            response = openai.ChatCompletion.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response["choices"][0]['message']['content']
            f = 1
        except:
            print("openai error, retry")
            time.sleep(2)
    return result

def run_llama3(prompt_input):

    template="""You are an AI assistant that helps people find information.
    {prompt_input}
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm

    result = llm_chain.invoke(prompt_input)
    
    return result

def run_llama2(prompt_input):

    template="""[INST] <<SYS>> You are an AI assistant that helps people find information.<</SYS>>
    {prompt_input}[/INST]
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm

    result = llm_chain.invoke(prompt_input)
    
    return result

def run_mixtral(prompt_input):

    template="""<s>[INST] You are an AI assistant that helps people find information.
    {prompt_input} [/INST]
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm
    # print(prompt_input)
    result = llm_chain.invoke({"prompt_input": prompt_input})
    
    return result.content

def run_mistral(prompt_input):

    template="""<s>[INST] You are an AI assistant that helps people find information.
    {prompt_input} [/INST]
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm
    # print(prompt_input)
    result = llm_chain.invoke(prompt_input)
    
    return result

def run_gpt(prompt_input):

    template="""You are an AI assistant that helps people find information.
    {prompt_input}
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm
    result = llm_chain.invoke({"prompt_input": prompt_input})
    
    return result.content

def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)

def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates

def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        # print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    
def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name, ans_from, qid, question_type, prompt_template):
    dict = {"id": qid, "question_type": question_type, "ans_from": ans_from ,"answer": answer, "prompt_template": prompt_template, "reasoning_chains": cluster_chain_of_entities}
    with open("../output/ToG_{}_output.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    
def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False

def generate_without_explored_paths(question, args, question_type):
    # print("!! Answer without KG !!!")
    # prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    if question_type == "Yes/no":
        prompt = condaqa_yn_ans_without_kg_prompt + "\n\nQuestion: " + question + "\nOutput:"
        prompt_template = "condaqa_yn_ans_without_kg_prompt"
    elif question_type == "Yes/no - conditions":
        prompt = condaqa_yn_ans_cond_without_kg_prompt + "\n\nQuestion: " + question + "\nOutput:"
        prompt_template = "condaqa_yn_ans_cond_without_kg_prompt"
    elif question_type == "span":
        prompt = condaqa_span_ans_without_kg_prompt + "\n\nQuestion: " + question + "\nAnswer:"
        prompt_template = "condaqa_span_ans_without_kg_prompt"
    else:
        prompt = condaqa_span_ans_cond_without_kg_prompt + "\n\nQuestion: " + question + "\nAnswer:"
        prompt_template = "condaqa_span_ans_cond_without_kg_prompt"

    if (args.LLM_type).startswith("gpt"):
        # response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
        response = run_gpt(prompt)
    elif args.LLM_type == "llama3":
        response = run_llama3(prompt)
    elif args.LLM_type == "llama2":
        response = run_llama2(prompt)
    elif args.LLM_type == "mixtral":
        response = run_mixtral(prompt)
    elif args.LLM_type == "mistral":
        response = run_mistral(prompt)
    # print("!! LLM Answer\n", response)
    return prompt_template, response

def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst

def prepare_dataset(dataset_name):
    if dataset_name == 'conditionalqa':
        with open('../data/conditional_qa_dev_entities_v3_gpt3.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'context_question'
    else:
        print("dataset not found, you should pick from {conditionalqa}.")
        exit(-1)
    return datas, question_string