from tqdm import tqdm
import argparse
from utils_condqa import *
from kg_utils_condqa import *
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="conditionalqa", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0.1, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="select llm for prune tools for ToG (same as LLM_type)")
    args = parser.parse_args()

    if args.LLM_type == "llama3":
        initialise_llama3(args.temperature_reasoning, args.max_length)
    elif args.LLM_type == "mixtral":
        initialise_mixtral(args.temperature_reasoning, args.max_length)
    elif args.LLM_type == "llama2":
        initialise_llama2(args.temperature_reasoning, args.max_length)
    elif args.LLM_type == "mistral":
        initialise_mistral(args.temperature_reasoning, args.max_length)
    elif args.LLM_type.startswith("gpt"):
        initialise_gpt(args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

    datas, question_string = prepare_dataset(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)
    
    for j, data in tqdm(enumerate(datas)):
        print("\n************ for question no: ", j+1)
        qid = data['id']
        print(qid)
        question = data[question_string]
        topic_entity = data['topic_entity']
        question_type = []
        if len(data['answers'])>0:
            answers = data['answers'][0]
            if answers[0] in ["yes", "no"]:
                if len(answers[1]) > 0:
                    question_type = "Yes/no - conditions"
                else:
                    question_type = "Yes/no"
            else:
                if len(answers[1]) > 0:
                    question_type = "span - conditions"
                else:
                    question_type = "span"
        else:
            question_type = "span"

        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            prompt_template, results = generate_without_explored_paths(question, args, question_type)
            save_2_jsonl(question, results, [], file_name=args.dataset,ans_from="LLM", qid=qid, question_type=question_type, prompt_template=prompt_template)
            continue
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if len(pre_heads)==0:
                    pre_heads_exists = False
                else:
                    pre_heads_exists = pre_heads[0]
                if entity!="[FINISH_ID]":
                    retrieve_relations_with_scores = relation_search_prune(entity, entity, pre_relations, pre_heads_exists, question, args)  ## topic_entity[entity] changed to entity as there is no entity id in condqa
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            for entity in current_entity_relations_list:
                try:
                    if entity['head']:
                        entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                    else:
                        entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                    
                    if args.prune_tools == "llm":
                        if len(entity_candidates_id) >=20:
                            entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                    if len(entity_candidates_id) ==0:
                        continue
                    scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                    
                    total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
                except:
                    continue
            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, depth, args, qid, question_type)
                flag_printed = True
                break
                
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    half_stop(question, cluster_chain_of_entities, depth, args, qid, question_type)
                    flag_printed = True
                    break
                else:
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args, qid, question_type)
                        flag_printed = True
            else:
                half_stop(question, cluster_chain_of_entities, depth, args, qid, question_type)
                flag_printed = True
        
        if not flag_printed:
            prompt_template, results = generate_without_explored_paths(question, args, question_type)
            save_2_jsonl(question, results, [], file_name=args.dataset, ans_from="LLM", qid=qid, question_type=question_type, prompt_template=prompt_template)
