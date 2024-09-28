from utils_condqa import *
from prompt_list_condqa import *
import re
from neo4j import GraphDatabase

uri = "bolt://localhost:7688"
user = "neo4j"
password = "P@ssw0rd"

driver = GraphDatabase.driver(uri, auth=(user, password))
session = driver.session()

query_head_relations = """MATCH (entity)-[relation]->(x) WHERE toLower(entity.id) =~ $re RETURN type(relation) AS relation"""
query_tail_relations = """MATCH (x)-[relation]->(entity) WHERE toLower(entity.id) =~ $re RETURN type(relation) AS relation"""
query_head_entities_extract = """MATCH (headEntity)-[relation]->(entity) WHERE toLower(entity.id) = toLower($entity_id) AND type(relation) = $relation_name RETURN headEntity.id AS entity_found"""
query_tail_entities_extract = """MATCH (entity)-[relation]->(tailEntity) WHERE toLower(entity.id) = toLower($entity_id) AND type(relation) = $relation_name RETURN tailEntity.id AS entity_found"""

def execute_relation_query(cypher_query, entity_id):
    re = r"(?i).*\b"+entity_id+r"\b.*"
    try:
        result = session.run(cypher_query, parameters={"re":re})
        return [record["relation"] for record in result]
    except:
        return []

def execute_entity_search_query(cypher_query, entity_id, relation_name):
    result = session.run(cypher_query, parameters={"entity_id":entity_id, "relation_name": relation_name})
    return [record["entity_found"] for record in result]

def clean_relations(string, entity_id, head_relations):
    pattern = r"(?P<relation>[A-Z_]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations

def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)

def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt_condqa % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        
def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    query_relations_extract_head = query_head_relations
    head_relations = execute_relation_query(query_relations_extract_head, entity_id)
    
    query_relations_extract_tail = query_tail_relations 
    tail_relations = execute_relation_query(query_relations_extract_tail, entity_id)

    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal
    flag = False
    if len(total_relations) > args.width: 

        if args.prune_tools == "llm":
            prompt = construct_relation_prune_prompt(question, entity_name, total_relations[:2000], args)
            
            if (args.LLM_type).startswith("gpt"):
                result = run_gpt(prompt)
            elif args.LLM_type == "llama3":
                result = run_llama3(prompt)
            elif args.LLM_type == "llama2":
                result = run_llama2(prompt)
            elif args.LLM_type == "mixtral":
                result = run_mixtral(prompt)
            elif args.LLM_type == "mistral":
                    result = run_mistral(prompt)
            flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations) 
            
    elif len(total_relations) > 0:
        relations = []
        for relation in total_relations:
            score = round(float(1/len(total_relations)),2)
            if relation in head_relations:
                relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
            else:
                relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
        flag = True
        retrieve_relations_with_scores = relations 
        
    else: 
        flag = False

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    
def entity_search(entity, relation, head=True):
    # print("---inside entity search---")
    if head:
        tail_entities_extract = query_tail_entities_extract
        entities = execute_entity_search_query(tail_entities_extract, entity, relation)
    else:
        head_entities_extract = query_head_entities_extract
        entities = execute_entity_search_query(head_entities_extract, entity, relation)

    return entities

def entity_score(question, entity_candidates_id, score, relation, args):
    entity_candidates = entity_candidates_id #since there are no entity ids in condqa
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        if (args.LLM_type).startswith("gpt"):
            result = run_gpt(prompt)
        elif args.LLM_type == "llama3":
            result = run_llama3(prompt)
        elif args.LLM_type == "llama2":
            result = run_llama2(prompt)
        elif args.LLM_type == "mixtral":
            result = run_mixtral(prompt)
        elif args.LLM_type == "mistral":
            result = run_mistral(prompt)
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id
    
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head

def half_stop(question, cluster_chain_of_entities, depth, args, qid, question_type):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    prompt_template, answer = generate_answer(question, cluster_chain_of_entities, args, question_type)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset, ans_from="LLM+KG-half_stop", qid=qid, question_type=question_type, prompt_template=prompt_template)

def generate_answer(question, cluster_chain_of_entities, args, question_type): 
    if question_type == "Yes/no":
        prompt = condaqa_yn_ans_with_kg_prompt + "\n\nQuestion: " + '\n'
        prompt_template = "condaqa_yn_ans_with_kg_prompt"
    elif question_type == "Yes/no - conditions":
        prompt = condaqa_yn_ans_cond_with_kg_prompt + "\n\nQuestion: " + '\n'
        prompt_template = "condaqa_yn_ans_cond_with_kg_prompt"
    elif question_type == "span":
        prompt = condaqa_span_ans_with_kg_prompt + "\n\nQuestion: " + question + '\n'
        prompt_template = "condaqa_span_ans_with_kg_prompt"
    else:
        prompt = condaqa_span_ans_cond_with_kg_prompt + "\n\nQuestion: " + question + '\n'
        prompt_template = "condaqa_span_ans_cond_with_kg_prompt"
    
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'Output: '
    if (args.LLM_type).startswith("gpt"):
        result = run_gpt(prompt)
    elif args.LLM_type == "llama3":
        result = run_llama3(prompt)
    elif args.LLM_type == "llama2":
        result = run_llama2(prompt)
    elif args.LLM_type == "mixtral":
        result = run_mixtral(prompt)
    elif args.LLM_type == "mistral":
        result = run_mistral(prompt)
    return prompt_template, result

def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads

def reasoning(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    
    if (args.LLM_type).startswith("gpt"):
        response = run_gpt(prompt)
    elif args.LLM_type == "llama3":
        response = run_llama3(prompt)
    elif args.LLM_type == "llama2":
        response = run_llama2(prompt)
    elif args.LLM_type == "mixtral":
        response = run_mixtral(prompt)
    elif args.LLM_type == "mistral":
        response = run_mistral(prompt)
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response
    



