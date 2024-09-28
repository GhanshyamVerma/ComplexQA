import re
# Function to parse the generated text and extract the triplets
# Rebel outputs a specific format. This code is mostly copied from the model card!

def clean_triplets(input_text, triplets):
    """Sometimes the model hallucinates, so we filter out entities
       not present in the text"""
    text = input_text.lower()
    clean_triplets = []
    for triplet in triplets:

        if (triplet["head"] == triplet["tail"]):
            continue

        head_match = re.search(
            r'\b' + re.escape(triplet["head"].lower()) + r'\b', text)
        if head_match:
            head_index = head_match.start()
        else:
            head_index = text.find(triplet["head"].lower())

        tail_match = re.search(
            r'\b' + re.escape(triplet["tail"].lower()) + r'\b', text)
        if tail_match:
            tail_index = tail_match.start()
        else:
            tail_index = text.find(triplet["tail"].lower())

        if ((head_index == -1) or (tail_index == -1)):
            continue

        clean_triplets.append((triplet["head"], triplet["type"], triplet["tail"]))

    return clean_triplets

def extract_triplets(input_text):
    text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(input_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])[0]

    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail':object_.strip()})
    clean = clean_triplets(input_text, triplets)
    return clean