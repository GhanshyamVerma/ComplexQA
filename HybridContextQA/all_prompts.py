from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

############################################
# zero shot Prompt
############################################

yes_no_qa_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers can be yes or no. You have to write yes or no and nothing else. Do not write it depends or anything similar.\n" 
        "You HAVE to write only yes or no, even if you are uncertain.\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __yes/no__ \n"
    )

span_qa_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers must be a short span of the document. Do not write it depends or anything similar.\n" 
        "You have to extract the span from the document. Do not write anything else.\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __answer__ \n"
    )
    
yes_no_con_qa_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers can be yes or no. You have to write yes or no and nothing else. Do not write it depends or anything similar.\n" 
        "You HAVE to write only yes or no, even if you are uncertain.\n"
        "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) 'conditions'. You must have used this sentence(s) in your reasoning steps.\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __yes/no__ , Conditions: __condition__ \n"
    )

span_con_qa_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers must be a short span extracted from the Context information. Do not write it depends or anything similar.\n" 
        "You have to only extract the span of text from the Context information. Do not write anything else.\n"
        "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) 'conditions'. You must have used this sentence(s) in your reasoning steps.\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __answer__ , Conditions: __condition__ \n"
    )

############################################
# static few_shot Prompt   4 examples   
############################################

yes_no_qa_s4_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers can be yes or no. You have to write yes or no and nothing else. Do not write it depends or anything similar.\n" 
        "You HAVE to write only yes or no, even if you are uncertain.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        "Question: I served with the Royal Engineers during the war in Afghanistan from 2012-13, and now believe I have recently begun to experience traumatic flashbacks. . Can I claim for Post-Traumatic Stress, given the lateness of the onset of this?\n"
        "Answer: yes\n"
        "Question: My civil partner went missing just over 7 years ago. We are both British but he disappeared in Canada where we were living at the time; I came home to live in Wales 18 months ago. . Can I make a claim for a declaration of presumed death even though the missing person went missing abroad and was not living in England or Wales at the time?\n"
        "Answer: yes\n"
        "Question: My aunt has appointed me as her next of kin to oversee her finances after suffering from stroke. I want to apply as her tax relief appointee to act on her behalf. . Will I be automatically authorised to undertake this role and help her claim tax relief?\n"
        "Answer: no\n"
        "Question: My grandfather has appointed me as his financial affair attorney and i have been running his businesses efficiently . He has been diagnosed with a terminal illness and has few day to live. . Will a power of attorney still be valid after my grandfather dies?\n"
        "Answer: no\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __yes/no__ \n"
    )

span_qa_s4_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers must be a short span of the document. Do not write it depends or anything similar.\n" 
        "You have to extract the span from the document. Do not write anything else.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        "Question: I was badly injured whilst working on a construction site. I was evaluated as 80% disabled. I have no idea if or whether I will be able to return to work. . How much can I claim from this benefit?\n"
        "Answer: £146.32\n"
        "Question: My uncle has parkinsonism disease and appointed me to oversee his estate as his property deputy but during the application process i was overchaged by £100 by the office of the Guardian and i need a refund. . How long will it take for me to get the refund?\n"
        "Answer: up to 10 weeks to get a decision and a further 2 weeks to receive the refund\n"
        "Question: My child is 10 and has cerebal palsy. This means he is unable to walk. It is a struggle for me to get him around the house as his bedroom and the bathroom are upstairs. . Where can I get help to make adaptations to my home to help my child?\n"
        "Answer: your local council\n"
        "Question: I am a 16 year old living in Derby and was born male, I do not feel like I fit into any gender category. . What age can I apply for a certificate?\n"
        "Answer: 18 or over\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __answer__ \n"
    )
    
yes_no_con_qa_s4_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers can be yes or no. You have to write yes or no and nothing else. Do not write it depends or anything similar.\n" 
        "You HAVE to write only yes or no, even if you are uncertain.\n"
        "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) 'conditions'. You must have used this sentence(s) in your reasoning steps.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        "Question: I live in England, and my wife and I will soon be welcoming our first adoptive child into our home. As a full time employee who has been with the same company for 4 years, I have decided to take my full 52-week entitlement of Adoption Leave and have given the required notice to my employer. . Is it possible for my wife to claim Paternity Leave (even though she is a woman)?\n"
        "Answer: yes Conditions: <p>You need to qualify to get the fuel allowance through the National Concessionary Fuel Scheme (NCFS), and you can only get the cash allowance if you’re already getting fuel through the scheme.</p>\n"
        "Question: I am 24, live in England with my partner and 5 year old son, and am currently studying for a degree at University. I get undergraduate student finance, and my partner works full time. . Can I apply for a grant to cover the cost of childcare for my son whilst I attend my course?\n"
        "Answer: yes Conditions: <li>your childcare provider is on the Ofsted Early Years Register or General Childcare Register - check with your provider</li>\n"
        "<li>neither you or your partner are claiming Tax-Free Childcare, the childcare element of working Tax Credit or Universal Credit</li>\n"
        "<li>neither you or your partner receive help with childcare costs from the National Health Service (NHS)</li>\n"
        "<li>you’re not getting a Postgraduate Loan</li>\n"
        "<li>the children in your grant application are financially dependent on you</li>\n"
        "<li>if your child is cared for at home, the carer cannot be a relative and must be registered with an appropriate body - check with Student Finance England</li>\n"
        "Question: Me and my husband have decided to adopt a child from China. The child is arriving in England in 2 weeks. I work full time and I want to sort out time off work to care for it. . Am I eligible for statutory adoption pay and how long is it paid for?\n"
        "Answer: up to 39 weeks Conditions: <li>have been continuously employed by your employer for at least 26 weeks by the week you were matched with a child</li>\n"
        "<li>earn on average at least £120 a week (before tax)</li>\n"
        "<li>give the correct notice</li>\n"
        "<li>give proof of the adoption or surrogacy</li>\n"
        "<p>The requirements are the same if you’re adopting from overseas, except you must have been continuously employed by your employer for at least 26 weeks when you start getting adoption pay.</p>\n"
        "<p>You must also sign form SC6 if you’re adopting from overseas with a partner. This confirms you’re not taking paternity leave or pay.</p>\n"
        "Question: We are blessed with triplet baby boys. I am so happy about it. I am planning take paternity leave to take care of the boys . Can I take paid paternity leave and how many weeks of paternity leave can I take off in one go?\n"
        "Answer: 1 or 2 weeks Conditions: <li>be an employee</li>\n"
        "<li>give the correct notice</li>\n"
        "<li>have been continuously employed by your employer for at least 26 weeks up to any day in the ‘qualifying week’</li>\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __yes/no__ , Conditions: __condition__ \n"
    )

span_con_qa_s4_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers must be a short span extracted from the Context information. Do not write it depends or anything similar.\n" 
        "You have to only extract the span of text from the Context information. Do not write anything else.\n"
        "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) 'conditions'. You must have used this sentence(s) in your reasoning steps.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        "Question: I have written a will with a help of a solicitor before six years and I have lot of changes in my assets and I am planning to make some changes to my will . Can I make changes to my will once I have signed ? what is the process to amend a will ?\n"
        "Answer: make a new will Conditions: <p>For major changes you should make a new will.</p>\n"
        "Question: I work as a healthcare assistant  and i have tested positive for corona virus and i am in isolation with severe covid symptoms. I can’t go to work due to this effect and i want to seek income support from my employer. . Can i get help and how much will my employer pay me?\n"
        "Answer: £96.35 a week Conditions: <p>To qualify for Statutory Sick Pay (SSP) you must:</p>\n"
        "<li>be classed as an employee and have done some work for your employer</li>\n"
        "<li>earn an average of at least £120 per week</li>\n"
        "Question: My brother and his wife tragically died in a car accident last year. I agreed to become the guardians of their children. . How much guardian's allowance can I claim?\n"
        "Answer: £18 a week per child Conditions: <li>you qualify for Child Benefit</li>\n"
        "<li>one of the parents was born in the UK (or was living in the UK since the age of 16 for at least 52 weeks in any 2-year period)</li>\n"
        "Question: I am a UK taxpayer and I believe my income tax has been miscalculated by HMRC. They have sent me a letter telling me I owe them £5000. I want to appeal this decision. . When will I receive the tribunal decision?\n"
        "Answer: within 1 month Conditions: <li>in writing within 1 month, if you’ve had a ‘basic’ case - you’ll sometimes get a decision on the day</li>\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __answer__ , Conditions: __condition__ \n"
    )

############################################
# static few_shot Prompt   6 examples   
############################################

yes_no_qa_s6_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers can be yes or no. You have to write yes or no and nothing else. Do not write it depends or anything similar.\n" 
        "You HAVE to write only yes or no, even if you are uncertain.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        # "{few_shot_examples}\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __yes/no__ \n"
    )

span_qa_s6_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers must be a short span of the document. Do not write it depends or anything similar.\n" 
        "You have to extract the span from the document. Do not write anything else.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        # "{few_shot_examples}\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __answer__ \n"
    )
    
yes_no_con_qa_s6_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers can be yes or no. You have to write yes or no and nothing else. Do not write it depends or anything similar.\n" 
        "You HAVE to write only yes or no, even if you are uncertain.\n"
        "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) 'conditions'. You must have used this sentence(s) in your reasoning steps.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        # "{few_shot_examples}\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __yes/no__ , Conditions: __condition__ \n"
    )

span_con_qa_s6_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers must be a short span extracted from the Context information. Do not write it depends or anything similar.\n" 
        "You have to only extract the span of text from the Context information. Do not write anything else.\n"
        "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) 'conditions'. You must have used this sentence(s) in your reasoning steps.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        # "{few_shot_examples}\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __answer__ , Conditions: __condition__ \n"
    )

############################################
# Dynamic few_shot Prompt
############################################


yes_no_qa_d_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers can be yes or no. You have to write yes or no and nothing else. Do not write it depends or anything similar.\n" 
        "You HAVE to write only yes or no, even if you are uncertain.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        "{few_shot_examples}\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __yes/no__ \n"
    )

span_qa_d_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers must be a short span of the document. Do not write it depends or anything similar.\n" 
        "You have to extract the span from the document. Do not write anything else.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        "{few_shot_examples}\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __answer__ \n"
    )
    
yes_no_con_qa_d_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers can be yes or no. You have to write yes or no and nothing else. Do not write it depends or anything similar.\n" 
        "You HAVE to write only yes or no, even if you are uncertain.\n"
        "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) 'conditions'. You must have used this sentence(s) in your reasoning steps.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        "{few_shot_examples}\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __yes/no__ , Conditions: __condition__ \n"
    )

span_con_qa_d_prompt = PromptTemplate(
        "You are a helpful assistant that answers questions given Context information and not prior knowledge.\n"
        "Answers must be a short span extracted from the Context information. Do not write it depends or anything similar.\n" 
        "You have to only extract the span of text from the Context information. Do not write anything else.\n"
        "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) 'conditions'. You must have used this sentence(s) in your reasoning steps.\n"
        "Some examples are given below.\n"
        "---------------------\n"
        "{few_shot_examples}\n"
        "---------------------\n"
        "Context information: {context_str}\n"
        "---------------------\n"
        "Question : {query_str}\n"
        "Output Format: Answer: __answer__ , Conditions: __condition__ \n"
    )

############################################
# Knowledge-Graph Triplet Extraction Prompt
############################################

DEFAULT_KG_TRIPLET_EXTRACT_TMPL_1 = (
    "You are a Knowledge Graph creation expert. Some text is provided below. Given the text, extract all the relevant knowledge graph triplets in the form of (subject, predicate, object). Avoid stopwords. which will contain relevant information to answer questions . \n"
    "---------------------\n"
    "Example:\n"
    "Text: <p>Apply by the overseas route if your acquired gender has been legally accepted in an 'approved country or territory' and you have documents to prove it.</p>\n"
    "Triplets:\n(acquired gender, accepted in, approved country or territory)\n"
    "Text: <p>You must be 18 or over.</p>\n"
    "Triplets:\n(you, must be, 18 or over)\n"
    "Text: <tr>Overseas route | Form T453 | Leaflet T454</tr>\n"
    "Triplets:\n(apply, using, overseas route)\n"
    "Text: <p>If you’re applying using the overseas route, you must prove that your gender has been legally recognised in an 'approved country or territory'. Send original or certified copies of the following (if you have them):</p>\n"
    "Triplets:\n"
    "(gender, recognised in, approved country or territory)\n"
    "(send, copies of, documents)\n"
    "Text: <p>Apply by the standard route if all the following are true:</p>\n"
    "Triplets:\n(apply, by, standard route)\n"
    "Text: <li>you’re 18 or over</li>\n"
    "Triplets:\n(you, must be, 18 or over)\n"
    "Text: <p>You’ll get an 'interim certificate' if you or your spouse do not want to remain married, or if your spouse does not fill in a statutory declaration. You can use the interim certificate as grounds to end the marriage.</p>\n"
    "Triplets:\n"
    "(you, get, interim certificate)\n"
    "(spouse, fill in, statutory declaration)\n"
    "(use, interim certificate, end marriage)\n"
    "Text: <p>You and your spouse must fill in a statutory declaration saying you both agree to stay married.</p>\n"
    "Triplets:\n"
    "(you, spouse, fill in statutory declaration)\n"
    "(agree, stay, married)\n"
    "Text: <p>You can stay married if you apply for a Gender Recognition Certificate.</p>\n"
    "Triplets:\n(apply for, Gender Recognition Certificate, stay married)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)

DEFAULT_KG_TRIPLET_EXTRACT_TMPL_2 = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:\n"
    "Text: <p>Apply by the overseas route if your acquired gender has been legally accepted in an 'approved country or territory' and you have documents to prove it.</p>\n"
    "Triplets:\n(acquired gender, accepted in, approved country or territory)\n"
    "Text: <p>You must be 18 or over.</p>\n"
    "Triplets:\n(you, must be, 18 or over)\n"
    "Text: <tr>Overseas route | Form T453 | Leaflet T454</tr>\n"
    "Triplets:\n(apply, using, overseas route)\n"
    "Text: <p>If you’re applying using the overseas route, you must prove that your gender has been legally recognised in an 'approved country or territory'. Send original or certified copies of the following (if you have them):</p>\n"
    "Triplets:\n"
    "(gender, recognised in, approved country or territory)\n"
    "(send, copies of, documents)\n"
    "Text: <p>Apply by the standard route if all the following are true:</p>\n"
    "Triplets:\n(apply, by, standard route)\n"
    "Text: <li>you’re 18 or over</li>\n"
    "Triplets:\n(you, must be, 18 or over)\n"
    "Text: <p>You’ll get an 'interim certificate' if you or your spouse do not want to remain married, or if your spouse does not fill in a statutory declaration. You can use the interim certificate as grounds to end the marriage.</p>\n"
    "Triplets:\n"
    "(you, get, interim certificate)\n"
    "(spouse, fill in, statutory declaration)\n"
    "(use, interim certificate, end marriage)\n"
    "Text: <p>You and your spouse must fill in a statutory declaration saying you both agree to stay married.</p>\n"
    "Triplets:\n"
    "(you, spouse, fill in statutory declaration)\n"
    "(agree, stay, married)\n"
    "Text: <p>You can stay married if you apply for a Gender Recognition Certificate.</p>\n"
    "Triplets:\n(apply for, Gender Recognition Certificate, stay married)\n"
    "Text: {text}\n"
    "Triplets:\n"
)

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)