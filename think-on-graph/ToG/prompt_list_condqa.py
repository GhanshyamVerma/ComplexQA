extract_relation_prompt_condqa = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1). Do not return any other text which is not relevant to the given question.
Q: Before my father died last year  June 2021 ,he appointed as his family property executor. I have inherited all our family properties. When can i start paying the inheritance tax ?
Topic Entity: Inheritance Tax
Relations: APPLY_FOR; APPLY_TO; BE_PAID_TO; CAN'T_OR_DOESN'T_PAY; CAN'T_PAY; DEPENDS_ON; DOESN'T_PAY; IS_PAID_BY; MAY_NEED_TO_PAY; PAID; PAID_BY; PAID_IF; PAID_ON; PAID_OUT_OF; PAY; PAY; PAY_ON; RESPONSIBLE_FOR
A: 1. PAY (Score: 0.6) - The question is about when to start paying the inheritance tax.
2. RESPONSIBLE_FOR (Score: 0.3) - The executor is responsible for handling the family properties, including paying the inheritance tax.
3. DEPENDS_ON (Score: 0.1) - The timing of paying the inheritance tax may depend on certain conditions or procedures.

Q: """

score_entity_candidates_prompt = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: film.producer.film
Entites: The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones
Score: 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
The movie that matches the given criteria is "So Undercover" with Miley Cyrus and produced by Tobin Armbrust. Therefore, the score for "So Undercover" would be 1, and the scores for all other entities would be 0.

Q: {}
Relation: {}
Entites: """

prompt_evaluate="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: {Yes}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: {No}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: {Yes}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: {No}. Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.

"""

condaqa_yn_ans_with_kg_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity)\
you are asked to answer the question with these triplets and your knowledge.
1. The answer must be either "yes" or "no".
2. You must write "yes" or "no" and nothing else. Do not write "it depends" or anything similar.
3. You HAVE to write only "yes" or "no", even if you are uncertain.

Example:
Question: My partner has just had our second child and she wants to go back to work immediately, I will not be taking time off either.\
Would we still receive any sort of benefit if we are both working full time?
Knowledge Triplets: mother, CANNOT_RETURN_TO_WORK, before end of compulsory 2 weeks of maternity leave
parents, MAY_GET, Shared Parental Leave
Output: Answer: no

Question: My father, who was a widower and the owner of several large properties in Wales, died recently and apparently intestate. \
My paternal uncle is applying for probate, but I believe that I have a stronger claim. Do I have a greater right to probate in respect of my late father's estate?
Knowledge Triplets: Probate, IS_NECESSARY_FOR, intestate estates
probate, DETERMINES, legal right to manage deceased's estate
probate, SHOULD_BE_GRANTED_TO, closest kin
closest kin, INCLUDES, children
Output: Answer: yes

Question: My aunt has appointed me as her next of kin to oversee her finances after suffering from stroke. I want to apply as her tax relief appointee to act on her behalf. \
Will I be automatically authorised to undertake this role and help her claim tax relief?
Knowledge Triplets: aunt, APPOINTED, me as next of kin
aunt, SUFFERING_FROM, stroke
permission, NEEDED_TO, deal with someone else's tax credits
Output: Answer: no

Question: "I served with the Royal Engineers during the war in Afghanistan from 2012-13, and now believe I have recently begun to experience traumatic flashbacks. \
Can I claim for Post-Traumatic Stress, given the lateness of the onset of this?
Knowledge Triplets: I, SERVERD_DURING, war in Afghanistan
I, HAVE, traumatic flashbacks
I, CAN_CLAIM, Post-Traumatic Stress
compensation, CAN_BE_CLAIMED_FOR, any injury or illness from service
to qualify for compensation, MUST_BE, current or former member of armed forces
Output: Answer: yes

"""

condaqa_yn_ans_cond_with_kg_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity),\
you are asked to answer the question with these triplets and your knowledge.
1. The answer must be either "yes" or "no". Do not write "it depends" or anything similar. Write only "yes" or "no", even if you are uncertain.
2. If some answers require the assumption of certain conditions to be true, write the full sentence(s) that you think are required to be true after the answer. \
We call these sentence(s) "conditions". You must use these sentence(s) in your reasoning steps.
3. Ensure the output is in the format: Answer: [yes/no]. Conditions: [condition(s)].

Example:
Question: I was born and raised in Australia. I have changed my gender and got a certificate in Australia. I have moved to UK three years back.\
I would like to know whether I am eligible to apply for Gender Recognition Certificate in UK ?
Knowledge Triplets: Gender, IS, Changed
Gender, HAS_BEEN, legally recognised
Gender Recognition Certificate, ISSUED_BY, UK government
Gender Recognition Certificate, ELIGIBILITY_CONDITION, age 18 or over
Output: Answer: yes. Conditions: You must be 18 or over.

Question: I am currently getting Reduced Earnings Allowance and really not an a regular work and will be soon turn 66. Will I still be eligible for Reduced Earnings Allowance?
Knowledge Triplets: Reduced Earnings Allowance, ELIGIBILITY, state pension age
Reduced Earnings Allowance, REPLACED_BY, Retirement Allowance
Retirement Allowance, APPLIES_IF, reach State Pension age)
Retirement Allowance, APPLIES_IF, not in regular employment)
Reduced Earnings Allowance, STOPS_WHEN, Retirement Allowance starts
Output: Answer: no. Conditions: you reach State Pension age

Question: My husband and I are both retired and on a low income, so we get pension credit. However, we had some savings so we also get savings credit.\
Are we eligible for the warm home discount even though we receive savings credit?
Knowledge Triplets: we, GET, pension credit
Warm Home Discount Scheme, APPLIES_TO, core group
core group, INCLUDES, those on Guarantee Credit element of Pension Credit
energy supplier, IS_PART_OF, scheme
Output: Answer: yes. Conditions: your energy supplier is part of the scheme. your name (or your partner\u2019s) is on the bill

Question: My sister is a widow and on benefits. She got a volunteer job offer. She will be reimbursed out of pocket expenses for travels and meals.\
Will this affect her benefits?
Knowledge Triplets: sister, IS, a widow
sister, GOT, volunteer job offer
sister, WILL_BE_REIMBURSED_FOR, out of pocket expenses for travels and meals
volunteering, will not affect, her benefits
Output: Answer: no. Conditions: you continue to meet the conditions of the benefit you get.

"""

condaqa_span_ans_with_kg_prompt = """You are a helpful assistant that answers questions given knowledge graph \
triplets information and not prior knowledge.\n
1. The answer must be a short span of text.
2. The span must be relevant to the question.
3. Do not write anything else. Do not generate triples or functions.

Example:
Question: I am a 16 year old living in Derby and was born male, I do not feel like I fit into any gender category.\
What age can I apply for a certificate?
Knowledge Triplets: Apply for gender identity certificate, ELIGIBILITY, 18 or over
Output: Answer: 18 or over

Question: I'm 28, and have worked full-time for my current employer for just over 3 years. My wife is expecting our first child in a few months,\
and I intend to claim paid Paternity Leave when the baby is born. How much notice am I required to give my employer with regards to the starting date of my leave period?
Knowledge Triplets: Paternity Leave, NOTICE_PERIOD, 15 weeks before due date
Output: Answer: at least 15 weeks before the baby is due

Question: My child is 10 and has cerebal palsy. This means he is unable to walk. It is a struggle for me to get him around the house\
as his bedroom and the bathroom are upstairs. Where can I get help to make adaptations to my home to help my child?
Knowledge Triplets: Cerebral palsy, IS_A, disability
local council, OFFERS_ASSISTANCE_FOR, disabilities
Local council, IS_RESPONSIBLE_FOR, Home Adaptations
Output: Answer: your local council

Question: My uncle has parkinsonism disease and appointed me to oversee his estate as his property deputy but during the application process\
i was overchaged by \u00a3100 by the office of the Guardian and i need a refund. How long will it take for me to get the refund?
Knowledge Triplets: I, NEED, a refund
overcharge, BY, Office of the Guardian
Office of the Guardian refund, INVOLVES, Office of the Guardian
Office of the Guardian refund process, TAKES_UP_TO, 10 weeks for a decision
Office of the Guardian refund, TAKES_FURTHER, 2 weeks to receive
Output: Answer: up to 10 weeks to get a decision and a further 2 weeks to receive the refund
"""

condaqa_span_ans_cond_with_kg_prompt = """You are a helpful assistant that answers questions given knowledge graph \
triplets information and not prior knowledge.\n
1. The answer must be a short span of text relevant to the question.
2. Do not write anything else.
3. Some answers may require the assumption of some sentences from the text to be true. If you think that is the case,\
you must write the full sentence(s) that you think are required to be true after the answer. We call these sentences "conditions".\
You must use these sentences in your reasoning steps.
4. Ensure the output is in the format: Answer: [short span of text]. Conditions: [condition(s)].

Example:
Question: We are blessed with triplet baby boys. I am so happy about it. I am planning take paternity leave to take care of the boys.\
Can I take paid paternity leave and how many weeks of paternity leave can I take off in one go?
Knowledge Triplets: paternity leave, DURATION, 1 or 2 weeks
paternity leave, AVAILABLE_TO, employees
Output: Answer: 1 or 2 weeks. Conditions: be an employee, give the correct notice, have been continuously employed by your employer for\
at least 26 weeks up to any day in the \u2018qualifying week\u2019.

Question: I am not a British citizen but live in UK for the last 10 years and have Indefinite Leave to Remain permit. I am on a permanent employment and planning to adopt a child in Malaysia. What the proof I need to submit to get leave and get paid ?
Knowledge Triplets: proof of adoption, REQUIRED_FOR, Statutory Adoption Leave
proof of adoption, REQUIRED_FOR, Statutory Adoption Pay
Output: Answer: proof of adoption. Conditions: You must give your employer proof of adoption to qualify for Statutory Adoption Pay. Proof is not needed for Statutory Adoption Leave unless they request it.

Question: I have written a will with a help of a solicitor before six years and I have lot of changes in my assets and I am planning to make some changes to my will. Can I make changes to my will once I have signed ? what is the process to amend a will ?
Knowledge Triplets: codicil, OFFICIAL_ALTERATION_FOR, will
process to amend a will, INVOLVES, making a new will for major changes
process to amend a will, INVOLVES, making a codicil for minor changes
Output: Answer: make a new will. Conditions: For major changes you should make a new will., Answer: making an official alteration called a codicil. Conditions: You cannot amend your will after it\u2019s been signed and witnessed.\
The only way you can change a will is by making an official alteration called a codicil.

Question: My grandfather died 3 months ago and he did not leave a will. His house is currently empty and bank accounts are untouched. How much of his estate can I claim?
Knowledge Triplets: grandfather, died, 3 months ago
grandfather, DID_NOT_LEAVE, a will
if no will, FIRST_CLAIM, spouse or civil partner
if no will,ENTITLED_TO_A_SHARE, anyone descended from grandparent
Output: Answer: a share. Conditions: if there\u2019s no spouse or child, anyone descended from a grandparent of the person is entitled to a share in the estate
"""

condaqa_yn_ans_without_kg_prompt = """Answers can be "yes" or "no". You have to write "yes" or "no" and nothing else. Do not write "it depends or anything similar". You HAVE to write only "yes" or "no", even if you are uncertain.
Question: My father, who was a widower and the owner of several large properties in Wales, died recently and apparently intestate. My paternal uncle is applying for probate, but I believe that I have a stronger claim. Do I have a greater right to probate in respect of my late father's estate?
Output: Answer: "yes"

Question: My partner has just had our second child and she wants to go back to work immediately, I will not be taking time off either. Would we still receive any sort of benefit if we are both working full time?
Output: Answer: "no"

Question: My aunt has appointed me as her next of kin to oversee her finances after suffering from stroke. I want to apply as her tax relief appointee to act on her behalf. \
Will I be automatically authorised to undertake this role and help her claim tax relief?
Output: Answer: "no"

Question: "I served with the Royal Engineers during the war in Afghanistan from 2012-13, and now believe I have recently begun to experience traumatic flashbacks. \
Can I claim for Post-Traumatic Stress, given the lateness of the onset of this?
Output: Answer: "yes"

"""

condaqa_yn_ans_cond_without_kg_prompt = """Answers can be "yes" or "no". You have to write "yes" or "no" and nothing else. Do not write "it depends or anything similar". You HAVE to write only "yes" or "no", even if you are uncertain.
Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) "conditions". You must have used this sentence(s) in your reasoning steps.
Question: I was born and raised in Australia. I have changed my gender and got a certificate in Australia. I have moved to UK three years back. I would like to know whether I am eligible to apply for Gender Recognition Certificate in UK ?
Output: Answer: "yes". Conditions: "You must be 18 or over."

Question: I am currently getting Reduced Earnings Allowance and really not an a regular work and will be soon turn 66. Will I still be eligible for Reduced Earnings Allowance?
Output: Answer: "no". Conditions: "you reach State Pension age"

Question: My husband and I are both retired and on a low income, so we get pension credit. However, we had some savings so we also get savings credit.\
Are we eligible for the warm home discount even though we receive savings credit?
Output: Answer: "yes". Conditions: "your energy supplier is part of the scheme". "your name (or your partner\u2019s) is on the bill"

Question: My sister is a widow and on benefits. She got a volunteer job offer. She will be reimbursed out of pocket expenses for travels and meals.\
Will this affect her benefits?
Output: Answer: "no". Conditions: "you continue to meet the conditions of the benefit you get."

"""

condaqa_span_ans_without_kg_prompt = """Answers must be a short span of the text. The span must be relevant to the question. Do not write anything else.
Question: I'm 28, and have worked full-time for my current employer for just over 3 years. My wife is expecting our first child in a few months, and I intend to claim paid Paternity Leave when the baby is born. How much notice am I required to give my employer with regards to the starting date of my leave period?
Output: Answer: "at least 15 weeks before the baby is due"

Question: My child is 10 and has cerebal palsy. This means he is unable to walk. It is a struggle for me to get him around the house as his bedroom and the bathroom are upstairs. Where can I get help to make adaptations to my home to help my child?
Output: Answer: "your local council"

Question: I am a 16 year old living in Derby and was born male, I do not feel like I fit into any gender category.\
What age can I apply for a certificate?
Output: Answer: "18 or over"

Question: My uncle has parkinsonism disease and appointed me to oversee his estate as his property deputy but during the application process\
i was overchaged by \u00a3100 by the office of the Guardian and i need a refund. How long will it take for me to get the refund?
Output: Answer: "up to 10 weeks to get a decision and a further 2 weeks to receive the refund"

"""

condaqa_span_ans_cond_without_kg_prompt = """Answers must be a short span of the text. The span must be relevant to the question. Do not write anything else.
Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) "conditions". You must have used this sentence(s) in your reasoning steps.
Question: I am not a British citizen but live in UK for the last 10 years and have Indefinite Leave to Remain permit. I am on a permanent employment and planning to adopt a child in Malaysia. What the proof I need to submit to get leave and get paid ?
Output: Answer: "proof of adoption", Conditions: "You must give your employer proof of adoption to qualify for Statutory Adoption Pay. Proof is not needed for Statutory Adoption Leave unless they request it."

Question: We are blessed with triplet baby boys. I am so happy about it. I am planning take paternity leave to take care of the boys. Can I take paid paternity leave and how many weeks of paternity leave can I take off in one go?
Output: Answer: "1 or 2 weeks", Conditions: "be an employee", "give the correct notice", "have been continuously employed by your employer for at least 26 weeks up to any day in the \u2018qualifying week\u2019"

Question: I have written a will with a help of a solicitor before six years and I have lot of changes in my assets and I am planning to make some changes to my will. Can I make changes to my will once I have signed ? what is the process to amend a will ?
Output: Answer:" make a new will". Conditions: "For major changes you should make a new will.", Answer: "making an official alteration called a codicil." Conditions: "You cannot amend your will after it\u2019s been signed and witnessed.\
The only way you can change a will is by making an official alteration called a codicil."

Question: My grandfather died 3 months ago and he did not leave a will. His house is currently empty and bank accounts are untouched. How much of his estate can I claim?
Output: Answer: "a share". Conditions: "if there\u2019s no spouse or child, anyone descended from a grandparent of the person is entitled to a share in the estate"

"""
