from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings

# classify_single_question 
def classify_single_question(references):
    """Function to classify a single question type into 'yes/no', 'yes/no_conditional', 'span', or 'span_conditional'."""
    
    if not references:
        return "unanswerable"
    
    # Check for yes/no answers
    if any(ans[0] in ["yes", "no"] for ans in references):
        if any(ans[1] for ans in references):
            return "yes/no_conditional"
        else:
            return "yes/no"
    
    # Check for span answers
    else:
        if any(ans[1] for ans in references):
            return "span_conditional"
        else:
            return "span"



class QuestionTypeClassifier():
    """
    A class for classifying the type of a given question.

    Attributes:
        llm (LanguageModel): The language model to use for classification.
        examples (list): A list of example questions and their corresponding types.
        task_description (str): A description of the task to be performed.
        qtype_chain (Chain): A chain of ICL and LLM models to classify the question type.
    """

    def __init__(self):
        """
        Initializes a new instance of the QuestionTypeClassifier class.

        Args:
            llm (LanguageModel): The language model to use for classification.
        """
        self.examples = [
            {
                "question": "Do I have a greater right to probate in respect of my late father's estate?",
                "type": "Yes/No",
            },
            {
                "question": "When can i start paying the inheritance tax?",
                "type": "Span",
            },
            {
                "question": "Is interest payable if I agree to pay it in installments?",
                "type": "Yes/No",
            },
            {
                "question": "Am I eligible for the Maternity allowance and if so how much will I get?",
                "type": "Span",
            },
            {
                "question": "Can I apply to change my father's will, or is this a matter for the deputy handling his financial affairs?",
                "type": "Yes/No",
            },
            {
                "question": "How long can I get help for after my main benefits stop?",
                "type": "Span",
            },
            {
                "question": "Do we have to pay the High Income Child Benefit Tax Charge?",
                "type": "Yes/No",
            },
            {
                "question": "What level of Blind Person' Allowance can I claim?",
                "type": "Span",
            },
        ]
        
        self.task_description = 'Your task is to classify whether a question should be answer with Yes/No or a full span. I\'ll give you some examples first. You should only answer "Yes/No" or "Span".'
        
        self.list_icl_chat_examples = [ ChatMessage(role="system", content=self.task_description)]
        
        self.prompt_template = "Question: {question}\nQuestion Type:"
        
        for x in self.examples:
            question = x["question"]
            qtype = x["type"]
            
            self.list_icl_chat_examples.append(ChatMessage(role="user", content=self.prompt_template.format(question=question)))
            self.list_icl_chat_examples.append(ChatMessage(role="assistant", content=qtype))


    def classify(self, question):
        """
        Classifies the type of the given question.

        Args:
            question (str): The question to classify.

        Returns:
            str: The type of the question.
        """
        chat_examples = self.list_icl_chat_examples.copy()
        
        chat_examples.append(ChatMessage(role="user", content="Question: "+question+"\nQuestion Type:"))

        return Settings.llm.chat(chat_examples).message.content