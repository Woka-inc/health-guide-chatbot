from langchain_openai import ChatOpenAI
from langsmith import Client
from typing_extensions import Annotated, TypedDict

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]

class LLMJudge():
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(api_key=openai_api_key, model='gpt-4o', temperature=0)
        self.client = Client()
        self.evaluators = {
            "correctness": self.correctness,
            "relevance": self.relevance,
            "groundedness": self.groundedness,
            "retrieval_relevance": self.retrieval_relevance,
        }
    
    def correctness(self, inputs:dict, outputs:dict, reference_outputs:dict) -> bool:
        grader_llm = self.llm.with_structured_output(CorrectnessGrade, method='json_schema', strict=True)
        instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""
        answers = f"""QUESTION: {inputs['text']}
GROUND TRUTH ANSWER: {reference_outputs['label']}
STUDENT ANSWER: {outputs['response']}"""
        grade = grader_llm.invoke([
            {"role": "system", "content":instructions}, 
            {"role": "user", "content": answers}
            ])
        return grade["correct"]
    
    def relevance(self, inputs:dict, outputs:dict) -> bool:
        grader_llm = self.llm.with_structured_output(RelevanceGrade, method="json_schema", strict=True)
        instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""
        answer = f"""QUESTION: {inputs['text']}
STUDENT ANSWER: {outputs['response']}"""
        grade = grader_llm.invoke([
            {"role": "system", "content": instructions}, 
            {"role": "user", "content": answer}
            ])
        return grade["relevant"]
    
    def groundedness(self, inputs:dict, outputs:dict) -> bool:
        grader_llm = self.llm.with_structured_output(GroundedGrade, method="json_schema", strict=True)
        instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""
        doc_string = ""
        doc_string.join(doc.page_content for doc in outputs["documents"])
        answer = f"""FACTS: {doc_string}
STUDENT ANSWER: {outputs['response']}"""
        grade = grader_llm.invoke([
            {"role": "system", "content": instructions}, 
            {"role": "user", "content": answer}
            ])
        return grade["grounded"]
    
    def retrieval_relevance(self, inputs:dict, outputs:dict) -> bool:
        grader_llm = self.llm.with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)
        instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.
"""
        doc_string = ""
        doc_string.join(doc.page_content for doc in outputs["documents"])
        answer = f"""FACTS: {doc_string}
QUESTION: {inputs['text']}"""
        grade = grader_llm.invoke([
            {"role": "system", "content": instructions}, 
            {"role": "user", "content": answer}
            ])
        return grade["relevant"]
    
    def evaluate(self, target, dataset_name, evaluator_keywords:list, prefix, metadata=None):
        evaluators = [self.evaluators[keyword] for keyword in evaluator_keywords]
        results = self.client.evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=prefix,
            num_repetitions=1,
            metadata=metadata
        )
        return results