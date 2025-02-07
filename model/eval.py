from langchain_openai import ChatOpenAI
from langsmith import Client
from typing_extensions import Annotated, TypedDict
from model.openai_langchain import BaseOpenAIChain
from preprocessor.similarity import EmbeddingsSimilarity

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    example_result: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    example_result: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    example_result: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    example_result: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]

class LLMJudge():
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(api_key=openai_api_key, model='gpt-4o', temperature=0)
        self.client = Client()
        self.evaluators = {
            "correctness": self.correctness,
            "relevance": self.relevance,
            "groundedness": self.groundedness,
            "retrieval_relevance": self.retrieval_relevance,
        }
        self.eval_results = {
            'correctness':[], 
            'relevance': [],
            'groundedness': [], 
            'retrieval_relevance': []
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
Your explanation should be in Korean.

Avoid simply stating the correct answer at the outset."""
        answers = f"""QUESTION: {inputs['question']}\n
GROUND TRUTH ANSWER: {reference_outputs['ground_truth']}\n
STUDENT ANSWER: {outputs['response']}"""
        grade = grader_llm.invoke([
            {"role": "system", "content":instructions}, 
            {"role": "user", "content": answers}
            ])
        grade['given_to_llm_judge'] = answers
        self.eval_results['correctness'].append(grade)
        return grade["example_result"]
    
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
Your explanation should be in Korean.

Avoid simply stating the correct answer at the outset."""
        answer = f"""QUESTION: {inputs['question']}\n
STUDENT ANSWER: {outputs['response']}"""
        grade = grader_llm.invoke([
            {"role": "system", "content": instructions}, 
            {"role": "user", "content": answer}
            ])
        grade['given_to_llm_judge'] = answer
        self.eval_results['relevance'].append(grade)
        return grade["example_result"]
    
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
Your explanation should be in Korean.

Avoid simply stating the correct answer at the outset."""
#         docs_string = ""
#         for doc in outputs['documents']:
#             doc_string = f"<< metadata of fact >> {doc.metadata}\n<< fact >> {doc.page_content}"
#             docs_string += doc_string
        docs_string = "\n\n".join(doc.page_content for doc in outputs['documents'])
        answer = f"""<<< FACTS >>>\n{docs_string}\n\n\n
<<< STUDENT ANSWER>>>\n{outputs['response']}"""
        grade = grader_llm.invoke([
            {"role": "system", "content": instructions}, 
            {"role": "user", "content": answer}
            ])
        grade['given_to_llm_judge'] = answer
        grade['given_outputs'] = outputs
        self.eval_results['groundedness'].append(grade)
        return grade["example_result"]
    
    def retrieval_relevance(self, inputs:dict, outputs:dict) -> bool:
        """
        outputs["documents"]: 검색된 문서
        inputs['question']: 사용자 질문(query)
        """

        # 질환 코사인 유사도 계산기
        similarity_calculator = EmbeddingsSimilarity(openai_api_key=self.openai_api_key)

        # 쿼리/문서 제목에서 질환명 추출
        disease_extractor_msg = [
            ("system", """You are an AI assistant specialized in extracting disease names from text. 
Your task is to identify the disease mentioned in either a document title or a user query and return the name in Korean.

### Instructions:
1. You will receive either:
   - A **document title**, which represents the main topic of a medical document.
   - A **user query**, which is a question asked by a user about a medical condition.

2. **For a document title**:
   - Extract the primary disease that the document focuses on.
   - If the title explicitly contains a disease name, return only that name in Korean.
   - If the title does not explicitly mention a disease but implies one, extract the most relevant disease name.
   - If the title is too general or does not relate to a specific disease, return "Not specified"

3. **For a user query**:
   - Identify the disease the user is asking about.
   - If the query does not explicitly mention a disease but implies one, extract the most relevant disease name.
   - If the query is too general or does not relate to a specific disease, return "Not specified".

4. **Output format**:
   - If a disease is identified, return only the disease name in Korean as a string (e.g., "고혈압").
   - If no specific disease is mentioned or implied, return "Not specified".

### **Examples:**

#### **Document Title Examples**
**Input:** Document Title: "식사요법 - 고혈압식"  
**Output:** 고혈압  

**Input:** Document Title: "식사요법 - 당뇨병식"  
**Output:** 당뇨병  
             
**Input:** Document Title: "스트레스 관리법"  
**Output:** Not specified

#### **User Query Examples**
**Input:** User Query: "고혈압 환자가 피해야 할 음식은?"  
**Output:** 고혈압  

**Input:** User Query: "당뇨병 예방을 위한 식단 추천해주세요"  
**Output:** 당뇨병  

**Input:** User Query: "건강에 좋은 음식이 무엇인가요?"  
**Output:** Not specified

---

Now, extract the disease name based on the given input and return it in Korean."""),
        ("user", "{text_type}: \"{text}\"")
        ]
        disease_extractor_llm = BaseOpenAIChain(messages=disease_extractor_msg, api_key=self.openai_api_key, model='gpt-4o-mini')
        for doc in outputs['documents']:
            doc.metadata['relevant_disease'] = disease_extractor_llm.get_response(
                {"text_type": "Document Title", "text": doc.metadata['title']}
                )
        query_disease = disease_extractor_llm.get_response({"text_type": "User Query",
                                                            "text": inputs['question']})

        # 문서와 쿼리 모두 질환이 특정되면 코사인 유사도 계산, 아니라면 Unmeasurable
        for doc in outputs['documents']:
            if (doc.metadata['relevant_disease'] != "Not specified") and (query_disease != "Not specified"):
                doc.metadata['COSINE_SIMILARITY'] = similarity_calculator.get_cosine_similarity(doc.metadata['relevant_disease'], query_disease)
            else:
                doc.metadata['COSINE_SIMILARITY'] = "Unmeasurable"
        

        grader_llm = self.llm.with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)
        instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student.
Additionally, you will receive metadata about the FACTS, which includes the document title, as well as a COSINE_SIMILARITY score measuring the semantic similarity between the disease mentioned in the QUESTION and the disease in the document title.

Here is the grade criteria to follow:
(1) **If a COSINE_SIMILARITY score is provided:**
   - If **COSINE_SIMILARITY ≥ 0.4**, the FACTS are considered to be from a potentially relevant document.
     - However, you must still verify whether the FACTS actually contain the information requested in the QUESTION.
     - If the FACTS provide the requested information, assign a relevance value of **True**.
     - If the FACTS do not provide the requested information despite the similarity score, assign a relevance value of **False**.
   - If **COSINE_SIMILARITY < 0.4**, the FACTS are considered to be from an unrelated document, and the relevance value must be **False**.

(2) **If COSINE_SIMILARITY is "Unmeasurable" (when the disease name in the QUESTION or document title is "Not specified"):**
   - Carefully analyze the QUESTION to determine what information is being asked.
   - Evaluate whether the FACTS provide the requested information.
   - If the FACTS contain the necessary information, assign a relevance value of **True**.
   - If the FACTS lack the requested information, assign a relevance value of **False**.

Relevance:
- True
  - The FACTS are derived from a document that is **semantically relevant** to the disease or condition mentioned in the QUESTION (COSINE_SIMILARITY ≥ 0.4).  
  - The FACTS provide the requested information explicitly or contain details that sufficiently answer the QUESTION.  
  - If COSINE_SIMILARITY is "Unmeasurable," the FACTS still contain the necessary information asked in the QUESTION.
- False
  - The FACTS are derived from a document that is **not semantically relevant** to the disease or condition mentioned in the QUESTION (COSINE_SIMILARITY < 0.4).  
  - The FACTS do not contain the requested information, even if the document itself is relevant (COSINE_SIMILARITY ≥ 0.4 but lacks necessary details).  
  - If COSINE_SIMILARITY is "Unmeasurable," the FACTS do not provide sufficient information to answer the QUESTION.  

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Your explanation should be in Korean.

Avoid simply stating the correct answer at the outset.
"""
        docs_string = ""
        for doc in outputs['documents']:
            doc_string = f"<< metadata of FACT >>\n{doc.metadata}\n<< FACT >>\n{doc.page_content}\n"
            docs_string += doc_string
        # docs_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"""<<< FACTS >>>\n {docs_string}\n\n\n
<<< QUESTION >>> {inputs['question']}"""
        grade = grader_llm.invoke([
            {"role": "system", "content": instructions}, 
            {"role": "user", "content": answer}
            ])
        grade['documents_in_outputs'] = []
        for doc in outputs['documents']:
            grade['documents_in_outputs'].append(doc)
        grade['given_to_llm_judge'] = answer
        grade['given_inputs'] = inputs
        grade['given_outputs'] = outputs
        self.eval_results['retrieval_relevance'].append(grade)
        return grade['example_result']
    
    def evaluate(self, target, dataset_name, evaluator_keywords:list, repetition, prefix, description, metadata=None):
        evaluators = [self.evaluators[keyword] for keyword in evaluator_keywords]
        results = self.client.evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=prefix,
            num_repetitions=repetition,
            metadata=metadata,
            description=description
        )
        return results