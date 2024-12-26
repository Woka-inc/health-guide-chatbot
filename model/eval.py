from openai import OpenAI
from langsmith import wrappers, Client
from pydantic import BaseModel, Field

class Grade(BaseModel):
    # 평가 결과 출력형식 정의
    score: bool = Field(description="Boolean that indicates whether the response is accurate relative to the reference answer")

class LLMJudge():
    def __init__(self, openai_api_key):
        llm = OpenAI(api_key=openai_api_key)
        self.eval_client = wrappers.wrap_openai(llm)
        self.client = Client()
        self.evaluators = {
            "accuracy": self.accuracy,
        }

    def accuracy(self, outputs: dict, reference_outputs: dict) -> bool:
        # LLM Judge (LangSmith "Run an evaluation guide")
        instructions = """Evaluate Student Answer against Ground Truth for conceptual similarity and classify true or false: 
        - False: No conceptual match and similarity
        - True: Most or full conceptual match and similarity
        - Key criteria: Concept should match, not exact wording.
        """
        response = self.eval_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
            { "role": "system", "content": instructions },
            { "role": "user", "content": f"""<<< Ground Truth answer >>>
            {reference_outputs["label"]};\n
            <<< Student's Answer >>>
            {outputs['response']}"""
        }],
            response_format=Grade
        )
        return response.choices[0].message.parsed.score
    
    def evaluate(self, target, dataset_name, evaluator_keywords:list, prefix):
        evaluators = [self.evaluators[keyword] for keyword in evaluator_keywords]
        results = self.client.evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=prefix,
            num_repetitions=1
        )
        return results