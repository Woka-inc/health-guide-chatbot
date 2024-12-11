from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from preprocessor.image import get_resized_img, encode_bytesio_to_base64

class BaseOpenAIChain():
    def __init__(self, messages, message_input_keys, api_key, model='gpt-4o'):
        """
        messages: [("system", "..."), ("user", "...")] 형식의 message 리스트
        """
        prompt = ChatPromptTemplate.from_messages(messages)
        model = ChatOpenAI(model=model, api_key=api_key)
        self.chain = prompt | model
        self.message_input_keys = message_input_keys
    
    def get_response(self, message_inputs):
        """
        message_inputs: messages에 포함된 input key와 해당하는 내용 쌍의 dict {"user_query": query}
        """
        # message input dictionary 만들기
        message_query_dict = {}
        if len(message_inputs) == len(self.message_input_keys):
            for idx in range(len(message_inputs)):
                message_query_dict[self.message_input_keys[idx]] = message_inputs[idx]
        else:
            print("message에 포함된 input keys의 개수와 전달된 input query의 개수가 다릅니다.") 

        response = self.chain.invoke(message_query_dict)
        return response.content
    
class ImageDescriptionChain(BaseOpenAIChain):
    def __init__(self, system_prompt, api_key, model='gpt-4o'):
        messages = [
            ("system", system_prompt),
            ("user", "{user_query}"),
            ("user", [{
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
            }])
        ]
        super().__init__(messages, message_input_keys=["user_query", "image_data"], api_key=api_key, model=model)

    def get_response(self, user_query, image_file):
        resized_img = get_resized_img(image_file)
        encoded_img = encode_bytesio_to_base64(resized_img)
        message_query_dict = {
            "user_query": user_query,
            "image_data": encoded_img
        }
        response = self.chain.invoke(message_query_dict)
        return response.content
        
