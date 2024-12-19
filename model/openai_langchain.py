from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
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
    
class RAGChain(BaseOpenAIChain):
    def __init__(self, prompt_messages, prompt_input_keys, api_key, model='gpt-4o'):
        super().__init__(prompt_messages, prompt_input_keys, api_key, model)
        self.session_storage = {}
    
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.session_storage:
            self.session_storage[session_id] = InMemoryChatMessageHistory()
            return self.session_storage[session_id]
        
        # memory 객체로 불러오기
        memory = ConversationBufferMemory(
            chat_memory=self.session_storage[session_id],
            return_messages=True,
        )
        assert len(memory.memory_variables) == 1    # 메모리에 저장된 변수가 하나인지 확인
        key = memory.memory_variables[0]
        messages = memory.load_memory_variables({})[key]
        self.session_storage[session_id] = InMemoryChatMessageHistory(messages=messages)
        return self.session_storage[session_id]

    def get_response(self, message_inputs, session_id):
        # message input dictionary 만들기
        message_query_dict = {}
        if len(message_inputs) == len(self.message_input_keys):
            for idx in range(len(message_inputs)):
                message_query_dict[self.message_input_keys[idx]] = message_inputs[idx]
        else:
            raise ValueError("message에 포함된 input keys의 개수와 전달된 input query의 개수가 다릅니다.")
        
        # self.chain에 MessageHistory 추가
        with_msg_history = RunnableWithMessageHistory(
            self.chain, # 실행할 runnable 객체(chain)
            self.get_session_history,   # chain에 덧붙일 history 가져오는 메서드
            input_messages_key="query", # 최신 입력 메세지로 처리되는 키
            history_messages_key="chat_history" # 이전 메세지를 추가할 키
        )
        config = {"configurable": {"session_id": session_id}}

        # history가 추가된 체인에 message_query_dict로 invoke
        response = with_msg_history.invoke(message_query_dict, config=config)

        return response.content

    def reset_storage(self):
        self.session_storage = {}

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
        
