from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from preprocessor.image import get_resized_img, encode_bytesio_to_base64

class BaseOpenAIChain():
    def __init__(self, messages, api_key, model='gpt-4o'):
        """
        messages: [("system", "..."), ("user", "...")] 형식의 message 리스트
        """
        prompt = ChatPromptTemplate.from_messages(messages)
        model = ChatOpenAI(model=model, api_key=api_key)
        self.chain = prompt | model
    
    def get_response(self, message_inputs):
        """
        message_inputs: messages에 포함된 input key와 해당하는 내용 쌍의 dict {"user_query": query}
        """
        response = self.chain.invoke(message_inputs)
        return response.content
    
class RAGChain(BaseOpenAIChain):
    def __init__(self, prompt_messages, api_key, model='gpt-4o'):
        super().__init__(prompt_messages, api_key)
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

    def get_response(self, message_inputs, session_id=None):
        # session_id를 지정하지 않으면 memory 없이 구현 (평가용))
        if session_id:
            with_msg_history = RunnableWithMessageHistory(
                self.chain,  # 실행할 runnable 객체
                self.get_session_history,
                input_messages_key="query",  # 최신 입력 메세지로 처리되는 키
                history_messages_key="chat_history" # 이전 메세지를 추가할 키
            )
            response = with_msg_history.invoke(
                message_inputs,
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        else:
            message_inputs['chat_history'] = None
            response = self.chain.invoke(message_inputs)
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
        super().__init__(messages, api_key=api_key, model=model)

    def get_response(self, user_query, image_file):
        resized_img = get_resized_img(image_file)
        encoded_img = encode_bytesio_to_base64(resized_img)
        message_query_dict = {
            "user_query": user_query,
            "image_data": encoded_img
        }
        response = self.chain.invoke(message_query_dict)
        return response.content
        
