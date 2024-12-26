from crawler.healthcare_crawlers import AMCMealTherapyCrawler, SSHDiabetesCrawler
from data_loader.data_saver import JsonSaver
from data_loader.structured_data_loader import JsonLoader
from model.retriever import FAISSBM25Retriever
from model.openai_langchain import RAGChain
from preprocessor.structured_data import json_to_langchain_doclist
from database.table_manager import UserTableManager, ChatLogTableManager
from model.eval import LLMJudge

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import Client

import streamlit as st
from dotenv import load_dotenv
import os

# st.session_state 목록
# - OPENAI_API_KEY: 모델에 사용할 OpenAI API Key. 환경변수로부터 로드하거나 사용자에게 입력 받음
# - retriever: user_query를 입력받아 관련 문서를 검색. set_retriever에서 생성
# - rag_chain: set_chain에서 prompt template 정의 후 생성한 chain.
# - messages: {'role':, 'content':}로 구성된 리스트. 사용자 쿼리와 모델 응답을 담고 있음.
# - user: ['id':, 'username':] 현재 로그인된 사용자의 계정정보
# - session_id: 현재 사용자의 대화 session_id

def crawl_and_save(crawler, save_path, force_crawl=False, **kwargs):
    """
    크롤러를 실행하고 JSON 파일로 저장
    crawl_and_update에서 호출됨
    """
    if os.path.exists(save_path) and not force_crawl:
        print(f">>> 이미 존재하는 파일이 있습니다: {save_path} -> 새로 크롤링하지 않고 기존 데이터를 사용합니다.")
        return

    crawler_instance = crawler(**kwargs) if kwargs else crawler()
    articles = crawler_instance.run()
    json_saver = JsonSaver()
    json_saver.save(save_path, articles)
    print(f"저장 완료: {save_path}")

def crawl_and_update(crawl_tasks, force_crawl:bool):
    """
    실행할 크롤러를 명시, res의 json문서들을 업데이트
    main에서 호출됨
    """

    for task in crawl_tasks:
        crawl_and_save(
            task["crawler"],
            task["save_path"],
            force_crawl=force_crawl,
            **task["kwargs"]
        )

def split_documents(documents, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len)
    split_result = text_splitter.split_documents(documents)
    return split_result

def create_retriever(retriever, documents, **kwargs):
    # retriever 클래스와 임베딩할 documents를 넘겨받아 retriever 생성
    retriever_instance = retriever(documents, **kwargs) if kwargs else retriever(documents)
    return retriever_instance

@st.dialog("OpenAI API Key 요청")
def ask_openai_api_key():
    st.write("챗봇을 사용하기 위해 OpenAI의 API Key가 필요합니다.")
    st.write("\'확인\'버튼을 누른 후 잠시만 기다려주세요.")
    key = st.text_input("your api key here")
    if st.button("확인") and key:
        st.session_state['OPENAI_API_KEY']
        print(">>> OPENAI_API_KEY: 사용자로부터 입력 받음")
        st.rerun()

@st.dialog("Log in")
def user_login(db_user, db_chatlog):
    st.markdown("<span style='font-weight: bold;'>username</span>", unsafe_allow_html=True)
    username = st.text_input(label='username', label_visibility="collapsed")
    st.markdown("<span style='font-weight: bold;'>email</span>", unsafe_allow_html=True)
    email = st.text_input(label='email', label_visibility="collapsed")
    btn = st.button("login")
    if username and email and btn:
        user_info = db_user.check_user(username, email)
        if user_info:
            st.session_state['user'] = {'id': user_info[0], 'email': user_info[2]}
            st.session_state['session_id'] = str(db_chatlog.get_new_session_id(st.session_state['user']['id']))
            db_user.update_last_login(st.session_state['user']['id'])
            st.rerun()
        else:
            st.markdown("<span style='color:red;'>username과 email을 다시 확인해주세요.</span>", unsafe_allow_html=True)

@st.dialog("Join")
def user_join(db_user, db_chatlog):
    st.markdown("<span style='font-weight: bold;'>username</span>", unsafe_allow_html=True)
    username = st.text_input(label='username', placeholder="10자 이내, 필수", label_visibility="collapsed")
    st.markdown("<span style='font-weight: bold;'>email</span>", unsafe_allow_html=True)
    email = st.text_input(label='email', placeholder="필수", label_visibility="collapsed")
    btn = st.button("join and login")
    if username and email and btn:
        error = db_user.create_user(username, email)
        if error:
            if error == "Duplicate entry":
                st.markdown("<span style='color:red;'>이미 존재하는 이메일입니다.</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red;'>{error}</span>", unsafe_allow_html=True)
        else:
            user_info = db_user.check_user(username, email)
            st.session_state['user'] = {'id': user_info[0], 'email': user_info[2]}
            st.session_state['session_id'] = str(db_chatlog.get_new_session_id(st.session_state['user']['id']))
            db_user.update_last_login(st.session_state['user']['id'])
            st.rerun()

@st.dialog("대화 저장하기")
def archive_chat(db_chatlog):
    if len(st.session_state.messages) == 0:
        st.write("저장할 대화가 없습니다.")
        if st.button("확인"):
            st.rerun()
    else:
        st.write("대화 내용을 저장할 제목을 입력해주세요.")
        chat_title = st.text_input(label='chat title')
        btn = st.button("저장하기")
        session_id = st.session_state['session_id']
        user_id = st.session_state['user']['id']
        if chat_title and btn:
            # 채팅목록 추가 in chat_title
            db_chatlog.create_chat_title(session_id, user_id, chat_title)
            # 채팅 내용 저장 in chat_log
            for message in st.session_state.messages:
                sendor = message['role']
                content = message['content']
                db_chatlog.insert_chat_log(session_id, user_id, sendor, content)
            # 현재 대화 초기화
            st.session_state.messages = []
            st.session_state['rag_chain'].reset_storage()
            st.session_state['session_id'] = str(db_chatlog.get_new_session_id(st.session_state['user']['id']))
            st.rerun()

@st.cache_resource
def set_retriever():
    """RAG 0~3: 문서로드~검색기 생성"""
    # RAG 0. Crawl Data
    crawl_tasks = [
        {
            "crawler": AMCMealTherapyCrawler,
            "save_path": './res/amc-mealtherapy.json',
            "kwargs": {}
        },
        {
            "crawler": SSHDiabetesCrawler,
            "save_path": './res/ssh-diabetes.json',
            "kwargs": {"api_key": st.session_state['OPENAI_API_KEY']}
        }
    ]
    crawl_and_update(crawl_tasks, force_crawl=False) 
    
    # RAG 1. Load Data
    json_doc_paths = [crawler['save_path'] for crawler in crawl_tasks]
    json_loader = JsonLoader()
    documents = []
    for path in json_doc_paths:
        json_doc = json_loader.load(path)
        documents += json_to_langchain_doclist(json_doc)
    
    # RAG 2. Split Documents
    splitted_documents = split_documents(documents, 
                                    chunk_size=300, 
                                    overlap=100)
    
    # RAG 3. Indexing: Embed documents, set retriever
    return create_retriever(FAISSBM25Retriever, splitted_documents, **{"openai_api_key": st.session_state['OPENAI_API_KEY'], "top_k": 2})
    
def set_chain():
    """RAG 3.5: chain 생성"""
    print(">>> RAGChain 생성 in st.session_state")
    # RAG 3.5. setup chain
    rag_prompt_template = """당신은 사용자의 건강 상태와 상황을 이해하고, 공신력 있는 근거 자료를 바탕으로 깊이 있고 실질적인 건강 정보를 제공하는 전문가 AI 챗봇입니다. 
사용자의 질문에 대해 다음 기준을 따라 답변하세요:

1. **근거 자료 기반 응답**:  
제공되는 답변의 정보는 반드시 아래의 <<< 관련 근거자료 >>>에 근거해야 합니다.
아래의 <<< 관련 근거자료>>>로 제공된 정보를 벗어나 추측하지 말고, 모든 답변에는 실제 출처를 source_url과 함께 명확히 언급하세요.  
- '출처: 서울아산병원'

2. **맞춤형 초기 대화**:  
사용자 상황을 이해하기 위해 답변을 완료한 뒤에도 친근하고 구체적인 질문을 던지세요. 예시:  
- '현재 가장 걱정되는 건강 문제는 무엇인가요?'  
- '어떤 목표를 가지고 계신가요? 혈당 조절, 체중 관리, 아니면 전반적인 건강 개선인가요?'

3. **개인화된 결과 제공**:  
사용자의 정보(나이, 성별, 특정 질환)를 바탕으로 맞춤형 솔루션을 제안합니다. 예시:  
- '○○님(20대 여성)을 위한 맞춤형 혈당 관리 팁입니다.'  
- '2형 당뇨 환자에게 적합한 하루 식사 및 운동 가이드를 제공할게요.'

4. **전문적이고 공감하는 어조**:  
전문적이지만 친절하고 따뜻한 어조로 사용자에게 공감하며 안내하세요.

5. **구조화된 정보 제공(선택적)**:
사용자에게 제공하는 정보가 많다면, 아래 답변 예시를 참고해 구조화된 내용으로 전달하세요.
---
<<< 입력 예시 >>>
'나는 23살 여성이야. 며칠 전 제2형 당뇨병을 진단받았어. 혈당 수치를 정상으로 유지하는 식사 방법을 알려줘.'

<<< 답변 예시 >>> 
'안녕하세요. 제2형 당뇨병 진단을 받으셨군요. 혈당 조절은 정말 중요하면서도 신경 쓸 게 많아서 걱정이 크실 것 같아요. 
하지만 작은 습관부터 차근차근 실천하면 충분히 관리할 수 있으니 너무 부담 갖지 않으셔도 돼요. 제가 도움을 드릴 수 있도록 정확하고 실질적인 정보를 알려드릴게요! 

1. **식사 조절의 필요성**:  
당뇨병은 인슐린의 절대적 또는 상대적인 부족으로 인해 고혈당 및 대사 장애를 초래하는 질환입니다. 따라서, 혈당을 정상에 가깝게 유지하고 합병증을 최소화하기 위해 식사 조절이 필요합니다.  
- 출처: 서울아산병원 (link)

2. **추천 식단 및 조리 방법**:
- **간식**: 정규 식사 사이에 제철 과일과 저지방 우유를 섭취하는 것이 좋습니다.
- **조리 방법**: 지방 섭취를 줄이기 위해 튀기거나 부치기 대신 굽기, 찜, 삶는 방법을 주로 선택하세요. 맛을 내기 위해 적당량의 식물성 기름(참기름, 들기름 등)은 사용해도 좋습니다.
- 출처: 서울아산병원 (link)

개인의 건강 상태에 따라 다르게 적용될 수 있으니, 담당 의사나 영양사와 상의하는 것도 좋은 방법입니다. 건강 관리에 도움이 되시길 바랍니다!'
---
<<< 과거 사용자 채팅 내용 >>>
{chat_history}

<<< 사용자 입력 >>>
{query}

<<< 관련 근거자료 >>>
{context}
"""
    prompt_message = [
        ("system", rag_prompt_template),
        ("human", "<<< 사용자 입력 >>>\n{query}")
    ]
    chain = RAGChain(prompt_message, st.session_state['OPENAI_API_KEY'])
    return chain

def get_chain_response(user_query):
    """RAG 4~5: 검색 & 응답생성"""
    # RAG 4. Retrieval
    retrieved_documents = st.session_state['retriever'].search_docs(user_query)
    # RAG 5. Generate
    response = st.session_state['rag_chain'].get_response(message_inputs={'query': user_query, 'context': retrieved_documents}, session_id=st.session_state['session_id'])
    return response

def get_metric_ratio(metric, df):
    return df[metric].sum() / len(df) * 100

def app():
    def write_app_title():
        st.markdown("<h1 style='text-align: center;'>Health Guide ChatBot</h1>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>당신의 건강을 위한 신뢰할 수 있는 맞춤형 정보를 제공해드립니다.</h5>", unsafe_allow_html=True)
    
    print(">>> main() 실행")

    # retriever, chain 초기화 ----------------------------------
    if 'retriever' not in st.session_state:
        st.session_state['retriever'] = set_retriever()
    if 'rag_chain' not in st.session_state:
        st.session_state['rag_chain'] = set_chain()
    
    # database table manager 초기화
    db_user = UserTableManager()
    db_chatlog = ChatLogTableManager()

    # 채팅 키 초기화 ----------------------------------
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Streamlit UI - 사이드바 ----------------------------------
    with st.sidebar:
        if 'user' in st.session_state:
            st.write(f"user_id: {st.session_state.user['id']} / email: {st.session_state.user['email']}")

            if st.button("로그아웃"):
                if 'user' in st.session_state:
                    del st.session_state['user']
                st.session_state.messages = []
                st.session_state['session_id'] = None
                st.session_state['rag_chain'].reset_storage()
                st.rerun()

            if st.button("대화 내용 저장하고 새로 시작하기"):
                archive_chat(db_chatlog)
            
            if st.button("대화 새로 시작하기"):
                st.session_state.messages = []
                if 'show_chat_session' in st.session_state:
                    del st.session_state['show_chat_session']
                st.session_state['session_id'] = str(db_chatlog.get_new_session_id(st.session_state['user']['id']))
                st.session_state['rag_chain'].reset_storage()
                print(f">>> 현 session_id: {st.session_state.session_id}")

            # 과거 대화 내역 표시
            st.markdown("<h4>저장된 대화 내역</h4>", unsafe_allow_html=True)
            titles = db_chatlog.get_chat_titles(st.session_state['user']['id'])
            session_ids, chat_titles = [], []
            for i in range(len(titles)):
                session_ids.append(titles[i][0])
                chat_titles.append(titles[i][2])
                if st.button(chat_titles[i], use_container_width=True):
                    st.session_state['show_chat_session'] = session_ids[i]

    # Streamlit UI - 메인 화면 ----------------------------------
    if 'user' in st.session_state:
        # 로그인 정보가 있을 때 채팅 화면
        if 'show_chat_session' in st.session_state:
            # 저장된 대화 내역을 클릭한 상태
            selected_idx = session_ids.index(st.session_state['show_chat_session'])
            selected_chat_title = chat_titles[selected_idx]
            chat_log = db_chatlog.get_session_chat(st.session_state['user']['id'],
                                                st.session_state['show_chat_session'])
            st.markdown(f"<h4>{selected_chat_title}</h4>", unsafe_allow_html=True)
            chat_container = st.container()
            with chat_container:
                for chat in chat_log:
                    sender = chat[3]
                    message = chat[4]
                    st.chat_message(sender).write(message)
        else: 
            # session_state에 'show_chat_session'키가 없는 상태 = 과거 대화 내용 조회가 아닌 상태
            write_app_title()
            # 채팅내역 표시
            for message in st.session_state.messages:
                with st.chat_message(message['role']):
                    st.markdown(message['content'])
            
            # user input에 반응
            if user_query := st.chat_input("궁금한 점을 입력하세요."):
                with st.chat_message('user'):
                    st.markdown(user_query)
                # session_state.messages에 추가
                st.session_state.messages.append({"role": "user", "content": user_query})

                response = get_chain_response(user_query)
                with st.chat_message('ai'):
                    st.markdown(response)
                st.session_state.messages.append({"role": "ai", "content": response})
            
    else:
        # 로그인 정보가 없을 때 화면
        if 'show_chat_session' in st.session_state:
            del st.session_state['show_chat_session']
        write_app_title()
        st.markdown("<p style='text-align: center;'>이용을 위해 로그인해주세요!</p>", unsafe_allow_html=True)
        btn_cols = st.columns(2)
        login_btn = btn_cols[0].button("Log in", type="primary", use_container_width=True)
        join_btn = btn_cols[1].button("Join", use_container_width=True)
        if login_btn:
            user_login(db_user, db_chatlog)
        if join_btn:
            user_join(db_user, db_chatlog)

def eval():
    # 평가할 대상 준비 (streamlit 쓰지 않아야 함)
    retriever = set_retriever()
    rag_chain = set_chain()
    def target(inputs: dict) -> dict:
        # 평가할 응답 생성
        retrieved_documents = retriever.search_docs(inputs['text'])
        # RAG 5. Generate
        response = rag_chain.get_response(message_inputs={'query': inputs['text'], 'context': retrieved_documents})
        return {'response': response, 'documents': retrieved_documents}
    
    client = Client()
    # 데이터셋 불러오기
    dataset_generator = client.list_datasets()
    datasets = [dataset.name for dataset in dataset_generator]
    # 평가 메트릭
    metrics = ['correctness', 'relevance', 'groundedness', 'retrieval_relevance']
    metric_instructions = """<b>🍊 Correctness</b><br/>모델의 응답이 정답과 유사한가<br/>
    <b>🍊 Relevance</b><br/>모델의 응답이 질문과 관련있는가<br/>
    <b>🍊 Groundedness</b><br/>모델의 응답이 검색 문서에 기반했는가<br/>
    <b>🍊 Retrieval Relevance</b><br/>검색 문서가 질문과 관련있는가<br/>
    """
    st.sidebar.markdown(metric_instructions, unsafe_allow_html=True)

    eval_result, selected_i = None, None
    eval_status = st.status("세부 설정 후 run evaluation을 클릭하세요.", state='complete')

    # 평가 전 세부 선택
    with st.container(border=True):
        selected_database = st.selectbox("사용할 dataset을 선택하세요", datasets)
        selected_metrics = st.multiselect("사용할 평가 metric을 선택하세요.", metrics)
        if st.button("run evaluation"):
            eval_status.update(label="평가 진행 중", state='running')
            judge = LLMJudge(st.session_state['OPENAI_API_KEY'])
            st.session_state['eval_result'] = judge.evaluate(target, 
                                    selected_database, 
                                    ['correctness', 'relevance', 'groundedness', 'retrieval_relevance'], 
                                    prefix="RAG evaluation test",
                                    metadata={"version": "initial RAG Chain"})
            eval_status.update(label="평가 완료!", state='complete')

    eval_result = st.session_state['eval_result']
    # 평가 완료되면 결과 표시
    if eval_result:
        df = eval_result.to_pandas()
        columns = st.columns(len(selected_metrics))
        for i in range(len(selected_metrics)):
            columns[i].metric(selected_metrics[i], get_metric_ratio('feedback.' + selected_metrics[i], df))

        st.divider()
        questions = df.loc[:, 'inputs.text'].to_list()
        selected_question = st.selectbox("세부 결과를 확인할 질문을 선택하세요.", questions)
        selected_i = questions.index(selected_question)
        st.text("")
    if selected_i:
        columns = st.columns(len(selected_metrics))
        for i in range(len(selected_metrics)):
            value = df.loc[selected_i, 'feedback.' + selected_metrics[i]]
            columns[i].metric(selected_metrics[i], value)
        st.text("")
        question = df.loc[selected_i, 'inputs.text']
        response = df.loc[selected_i, 'outputs.response']
        documents = df.loc[selected_i, 'outputs.documents']
        truth = df.loc[selected_i, 'reference.label']
        st.subheader(question)
        st.markdown("<h4>🍊 truth</h4>", unsafe_allow_html=True)
        st.markdown(truth, unsafe_allow_html=True)
        st.markdown("<h4>🍊 response</h4>", unsafe_allow_html=True)
        st.markdown(response, unsafe_allow_html=True)
        st.markdown("<h4>🍊 retrieved documents</h4>", unsafe_allow_html=True)
        for document in documents:
            st.json(document.metadata)
            st.text(document.page_content)


def main():
    st.set_page_config(page_title="Health Guide ChatBot | Woka")

    if st.session_state.get('user', {}).get('email') == 'woka@admin':
        # admin 버전
        if st.sidebar.button("session_state 삭제"):
            # 개발버전에서만 쓰는 버튼
            st.session_state.clear()
            st.rerun()

        page_names_to_funcs = {
            "chatbot": app,
            "evaluation": eval
        }

        selection = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
        page_names_to_funcs[selection]()
    else:
        # 일반 유저 버전
        app()

if __name__ == "__main__":
    # 프로그램 시작 시 .env 등을 통해 전달된 OPENAI_API_KEY가 st.session_state에 있는지 확인
    if 'OPENAI_API_KEY' in st.session_state:
        main()
    else:
        load_dotenv()
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        if OPENAI_API_KEY:
            # 환경변수에 저장된 키가 있다면 불러오기
            st.session_state['OPENAI_API_KEY'] = OPENAI_API_KEY
            print(">>> OPENAI_API_KEY: 환경변수에서 로드")
            main()
        else:
            ask_openai_api_key()