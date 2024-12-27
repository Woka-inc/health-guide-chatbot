from crawler.healthcare_crawlers import AMCMealTherapyCrawler, SSHDiabetesCrawler
from data_loader.data_saver import JsonSaver
from data_loader.structured_data_loader import JsonLoader
from model.retriever import FAISSBM25Retriever
from model.openai_langchain import RAGChain
from preprocessor.structured_data import json_to_langchain_doclist
from database.table_manager import UserTableManager, ChatLogTableManager
from model.eval import LLMJudge
from model.prompts import rag_prompt_template

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import Client

import streamlit as st
from dotenv import load_dotenv
import inspect
import os

# st.session_state 목록
# - OPENAI_API_KEY: 모델에 사용할 OpenAI API Key. 환경변수로부터 로드하거나 사용자에게 입력 받음
# - retriever: user_query를 입력받아 관련 문서를 검색. set_retriever에서 생성
# - rag_chain: set_chain에서 prompt template 정의 후 생성한 chain.
# - messages: {'role':, 'content':}로 구성된 리스트. 사용자 쿼리와 모델 응답을 담고 있음.
# - user: ['id':, 'username':] 현재 로그인된 사용자의 계정정보
# - session_id: 현재 사용자의 대화 session_id

serviced_RAGset = {
    "version": "initial RAG Chain", 
    "splitter": RecursiveCharacterTextSplitter,
    "chunk_size": 300,
    "overlap": 100,
    "retriever": FAISSBM25Retriever,
    "top_k": 2,
    "prompt": "1.0",
    "llm": "gpt-4o"
}

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

def split_documents(documents, splitter, chunk_size, overlap):
    text_splitter = splitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len)
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
            print("user_info", user_info)
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
def set_retriever(RAGset):
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
                                        RAGset['splitter'],
                                        chunk_size=RAGset['chunk_size'], 
                                        overlap=RAGset['overlap']
                                        )
    
    # RAG 3. Indexing: Embed documents, set retriever
    return create_retriever(RAGset['retriever'], splitted_documents, **{"openai_api_key": st.session_state['OPENAI_API_KEY'], "top_k": RAGset['top_k']})
    
def set_chain(RAGset):
    """RAG 3.5: chain 생성"""
    print(">>> RAGChain 생성 in st.session_state")
    # RAG 3.5. setup chain
    rag_prompt = rag_prompt_template[RAGset['prompt']]
    prompt_message = [
        ("system", rag_prompt),
        ("human", "<<< 사용자 입력 >>>\n{query}")
    ]
    chain = RAGChain(prompt_message, st.session_state['OPENAI_API_KEY'], model=RAGset['llm'])
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
        st.session_state['retriever'] = set_retriever(serviced_RAGset)
    if 'rag_chain' not in st.session_state:
        st.session_state['rag_chain'] = set_chain(serviced_RAGset)
    
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
    if 'eval_result' not in st.session_state:
        st.session_state['eval_result'] = None

    # 평가할 대상 준비 (streamlit 쓰지 않아야 함)
    retriever, rag_chain = None, None
    def target(inputs: dict) -> dict:
        # 평가할 응답 생성
        retrieved_documents = retriever.search_docs(inputs['text'])
        # RAG 5. Generate
        response = rag_chain.get_response(message_inputs={'query': inputs['text'], 'context': retrieved_documents})
        return {'response': response, 'documents': retrieved_documents}
    
    # app에서 서비스 중인 RAGset 문자열로
    serviced_RAGset_str = {
        key: (serviced_RAGset[key].__name__ if inspect.isclass(serviced_RAGset[key]) else serviced_RAGset[key])
        for key in serviced_RAGset
        }
    test_RAGset = serviced_RAGset

    # 모델 선택 옵션
    llm_options = ['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini']

    # 수정할 수 없는 RAGset 설정 (metadata)
    fixed_keys = ['splitter', 'retriever', 'prompt']
    fixed_metadata = "\n".join(f"- **{key}**: {serviced_RAGset_str[key]}" for key in fixed_keys)

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
        eval_setting_cols = st.columns(2)
        with eval_setting_cols[0]:
            prefix = st.text_input("prefix", "RAG evaluation test")
            test_RAGset['version'] = st.text_input("RAG version", "initial RAG Chain")
            st.write(fixed_metadata, unsafe_allow_html=True)
            test_RAGset['chunk_size'] = st.slider("chunk_size", 100, 1000, 300, 50)
            test_RAGset['top_k'] = int(st.slider("top_k (EnsembleRetriever 사용 중이므로 2배수로 가능)", 2, 10, 2, 2) / 2) # 앙상블이라 2개씩
            test_RAGset['llm'] = st.pills("model", llm_options)
        with eval_setting_cols[1]:
            selected_database = st.selectbox("사용할 dataset을 선택하세요", datasets)
            selected_metrics = st.multiselect("사용할 평가 metric을 선택하세요.", metrics)
            repetition = st.slider("반복 횟수", 1, 5, 1, 1)

        if st.button("run evaluation", type="primary", use_container_width=True):
            if selected_metrics:
                eval_status.update(label="평가 진행 중", state='running')
                retriever = set_retriever(test_RAGset)
                rag_chain = set_chain(test_RAGset)
                judge = LLMJudge(st.session_state['OPENAI_API_KEY'])
                st.session_state['eval_result'] = judge.evaluate(target, 
                                        selected_database, 
                                        selected_metrics, 
                                        repetition=repetition,
                                        prefix=prefix,
                                        metadata=test_RAGset)
                eval_status.update(label="평가 완료!", state='complete')
            else:
                st.toast("평가 metric을 선택하세요!")

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