from crawler.healthcare_crawlers import AMCMealTherapyCrawler, SSHDiabetesCrawler
from data_loader.data_saver import JsonSaver
from data_loader.structured_data_loader import JsonLoader
from model.retriever import FAISSBM25Retriever
from preprocessor.structured_data import json_to_langchain_doclist

from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

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

def main():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

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
            "kwargs": {"api_key": openai_api_key}
        }
    ]
    # crawl_and_update(crawl_tasks, force_crawl=False) 
    
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
    
    # RAG 3. Embed documents, set retriever
    retriever = create_retriever(FAISSBM25Retriever, documents, **{"openai_api_key": openai_api_key, "top_k": 1})

if __name__ == "__main__":
    main()