from crawler.healthcare_crawlers import AMCMealTherapyCrawler, SSHDiabetesCrawler
from data_loader.data_saver import JsonSaver

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

    print(f">>> {crawler.__class__.__name__} 실행 중...")
    crawler_instance = crawler(**kwargs) if kwargs else crawler()
    articles = crawler_instance.run()
    json_saver = JsonSaver()
    json_saver.save(save_path, articles)
    print(f"저장 완료: {save_path}")

def crawl_and_update(openai_api_key:str, force_crawl:bool):
    """
    구현해둔 크롤러들로 res의 json문서들을 업데이트
    main에서 호출됨
    """
    tasks = [
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

    for task in tasks:
        crawl_and_save(
            task["crawler"],
            task["save_path"],
            force_crawl=force_crawl,
            **task["kwargs"]
        )

def main():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
   
    crawl_and_update(openai_api_key, force_crawl=False)

if __name__ == "__main__":
    main()