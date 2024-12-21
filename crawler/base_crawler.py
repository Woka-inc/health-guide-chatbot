from selenium import webdriver
from abc import ABC, abstractmethod
from tqdm import tqdm
import os, requests

class BaseCrawler(ABC):
    def __init__(self, base_url: str, headless: bool = True):
        """
        base_url: 기본 주소
        headless: 크롤링 중 브라우저를 띄우지 않는 옵션
        """
        self.base_url = base_url
        self.options = webdriver.ChromeOptions()
        if headless:
            self.options.add_argument("headless")
        self.driver = webdriver.Chrome(options=self.options)
        self.article_links = []
    
    @abstractmethod
    def get_article_links(self):
        """
        개별 기사 링크를 self.article_links.append()로 추가해야 함
        print(">>> crawling step 1/2")
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def crawl_articles(self, article_url: str) -> dict: 
        raise NotImplementedError("Subclasses should implement this method.")
    
    def download_image(self, image_url):
        # image_url에서 이미지를 다운로드한 후 로컬 경로 반환
        image_save_dir = './res/crawled_images'
        os.makedirs(image_save_dir, exist_ok=True)
        image_name = os.path.basename(image_url)
        save_path = os.path.join(image_save_dir, image_name)

        if os.path.exists(save_path):
            return save_path
        else:
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    print(f"이미지 다운로드 성공: {save_path}")
                    return save_path
            except Exception as e:
                print(f"이미지 다운로드 실패: {e}")
            return None

    def quit_driver(self):
        self.driver.quit()
    
    def run(self):
        # 기사 링크 모으기
        self.get_article_links()
        # 각 기사 크롤링
        data = []
        print(">>> crawling step 2/2")
        for article_url in tqdm(self.article_links):
            article_data = self.crawl_articles(article_url)
            if article_data:
                data.append(article_data)

        self.quit_driver()
        return data
