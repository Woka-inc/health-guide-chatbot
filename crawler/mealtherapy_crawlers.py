from .base_crawler import BaseCrawler
from selenium.webdriver.common.by import By
import re
from tqdm import tqdm

class AMCMealTherapyCrawler(BaseCrawler):
    def __init__(self):
        base_url = "https://www.amc.seoul.kr/asan/healthinfo/mealtherapy/mealTherapyList.do?pageIndex="
        super().__init__(base_url)
        self.ref_name = "서울아산병원"
    
    def get_article_links(self):
        # 기사 목록 페이지의 마지막 번호 구하기
        self.driver.get(self.base_url+"1")
        self.driver.implicitly_wait(2)
        last_page_btn_elem = self.driver.find_element(By.CSS_SELECTOR, 'a.lastPageBtn').get_attribute('onclick')
        match = re.search(r'fnList\((\d+)\)', last_page_btn_elem)
        if match:
            last_page_idx = int(match.group(1))
        else:
            raise ValueError("마지막 페이지 번호를 찾을 수 없습니다.")
        
        # 기사 목록 페이지를 돌면서 기사 url 수집하기
        print(">>> crawling step 1/2")
        for page_idx in tqdm(range(1, last_page_idx+1)):
            list_page_url = self.base_url + str(page_idx)
            self.driver.get(list_page_url)
            self.driver.implicitly_wait(2)
            article_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div.listCont > ul li > dl > dt > a')
            for elem in article_elements:
                self.article_links.append(elem.get_attribute('href'))
    
    def crawl_articles(self, article_url: str) -> dict: 
        self.driver.get(article_url)

        # 제목, 키워드 추출
        title = self.driver.find_element(By.CSS_SELECTOR, 'strong.contTitle').text
        keywords_list = self.driver.find_elements(By.CSS_SELECTOR, 'div.contBox > dl > dd')
        keywords = [elem.text for elem in keywords_list]

        # 본문 추출
        article_content = self.driver.find_element(By.CSS_SELECTOR, 'div.contDescription > dl.descDl').text

        # LangChain Documents 호환 JSON 형식
        return {
            "page_content": article_content,
            "metadata": {
                "title": title,
                "source_url": article_url,
                "author": self.ref_name,
                "tags": keywords
            }
        }