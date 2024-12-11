from crawler.base_crawler import BaseCrawler
from model.openai_langchain import ImageDescriptionChain

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import os, re, requests
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
    

class SSHDiabetesCrawler(BaseCrawler):
    def __init__(self, api_key):
        base_url = "http://www.samsunghospital.com/dept/main/index.do?DP_CODE=DM&MENU_ID=008051"
        super().__init__(base_url)
        self.ref_name = "삼성서울병원"
        system_prompt = """당신은 이미지에서 표를 감지하고 HTML 형식으로 추출하는 전문가입니다. 다음 조건을 따라 분석하고 출력을 제공합니다:

1. **표가 포함된 그림**:  
   이미지 내에 표가 포함되어 있으면, 해당 표를 **HTML 형식**으로 변환해 출력하세요.  
   출력 예시:  
   ```html
   <table>
     <tr><th>헤더 1</th><th>헤더 2</th></tr>
     <tr><td>데이터 1</td><td>데이터 2</td></tr>
   </table>
   ```

2. 표가 아닌 그림:
이미지에 표가 포함되어 있지 않으면 **"표가 없는 이미지"**라고만 출력하세요.

3. 추가 지시사항:
  - 표의 구조를 최대한 정확히 유지하세요.
	- 불명확하거나 손상된 부분이 있으면 그대로 표시하고 추정하지 마세요."""

        self.table_from_image_chain = ImageDescriptionChain(system_prompt, api_key)
    
    def get_article_links(self):
        self.driver.get(self.base_url)
        self.driver.implicitly_wait(2)

        # 왼쪽 사이드바 굵은 글씨 탭의 11개 element 가져오기
        li_elems = self.driver.find_elements(By.CSS_SELECTOR, 'div#leftMenu > ul > li > a')
        # 굵은 글씨 탭의 숨은 하위 element 9개 가져오기
        li_elems.extend(self.driver.find_elements(By.CSS_SELECTOR, 'div#leftMenu > ul > li > ul.deptsection_3depth_menu > li > a'))
        # 각 element의 링크 가져오기
        print(">>> crawling step 1/2")
        for li_elem in tqdm(li_elems):
            self.article_links.append(li_elem.get_attribute('href'))
        
    def crawl_articles(self, article_url):
        self.driver.get(article_url)

        # 제목 추출, 키워드 = 제목
        title = self.driver.find_element(By.CSS_SELECTOR, 'h2#pageTitle').text
        keywords = [title]

        # 본문 element 전체 추출
        try:
            content_div = self.driver.find_element(By.CSS_SELECTOR, "div.newDept.subContents")
        except NoSuchElementException:
            print(f">>> {article_url}에서 div.newDept.subContents를 찾지 못함")
            return None
        elements = content_div.find_elements(By.XPATH, "./*")

        # 본문 내 element별로 처리 후 포함
        processed_content = []
        for element in elements:
            tag_name = element.tag_name
            if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                # heading 태그 처리
                heading_text = element.text.strip()
                processed_content.append(f"<{tag_name}>{heading_text}</{tag_name}>")
            elif tag_name in ['span', 'p']:
                # span, p 태그 내부 텍스트 처리
                text = element.text.strip()
                if text:
                    processed_content.append(text)
                # span, p 태그 내부 img 태그 처리: 다운로드, 표 추출
                inner_elems = element.find_elements(By.XPATH, "./*")
                if inner_elems:
                    for inner_elem in inner_elems:
                        if inner_elem.tag_name == 'img':
                            img_src = inner_elem.get_attribute("src")
                            if img_src:
                                local_image_path = self.download_image(img_src)
                                if local_image_path:
                                    img_description = self.table_from_image_chain.get_response('', local_image_path)
                                    processed_content.append(img_description)
        
        # 처리된 본문 내용을 하나의 문자열로 병합
        combined_content = "\n".join(processed_content)

        # LangChain Documents 호환 JSON 형식
        return {
            "page_content": combined_content,
            "metadata": {
                "title": title,
                "source_url": article_url,
                "author": self.ref_name,
                "tags": keywords
            }
        }
