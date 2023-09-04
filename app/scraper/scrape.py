import time
import tqdm
import requests
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from undetected_chromedriver import By

MAX_RETRIES = 3

def is_website_okay(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def scraper(url):
    option = uc.ChromeOptions()
    option.add_argument('--disable-blink-features=AutomationControlled')
    option.add_argument('--disable-gpu')
    option.add_argument('--disable-extensions')
    option.add_argument('--profile-directory=Default')
    option.add_argument("--incognito")
    option.add_argument("--disable-plugins-discovery")
    option.add_argument("--start-maximized")
    option.add_argument("--disable-blink-features=AutomationControlled")
    option.add_argument("--disable-infobars")
    option.add_argument("--disable-blink-features")
    # option.add_argument("--headless")
    try:
        driver = uc.Chrome(options=option)
        driver.get(url)
        time.sleep(4)

        data = driver.page_source
        soup = BeautifulSoup(data, 'html.parser')
        print(soup)
        main_element = soup.find('main')
        if main_element:
            main_html = main_element.encode_contents().decode()
            return main_html
        
    except Exception as e:
        print(f"An error occurred while processing the website: {str(e)}")
        retries += 1
    finally:
        driver.quit()

if __name__ == '__main__':
    datas = pd.read_excel('./data/data.xlsx')
    datas['data'] = None
    for i,data in enumerate(datas['Field1_links_Link']):
        print(i)
        data = scraper(data)
        datas['data'][i] = data
        time.sleep(1)
        datas.to_excel('./data/data.xlsx', index=False)
