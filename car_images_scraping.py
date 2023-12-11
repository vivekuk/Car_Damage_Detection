from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests

"""Create a new instance of the Chrome driver"""
driver = webdriver.Chrome()

"""Open the website"""
url = "https://www.istockphoto.com/search/2/image-film?phrase=car%20damage&page=1"
driver.get(url)

"""Maximize the browser window (make it full screen)"""
driver.maximize_window()

image_links = list()

# for getting the image links of all the web pages by iterating over the pagination

for page_no in range(2, 101):
    time.sleep(5)
    url = '='.join(url.split("=",2)[:2])+"="+str(page_no)
    for image_element in driver.find_elements("xpath", "//*[@data-testid='gallery-items-container']//picture/img"):
        image_links.append(image_element.get_attribute('src'))
    driver.get(url)

# for downloading the images using the image link

for image_link in image_links:
    response = requests.get(image_link)
    with open("images/"+image_link.split('/')[6].split('?')[0], "wb") as f:
        f.write(response.content)




