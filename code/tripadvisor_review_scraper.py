##############################################################################
##############################################################################
#                                                                            #
#                          TripAdvisor Review Scraper                        #
#                               Master Thesis                                #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################


import os
import time
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pyarrow.parquet as pq

os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/code')
##############################################################################
# Import own Functions
from functions_tripadvisor_scraping import headers_for_bs, hotel_info_function, add_ranking
##############################################################################

# Directory
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')




##############################################################################
# Link Scraper
##############################################################################
# first we scrape urls to hotel pages on tripadvisor
# from where we will extract the reviews

# Install WebDriver from the location (path) where Chrome webdriver is saved
path_webdriver = ''
driver = webdriver.Chrome(path_webdriver)

# get url
url = "https://www.tripadvisor.co.uk/Search?q=united%20kingdom&searchSessionId=73C73D11F5E64C2E409C286BBD00AAC61655044053862ssid&searchNearby=false&geo=186338&sid=4FE063B7636749AE989E2A45EAFA52751655044068670&blockRedirect=true&ssrc=h&rf=26&o=900"
driver.get(url)

# link scraper loop
hotel_link_list = []
for i in range(0,30):
    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME,"ui_column.is-12.content-column.result-card")))
    time.sleep(3)
    link_containers = driver.find_elements(By.CLASS_NAME, "ui_column.is-12.content-column.result-card")
    
    for k in link_containers:
        
        try: 
            x = k.find_element(By.TAG_NAME, "a")
            hotel_link_list.append(x.get_attribute('href'))
        except:
            print("no tag here")
    

        
    driver.execute_script("window.scrollTo(0, 9000);")
    time.sleep(2)
    
    driver.find_element(By.XPATH, '//*[@id="BODY_BLOCK_JQUERY_REFLOW"]/div[2]/div/div[2]/div/div/div/div/div[1]/div/div[1]/div/div[3]/div/div[2]/div/div/div/a[2]').click()




# Convert do DF
tripadvisor_link_list = pd.DataFrame(hotel_link_list)
tripadvisor_link_list = tripadvisor_link_list.reset_index()

# Save to CSV
tripadvisor_link_list.to_csv("tripadvisor_link_list.csv",index=False)


# Convert back to list for further use
hotel_link_list = tripadvisor_link_list.iloc[:,0].tolist()


# Add more links (url's)
# with below loop we can create urls of subsequent pages
# so we don't have to click the next button
# (beautifull soup cannot "click" anyways.)

hotel_links_more = []

for i in hotel_link_list:

    main_link = i + "#REVIEWS"
    substr = "-Reviews-"
    second = "or10-"
    third = "or20-"
    fourth = "or30-"
    fifth = "or40-"
    sixth = "or50-"
    seventh = "or60-"
    eighth = "or70-"
    ninth = "or80-"
    tenth = "or90-"

    hotel_links_more.append(main_link)
    hotel_links_more.append(main_link.replace(substr, substr + second))
    hotel_links_more.append(main_link.replace(substr, substr + third))
    hotel_links_more.append(main_link.replace(substr, substr + fourth))
    hotel_links_more.append(main_link.replace(substr, substr + fifth))
    hotel_links_more.append(main_link.replace(substr, substr + sixth))
    hotel_links_more.append(main_link.replace(substr, substr + seventh))
    hotel_links_more.append(main_link.replace(substr, substr + eighth))
    hotel_links_more.append(main_link.replace(substr, substr + ninth))
    hotel_links_more.append(main_link.replace(substr, substr + tenth))

hotel_links_more_df = pd.DataFrame(hotel_links_more)
hotel_links_more_df = hotel_links_more_df.reset_index()

# Save to CSV
hotel_links_more_df.to_csv("tripadvisor_long_link_list.csv",index=False)

# Sort desending
hotel_links_more_df = hotel_links_more_df.sort_values(['index'], ascending=[False])



###############################################################################
# Review Scraper
###############################################################################
#hotel_links_more_df = pd.read_csv("data/tripadvisor_long_link_list.csv")


headers = headers_for_bs()

hotel_id, review_title, review_text, review_rating, review_date, hotel_name, num_reviews, average_rating, excellent, very_good, average, poor, terrible, tripadv_ranking = hotel_info_function(hotel_links_more_df,requests,headers,BeautifulSoup)




# convert lists to data frame
df_new_hotel_reviews = pd.DataFrame(list(zip(hotel_id, review_title, review_text, review_rating, review_date, hotel_name, num_reviews, average_rating, excellent, very_good, average, poor, terrible, tripadv_ranking)),
               columns =['hotel_id', 'review_title', 'review_text', 'review_rating', 'review_date','hotel_name', 'num_reviews', 'average_rating', 'excellent', 'very_good', 'average', 'poor', 'terrible', 'tripadv_ranking'])


df_new_hotel_reviews = df_new_hotel_reviews[df_new_hotel_reviews['average_rating']<5.1]
df_new_hotel_reviews = df_new_hotel_reviews[df_new_hotel_reviews['review_rating']<5.1]
df_new_hotel_reviews['poor'] = df_new_hotel_reviews['poor'].apply(pd.to_numeric)
df_new_hotel_reviews = df_new_hotel_reviews.reset_index(drop=True)
    
        
# add ranking  
df_new_hotel_reviews = add_ranking(df_new_hotel_reviews)

# Adjust date column
df_new_hotel_reviews['review_date'] = df_new_hotel_reviews['review_date'].str.replace('Date of stay:', '')

# Drop Duplicates
df_new_hotel_reviews = df_new_hotel_reviews.drop_duplicates(subset='review_text', keep="last")



# Save DF to csv
df_new_hotel_reviews.to_csv('data/UK_hotel_reviews.csv',encoding="utf-8")

# Save as Parquet
df_new_hotel_reviews.to_parquet("data/UK_hotel_reviews.parquet", compression=None)





