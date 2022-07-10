##############################################################################
##############################################################################
#                                                                            #
#                                 Functions                                  #
#                          TripAdvisor Review Scraper                        #
#                               Master Thesis                                #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import re
import numpy as np



def headers_for_bs():
    hd = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
                              'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
                                                   'accept-encoding': 'gzip, deflate, br',
                                                   'accept-language': 'en-US,en;q=0.9',
                                                   'upgrade-insecure-requests': '1',
                                                   'scheme': 'https'}
    return hd





def hotel_info_function(hotel_links,requests,headers,BeautifulSoup):
    
    counter = 0
    hotel_id = []
    review_title = []
    review_text = []
    review_rating = []
    review_date = []
    hotel_id = []
    hotel_name = []
    num_reviews = []
    average_rating = []
    excellent = []
    very_good = []
    average = []
    poor = []
    terrible = []
    tripadv_ranking = []

    for h in range(0, len(hotel_links)):

        url = hotel_links.iloc[h,1]
        
   
        try: 
            resp = requests.get(url, headers = headers)
            soup = BeautifulSoup(resp.content, "html.parser")
        
        
            hi = hotel_links.iloc[h,0]
            try: 
                hn = soup.find('h1', class_='QdLfr b d Pn').text
            except:
                hn = "No Hotel Name Available"
 
               
            
            try: 
                trp_rank = soup.find('div', class_="cGAqf").text
            except:
                trp_rank = np.nan
            
            try:
                nr = soup.find('span', class_="qqniT").text
                nr = int(re.sub("[^0-9]", "", nr))
            except:
                nr = np.nan
            
            try: 
                av_rat = float(soup.find('span', class_="IHSLZ P").text)

            except:
                av_rat = np.nan
            
            # review categories
            exel = int(re.sub("[^0-9]", "", soup.find("input", {"id": "ReviewRatingFilter_5"}).text))
            vg = int(re.sub("[^0-9]", "", soup.find("input", {"id": "ReviewRatingFilter_4"}).text))
            av = int(re.sub("[^0-9]", "", soup.find("input", {"id": "ReviewRatingFilter_3"}).text))
            po = int(re.sub("[^0-9]", "", soup.find("input", {"id": "ReviewRatingFilter_2"}).text))
            ter = int(re.sub("[^0-9]", "", soup.find("input", {"id": "ReviewRatingFilter_1"}).text))
            
            # most recent reviews
            review_containers = soup.find_all('div', class_='YibKl MC R2 Gi z Z BB pBbQr')
            

            


            for i in range(0,len(review_containers)):
                try:
                    datee = review_containers[i].find('span', class_="teHYY _R Me S4 H3").text
                    review_date.append(datee)
                except AttributeError:
                    review_date.append(np.nan)
                    
                try: 
                    titlee = review_containers[i].find('div', class_="KgQgP MC _S b S6 H5 _a").text
                    review_title.append(titlee)
                except:
                    review_title.append(np.nan)
                    
                try: 
                    textee = review_containers[i].find('q', class_="QewHA H4 _a").text
                    review_text.append(textee)
                except:
                    review_text.append(np.nan)
                
                
                try: 
                    input_tag = review_containers[i].find('div', class_="Hlmiy F1").find('span')
                    review_rating.append(int(re.sub("[^0-9]", "", input_tag['class'][1])) /10 )
                except:
                    review_rating.append(np.nan)
                
                hotel_id.append(hi)
                hotel_name.append(hn)
                num_reviews.append(nr)
                average_rating.append(av_rat)
                
                excellent.append(exel)
                very_good.append(vg)
                average.append(av)
                poor.append(po)
                terrible.append(ter)
                
                tripadv_ranking.append(trp_rank)
            
            counter = counter +1
            
            print(counter, "of", len(hotel_links))

        except:
            print("no such url")
    
    return hotel_id, review_title, review_text, review_rating, review_date, hotel_name, num_reviews, average_rating, excellent, very_good, average, poor, terrible, tripadv_ranking
    









