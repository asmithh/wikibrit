import pickle
import random
import re
import time

from bs4 import BeautifulSoup
import mechanize


pickle_filename = "tmz_urls_pg_1_thru_100.pkl"


def scrape_individual_article(url):
    time.sleep(random.randint(1, 20))
    browser = mechanize.Browser()
    pg = browser.open(url)
    soup = BeautifulSoup(pg, "html.parser")
    txt_full = soup.text
    patt = "[0-9]+\/[0-9]+\/[0-9]{4} [0-9]+\:[0-9]+ [A-Z]+M [A-Z]+T"
    txt = re.split(patt, txt_full)[1].split("Related Articles")[0]
    filed = re.findall(patt, txt_full)[0]
    txt = re.sub("\n", "", txt)
    return txt, filed


failed = []
urls = pickle.load(open(pickle_filename, "rb"))
res = []
for url in urls:
    try:
        res.append(scrape_individual_article(url))
    except:
        print(url)
        failed.append(url)
        continue

    if len(res) % 100 == 0:
        pickle.dump(res, open("articles_scraped.pkl", "wb"))

pickle_filename = "tmz_urls_pg_100_thru_482.pkl"
urls = pickle.load(open(pickle_filename, "rb"))
for url in urls:
    try:
        res.append(scrape_individual_article(url))
    except:
        print(url)
        failed.append(url)
        continue

    if len(res) % 100 == 0:
        pickle.dump(res, open("articles_scraped.pkl", "wb"))

pickle.dump(failed, open("failed_tmz_urls.pkl", "wb"))
pickle.dump(res, open("articles_scraped_run_2.pkl", "wb"))
