import requests
from bs4 import BeautifulSoup
import pandas as pd

def data_for_year(year, df_cols):
    URL = "https://barttorvik.com/trank.php?year=" + str(year) + "&sort=&top=0&conlimit=All&venue=All&type=All#"
    year_df = pd.DataFrame(columns=df_cols)
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="content")

    seed_rows = results.find_all("tr", class_="seedrow")
    for seed_row in seed_rows:
        stats_row = []
        stats_row.append(seed_row.find("td", class_="teamname")['id'])
        stats_row.append(seed_row.find("td", class_="mobileout").text.strip())
        seed_row.find("td", class_="5").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(float(seed_row.find("td", class_="5").find("a").text.strip().split('-')[0])/float(seed_row.find("td", class_="6").text.strip()))
        seed_row.find("td", class_="1").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="1").text.strip())
        seed_row.find("td", class_="2").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="2").text.strip())
        seed_row.find("td", class_="3").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="3").text.strip())
        seed_row.find("td", class_="7").find("span", {"class":"lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="7").text.strip())
        seed_row.find("td", class_="8").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="8").text.strip())
        seed_row.find("td", class_="11").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="11").text.strip())
        seed_row.find("td", class_="12").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="12").text.strip())
        seed_row.find("td", class_="13").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="13").text.strip())
        seed_row.find("td", class_="14").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="14").text.strip())
        seed_row.find("td", class_="9").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="9").text.strip())
        seed_row.find("td", class_="10").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="10").text.strip())
        seed_row.find("td", class_="16").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="16").text.strip())
        seed_row.find("td", class_="17").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="17").text.strip())
        seed_row.find("td", class_="18").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="18").text.strip())
        seed_row.find("td", class_="19").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="19").text.strip())
        seed_row.find("td", class_="26").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="26").text.strip())
        seed_row.find("td", class_="34").find("span", {"class": "lowrow"}).replaceWith('')
        stats_row.append(seed_row.find("td", class_="34").text.strip())
        stats_row.append(year)
        year_df.loc[len(year_df)] = stats_row
    return year_df

def main():
    stats_cols = pd.read_csv('cbb.csv').columns
    stats_cols = stats_cols.drop(['POSTSEASON', 'SEED', 'W'])
    stats_df = pd.DataFrame(columns=stats_cols)
    for yr in range(2008, 2023):
        print(yr)
        stats_df = pd.concat([stats_df, data_for_year(yr, stats_cols)])
    stats_df.to_csv('cbb_stats_webscrape.csv', index=False)


if __name__ == '__main__':
    main()
