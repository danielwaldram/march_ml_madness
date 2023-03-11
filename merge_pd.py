"""
This script is used to evaluate the team_transform function, which transforms team names from the march_madness_85-21.csv
dataset. The script prints the number of teamnames that mismatched between the two databases and prints a list of these teams
to mismatched_teams.csv.
"""
import pandas as pd
from support_functions import *

def main():
    # read in the full dataset
    results = pd.read_csv('march_madness_85-21.csv')
    stats = pd.read_csv('cbb_stats_webscrape.csv')
    # Get the min and max years over which data needs to be matched. These are the years of data that will
    #   be used to train data, so we need the team names to match.
    min_year = min([results['YEAR'].max(), stats['YEAR'].max()])
    max_year = max([results['YEAR'].min(), stats['YEAR'].min()])

    # grab the same years from each: stats: 2013-2019, results: 1985-2021
    stats = stats[(min_year <= stats['YEAR']) & (max_year >= stats['YEAR'])].sort_values(['YEAR', 'TEAM'])
    results = results[(min_year <= results['YEAR']) & (max_year >= results['YEAR'])].sort_values(['YEAR', 'A_TEAM'])
    # replace all instances of State in results with St.
    results['A_TEAM'] = results['A_TEAM'].apply(teamname_transform)

    # Grab the unique values from the team column for each
    stats_unique = stats['TEAM'].unique()
    results_unique = results['A_TEAM'].unique()

    # mismatched_results grabs the teams from the tournament that can't be found in the stats
    mismatched_results = list(set(results_unique).difference(stats_unique))

    # find any remaining mismatched data and print save to csv
    mismatched_results_df = pd.DataFrame(data=mismatched_results, columns=['TEAM'])
    print(f'There are {len(mismatched_results_df)} mismatched team names')
    mismatched_results_df.to_csv('mismatched_teams.csv', index=False)


if __name__ == '__main__':
    main()
