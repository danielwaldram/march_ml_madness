import pandas as pd
from support_functions import *

def main():
    # read in the full dataset
    results = pd.read_csv('march_madness_85-21.csv')
    stats = pd.read_csv('cbb_stats_webscrape.csv')

    # grab the same years from each: stats: 2013-2019, results: 1985-2021
    stats = stats[2007 < stats['YEAR']][2023 > stats['YEAR']].sort_values(['YEAR', 'TEAM'])
    results = results[(2007 < results['YEAR']) & (2023 > results['YEAR'])].sort_values(['YEAR', 'A_TEAM'])
    # replace all instances of State in results with St.
    results['A_TEAM'] = results['A_TEAM'].apply(teamname_transform)

    # transform names in the results to match with stats using teamname conversion

    # Grab the unique values from the team column for each
    stats_unique = stats['TEAM'].unique()
    results_unique = results['A_TEAM'].unique()

    # mismatched_results grabs the teams from the tournament that can't be found in the stats
    mismatched_results = list(set(results_unique).difference(stats_unique))

    # find any remaining mismatched data and print save to csv
    mismatched_results_df = pd.DataFrame(data=mismatched_results, columns=['TEAM'])
    print(len(mismatched_results_df))
    mismatched_results_df.to_csv('mismatched_teams.csv', index=False)


if __name__ == '__main__':
    main()
