import pandas as pd
import numpy as np
import csv
from tensorflow.keras import layers, models
import tensorflow as tf

# average error calc for a set of examples
def average_error(y_values, predictions):
    error_sum = 0
    # run through every example
    for example in range(len(y_values)):
        error_sum += abs(y_values[example] - predictions[example][0])
    return error_sum/len(y_values)

# transform the team's name so that the correct URL is used
def transform_name_webscrape(team_name):
    if team_name[-1] is '_':
        team_name = team_name[:-1] + '.'
    return team_name.replace("__", "._").replace("_", "+")


team_conversion = pd.read_csv('teamname_conversion.csv')

def teamname_transform(team_name):
    team_name = team_name.replace('State', 'St.').replace('-', ' ')
    if team_name in list(team_conversion['RESULTS_TEAMNAME']):
        team_name = team_conversion[team_conversion['RESULTS_TEAMNAME'] == team_name]['STATS_TEAMNAME'].iloc[0]
    team_name = team_name.replace(' ', '_').replace('.', '_').replace('\'', '_').replace('&', '_')
    return team_name


def generate_test_train_data():
    # read in the full dataset
    results = pd.read_csv('march_madness_85-21.csv')
    # transform the teamnames from results to match those from the stats dataset
    results['A_TEAM'] = results['A_TEAM'].apply(teamname_transform)
    results['B_TEAM'] = results['B_TEAM'].apply(teamname_transform)

    stats = pd.read_csv('cbb_stats_webscrape.csv')

    # get the stats for both teams merged into the main df
    merged_df = results.merge(stats.add_prefix('A_'), how='inner', left_on=['A_TEAM', 'YEAR'],
                              right_on=['A_TEAM', 'A_YEAR'])
    merged_df = merged_df.merge(stats.add_prefix('B_'), how='inner', left_on=['B_TEAM', 'YEAR'],
                                right_on=['B_TEAM', 'B_YEAR'])

    # print(len(results[(2007 < results['YEAR']) & (2022 > results['YEAR'])].sort_values(['YEAR', 'A_TEAM'])))
    merged_df.to_csv('merged_df.csv', index=False)
    # REMOVE 2019 RESULTS ONLY FOR TESTING WITH 2019
    print(len(merged_df))
    merged_df = merged_df[merged_df['YEAR'] != 2019]
    print(len(merged_df))
    # Normalizing columns
    s = merged_df.select_dtypes("number").columns
    s = s.drop(['A_WIN', 'A_SCORE', 'B_SCORE', 'A_YEAR', 'B_YEAR'])
    merged_df[s].mean(numeric_only=True).to_csv('stats_mean.csv', index=True)
    merged_df[s].std(numeric_only=True).to_csv('stats_std_dev.csv', index=True)

    merged_df[s] = (merged_df[s] - merged_df[s].mean(numeric_only=True)) / merged_df[s].std(numeric_only=True)
    # shuffle
    shuffled = merged_df.sample(frac=1, random_state=1)
    numeric_cols = shuffled.select_dtypes("number").columns
    # dropping the scores and the wins because this isn't info I will have when making a bracket
    # dropping A_YEAR and B_YEAR because they do not add any information.
    # numeric_cols = numeric_cols.drop(['A_WIN', 'A_SCORE', 'B_SCORE', 'A_YEAR', 'B_YEAR'])

    # split into training and test data
    test_data = shuffled.iloc[:int(len(shuffled) * 0.2)]
    train_data = shuffled.iloc[int(len(shuffled) * 0.2):]

    # converting to a tensor form that can be input into the model
    train_tensor = tf.convert_to_tensor(shuffled[s].iloc[int(len(shuffled) * 0.2):])
    test_tensor = tf.convert_to_tensor(shuffled[s].iloc[:int(len(shuffled) * 0.2)])
    inp_num = len(train_tensor[0])
    return inp_num, train_tensor, test_tensor, train_data, test_data