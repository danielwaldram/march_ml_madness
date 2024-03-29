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

    merged_df.to_csv('merged_df.csv', index=False)
    # REMOVE 2019 RESULTS ONLY FOR TESTING WITH 2019
    # print(len(merged_df))
    # merged_df = merged_df[merged_df['YEAR'] != 2019]
    # print(len(merged_df))
    # Normalizing columns
    numeric_cols = merged_df.select_dtypes("number").columns
    # dropping the scores and the wins because this isn't info I will have when making a bracket
    # dropping A_YEAR and B_YEAR because they do not add any information.
    numeric_cols = numeric_cols.drop(['A_WIN', 'A_SCORE', 'B_SCORE', 'A_YEAR', 'B_YEAR'])
    merged_df[numeric_cols].mean(numeric_only=True).to_csv('stats_mean.csv', index=True)
    merged_df[numeric_cols].std(numeric_only=True).to_csv('stats_std_dev.csv', index=True)

    merged_df[numeric_cols] = (merged_df[numeric_cols] - merged_df[numeric_cols].mean(numeric_only=True)) / merged_df[numeric_cols].std(numeric_only=True)
    # shuffle
    shuffled = merged_df.sample(frac=1, random_state=1)

    # split into training and test data
    test_data = shuffled.iloc[:int(len(shuffled) * 0.2)]
    train_data = shuffled.iloc[int(len(shuffled) * 0.2):]

    # converting to a tensor form that can be input into the model
    train_tensor = tf.convert_to_tensor(train_data[numeric_cols])  # (shuffled[numeric_cols].iloc[int(len(shuffled) * 0.2):])
    test_tensor = tf.convert_to_tensor(test_data[numeric_cols])  # (shuffled[numeric_cols].iloc[:int(len(shuffled) * 0.2)])
    inp_num = len(train_tensor[0])
    return inp_num, train_tensor, test_tensor, train_data, test_data, numeric_cols

def utility_calc(matchup, scoring_sys):
    if scoring_sys == 'espn':
        # score for matchup = 2^(round -1 )*10
        return pow(2, int(matchup['ROUND']) - 1)*10
    elif scoring_sys == 'waldram':
        # score for matchup = points + max(0, diff in seed)
        rounds = [1, 2, 3, 4, 5, 6]
        points = [4, 8, 16, 20, 25, 30]
        return points[rounds.index(int(matchup['ROUND']))] + max([0, int(matchup['A_SEED']) - int(matchup['B_SEED'])])
    else:
        print("ERROR: SCORING SYSTEM NOT RECOGNIZED. SHOULD BE waldram OR espn!")
        exit()

def create_model(inp_num):
    # initialization method for the weights: xavier initialization for tanh
    x_initializer = tf.keras.initializers.GlorotNormal(seed=0)

    model = models.Sequential()
    model.add(layers.Dense(inp_num, input_dim=inp_num, activation='tanh', kernel_initializer=x_initializer))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(40, input_dim=40, activation='tanh', kernel_initializer=x_initializer))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, input_dim=30, activation='tanh', kernel_initializer=x_initializer))
    model.summary()
    return model
