"""
This script makes tournament predictions based on the neural network trained and saved as my_best_mode.hdf5. The script
takes in the full name of a csv with the input teams. Example format for input teams: "2022_tourney_input.csv".
Winning teams are selected based on the neural networks prediction for a given matchup starting with the first round.
The output selections is saved as tourney_output_22.csv.
"""
import pandas as pd
import sys
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from support_functions import *

def main():
    if len(sys.argv) < 2:
        print("TOO FEW ARGUMENTS PASSED. INCLUDE THE FULL NAME OF THE CSV FILE WITH INPUT TEAMS.")
    elif len(sys.argv) > 2:
        print("TOO MANY ARGUMENTS PASSED. ONLY PASS THE FULL NAME OF THE CSV FILE WITH INPUT TEAMS.")
    # read in the full dataset
    tourney_in = pd.read_csv(sys.argv[1])
    year = tourney_in['YEAR'][0]
    tourney_out = pd.DataFrame(columns=tourney_in.columns)
    tourney_out['prediction'] = []
    tourney_out['round_prediction'] = []
    print(tourney_out)
    tourney_round = 1
    # transform the teamnames from results to match those from the stats dataset
    tourney_in['A_TEAM'] = tourney_in['A_TEAM'].apply(teamname_transform)
    tourney_in['B_TEAM'] = tourney_in['B_TEAM'].apply(teamname_transform)
    stats = pd.read_csv('cbb_stats_webscrape.csv')
    # get the stats for both teams merged into the main df
    merged_df = tourney_in.merge(stats.add_prefix('A_'), how='inner', left_on=['A_TEAM', 'YEAR'],
                                 right_on=['A_TEAM', 'A_YEAR'])
    merged_df = merged_df.merge(stats.add_prefix('B_'), how='inner', left_on=['B_TEAM', 'YEAR'],
                                right_on=['B_TEAM', 'B_YEAR'])
    merged_df = merged_df.drop(columns=['A_YEAR', 'B_YEAR'])
    # Normalizing columns based on the training data values
    mean_df = pd.read_csv('stats_mean.csv')
    sd_df = pd.read_csv('stats_std_dev.csv')
    # normalize the numeric inputs
    for column in range(len(mean_df)):
        numeric_col = mean_df.iloc[column]['Unnamed: 0']
        if numeric_col == 'ROUND':
            av_round = mean_df.iloc[column]['0']
            sd_round = sd_df.iloc[column]['0']
        mean_val = mean_df.iloc[column]['0']
        std_dev_val = sd_df.iloc[column]['0']
        merged_df[numeric_col] = (merged_df[numeric_col] - mean_val) / std_dev_val

    s = merged_df.select_dtypes("number").columns
    print(s)

    inp_num = len(s)

    # model setup
    model = models.Sequential()
    model.add(layers.Dense(inp_num, input_dim=inp_num, activation='tanh'))
    model.add(layers.Dense(100, input_dim=inp_num, activation='tanh'))
    model.add(layers.Dense(100, input_dim=100, activation='tanh'))
    model.add(layers.Dense(inp_num, input_dim=inp_num, activation='tanh'))
    model.add(layers.Dense(1, input_dim=inp_num, activation='tanh'))
    model.summary()
    model.load_weights('my_best_mode.hdf5')

    # While loop goes through every round 1-6
    while tourney_round < 7:
        # converting to a tensor form that can be input into the model
        data_tensor = tf.convert_to_tensor(merged_df[s].to_numpy().astype('float'))

        # Match up predictions
        predictions = model.predict(data_tensor)
        round_predictions = predictions.round()

        # create a new df for the next round
        next_round = pd.DataFrame(columns=merged_df.columns)
        for match_up in range(int(len(merged_df) / 2)):
            # add an empty row to the new df
            new_row = pd.Series([None] * len(merged_df.iloc[0]), index=merged_df.columns)
            next_round = next_round.append(new_row, ignore_index=True)
            # set round and year
            next_round.iloc[match_up]['ROUND'] = (tourney_round + 1 - av_round) / sd_round
            next_round.iloc[match_up]['YEAR'] = merged_df.iloc[0]['YEAR']
            # in this case the A team from last round is the A team for the next
            if round_predictions[match_up * 2] == 1:
                for col in merged_df.filter(like='A_').columns:
                    next_round.iloc[match_up][col] = merged_df.iloc[match_up * 2][col]
            else:
                for col in merged_df.filter(like='A_').columns:
                    next_round.iloc[match_up][col] = merged_df.iloc[match_up * 2][col.replace('A_', 'B_')]
            # in this case the A team from last round is the B team for the next
            if round_predictions[match_up * 2 + 1] == 1:
                for col in merged_df.filter(like='B_').columns:
                    next_round.iloc[match_up][col] = merged_df.iloc[match_up * 2 + 1][col.replace('B_', 'A_')]
            else:
                for col in merged_df.filter(like='B_').columns:
                    next_round.iloc[match_up][col] = merged_df.iloc[match_up * 2 + 1][col]

        # add to the tourney_out dataframe
        merged_df['prediction'] = predictions
        merged_df['round_prediction'] = round_predictions
        merged_df['ROUND'] = [tourney_round] * len(merged_df)
        tourney_out = pd.concat([tourney_out, merged_df[
            ['YEAR', 'ROUND', 'A_SEED', 'A_TEAM', 'B_SEED', 'B_TEAM', 'prediction', 'round_prediction']]])
        # get ready for the next loop
        merged_df = next_round.copy()
        tourney_round += 1

    # Make the output not normalized and output to csv
    tourney_out['A_SEED'] = tourney_out['A_SEED'] * sd_df.loc[sd_df['Unnamed: 0'] == 'A_SEED']['0'].values[0] + \
        mean_df.loc[mean_df['Unnamed: 0'] == 'A_SEED']['0'].values[0]
    tourney_out['B_SEED'] = tourney_out['B_SEED'] * sd_df.loc[sd_df['Unnamed: 0'] == 'B_SEED']['0'].values[0] + \
        mean_df.loc[mean_df['Unnamed: 0'] == 'B_SEED']['0'].values[0]
    tourney_out['YEAR'] = [year] * len(tourney_out)
    tourney_out.to_csv('tourney_output_22.csv', index=False)


if __name__ == '__main__':
    main()
