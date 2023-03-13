"""
This script makes tournament predictions based on the neural network trained and saved as my_best_mode.hdf5. The script
takes in the full name of a csv with the input teams. Example format for input teams: "2022_tourney_teams.csv".
Winning teams are selected by estimating their total points up to the curent round (utility) starting with the final
round. The formula for utility is (probability*points + utility for last round). The probability of a team winning a
matchup is estimated by the neural network. The points a team scores in a given round are based on the rules for the
bracket challenge being used. The scoring system is passed as input to script.

INPUTS - listed in the order they should be passed as arguments
scoring system (The scoring system options and their descriptions are shown below)
    espn - basic scoring system in which potential points double each round
    waldram - potential points increase each round. Upsets are also rewarded based on the difference between the teams seeds.
input teams csv file - include the full name of the file. Example: '2022_tourney_teams.csv'
The output selections is saved as tourney_output_22.csv.
"""
# import itertools package
import itertools
import sys
import pandas as pd
from itertools import permutations
from support_functions import *
from tensorflow.keras import layers, models

if len(sys.argv) < 3:
    print("ERROR: TOO FEW ARGUMENTS PASSED. INCLUDE SCORING SYSTEM (espn or waldram) AND INPUT FILE NAME.")
    exit()
elif len(sys.argv) > 3:
    print("ERROR: TOO MANY ARGUMENTS PASSED. ONLY PASS SCORING SYSTEM (espn or waldram) AND INPUT FILE NAME.")
    exit()
scoring_sys = sys.argv[1]


# ------ READ IN THE TOURNAMENT TEAMS AND CREATE FULL DATASET WITH STATS  ------ #
# read in the full dataset
tourney_in = pd.read_csv(sys.argv[2])
# transform the teamnames from results to match those from the stats dataset
tourney_in['TEAM'] = tourney_in['TEAM'].apply(teamname_transform)
# initialize lists of all teams
list1 = list(tourney_in['TEAM'])
# get list of every possible combination of teams and create pd
unique_combinations = itertools.combinations(list1, 2)
combinations_pd = pd.DataFrame(list(unique_combinations), columns=['A_TEAM', 'B_TEAM'])
# add in the year  and the seed of both teams
combinations_pd['YEAR'] = tourney_in['YEAR'].iloc[0]
# Adding in the seeds for all the teams
combinations_pd = combinations_pd.merge(tourney_in.add_prefix('A_')[['A_TEAM', 'A_SEED']], how='left', left_on=['A_TEAM'],
                                right_on=['A_TEAM'])
combinations_pd = combinations_pd.merge(tourney_in.add_prefix('B_')[['B_TEAM', 'B_SEED']], how='left', left_on=['B_TEAM'],
                                right_on=['B_TEAM'])
# ------------ #

# ------ CREATE A LIST OF ROUNDS TO ADD TO THE COMBINATIONS DATABASE ------ #
# add the rounds of the tournament in which teams could meet into the dataframe
rounds = [[] for x in range(64)]
doubling_val = 2
# loop through each round
for i in range(6):
    # offsets contains the number of teams in a row that will be in a given round (1st round only has 1 in a row)
    offsets = range(int(doubling_val/2))
    # loop through each team in the tournament team value is the starting point. Offset is used to add other teams
    for team in range(int(64/doubling_val)):
        for offset in offsets:
            # add rounds to a given teams list
            rounds[team*doubling_val+offset] = rounds[team*doubling_val+offset] + [i + 1]*int(doubling_val/2)
    doubling_val += doubling_val
# finally break the lists into a single one to add to combinations db
full_rounds = [j for team in rounds for j in team]
combinations_pd['ROUND'] = full_rounds
# ------------ #

# ------ BRING IN THE STATS FOR EACH TEAM ------ #
stats = pd.read_csv('cbb_stats_webscrape.csv')
# get the stats for both teams merged into a df
merged_df = combinations_pd.merge(stats.add_prefix('A_'), how='left', left_on=['A_TEAM', 'YEAR'],
                              right_on=['A_TEAM', 'A_YEAR'])
merged_df = merged_df.merge(stats.add_prefix('B_'), how='left', left_on=['B_TEAM', 'YEAR'],
                                right_on=['B_TEAM', 'B_YEAR'])
merged_df.to_csv('utility_stats.csv')
# ------------ #

# ------ CREATE INPUT DF THAT FEEDS INTO THE NEURAL NET ------ #
input_cols = pd.read_csv('input_cols.csv')
input_df = merged_df[list(input_cols['input_cols'])]
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
    input_df[numeric_col] = (input_df[numeric_col] - mean_val) / std_dev_val
inp_num = len(input_df.iloc[0])
# ------------ #

# ------ ADD PREDICTIONS FOR EACH POSSIBLE MATCHUP ------ #
# model setup
model = create_model(inp_num)
model.load_weights('my_best_mode.hdf5')

# converting to a tensor form that can be input into the model
data_tensor = tf.convert_to_tensor(input_df.to_numpy().astype('float'))
predictions = model.predict(data_tensor)

# apply threshold of 0 and 1 to the the predictions
for prediction in range(len(predictions)):
    if predictions[prediction] > 1:
        predictions[prediction] = 1
    elif predictions[prediction] < 0:
        predictions[prediction] = 0
combinations_pd['predictions'] = predictions

# add the inverse prediction of every matchup to the dataframe
inverse_combinations_pd = combinations_pd.copy().rename(columns={"A_TEAM": "B_TEAM", "B_TEAM": "A_TEAM", "A_SEED": "B_SEED", "B_SEED": "A_SEED"})
inverse_combinations_pd['predictions'] = 1 - inverse_combinations_pd['predictions']
combinations_pd = pd.concat([combinations_pd, inverse_combinations_pd], ignore_index=True)
# ------------ #
combinations_pd.to_csv('trash.csv', index=False)

# ------ CREATE PROBABILITY TABLE FOR EACH TEAM IN EACH ROUND ------ #
probability_table = pd.DataFrame(data=tourney_in['TEAM'], columns=['TEAM'])
utility_table = pd.DataFrame(data=tourney_in['TEAM'], columns=['TEAM'])
prob_list = []
q_val_list = []
for team in probability_table['TEAM']:
    # grab the team and all its matchups in a given round
    prob_list.append(combinations_pd[(combinations_pd['A_TEAM'] == team) & (combinations_pd['ROUND'] == 1)]['predictions'].mean())
    q_val_list.append(combinations_pd[(combinations_pd['A_TEAM'] == team) & (combinations_pd['ROUND'] == 1)]['predictions'].mean()*utility_calc(combinations_pd[(combinations_pd['A_TEAM'] == team) & (combinations_pd['ROUND'] == 1)], scoring_sys))
probability_table['ROUND_1'] = prob_list
utility_table['ROUND_1'] = q_val_list
# iterate through round 2 to 6
for rnd in range(2, 7):
    prob_list = []
    q_val_list = []
    for team in probability_table['TEAM']:
        # select the df for the team and round
        sub_select = combinations_pd[(combinations_pd['A_TEAM'] == team) & (combinations_pd['ROUND'] == rnd)]
        prob = 0
        q_val = 0
        for row in range(len(sub_select)):
            # get the probability that the team matched up against made it
            prob_of_matchup = float(probability_table[probability_table['TEAM'] == sub_select.iloc[row]['B_TEAM']]['ROUND_' + str(rnd - 1)])
            # get the probability that the team would win this matchup
            prob_of_win = float(combinations_pd[(combinations_pd['A_TEAM'] == team) & (combinations_pd['B_TEAM'] == sub_select.iloc[row]['B_TEAM'])]['predictions'])
            # multiply probabilities together and add to the total probability
            prob += prob_of_matchup*prob_of_win
            q_val += utility_calc(sub_select.iloc[row], scoring_sys)*prob_of_matchup*prob_of_win
        # finally, the probability that a team will make it to the round is the sum times the probability that the team made it through the last round
        prob_list.append(prob*float(probability_table[probability_table['TEAM'] == sub_select.iloc[row]['A_TEAM']]['ROUND_' + str(rnd - 1)]))
        # q_val still needs the previous rounds added on
        q_val_list.append(q_val*float(probability_table[probability_table['TEAM'] == sub_select.iloc[row]['A_TEAM']]['ROUND_' + str(rnd - 1)])
                          + float(utility_table[utility_table['TEAM'] == sub_select.iloc[row]['A_TEAM']]['ROUND_' + str(rnd - 1)]))
    # add on the new round
    probability_table['ROUND_' + str(rnd)] = prob_list
    utility_table['ROUND_' + str(rnd)] = q_val_list

utility_table.to_csv('utility_table.csv', index=False)
probability_table.to_csv('prob_table.csv', index=False)
# ------------ #

# ------ SELECT WINNING TEAMS ------ #
final_teams = pd.DataFrame(columns=['rounds', 'teams'], data=[])
doubling_val = 1
# loop through each round starting with the last
for t_round in range(6):
    # loop through each subgroup of teams for which we need 1 winner
    # chop the table into doubling val number of sections
    for sub_group in range(doubling_val):
        # sub_table is the subgroup of teams
        sub_table = utility_table.iloc[int(len(utility_table)/doubling_val)*sub_group:int(len(utility_table)/doubling_val)*(sub_group+1)]
        # check that none of the teams in the subgroup is already in the final teams table by looking for duplicates
        if len(list(set(list(final_teams['teams']))) + list(sub_table['TEAM'])) != len(set(list(final_teams['teams']) + list(sub_table['TEAM']))):
            # in this case this team has already been selected to win so we continue
            continue
        else:
            # if the team isn't already a winner, they get added on for this round and any lower rounds, which they have to have won to get this far
            winning_team = sub_table[sub_table['ROUND_' + str(6 - t_round)] == sub_table['ROUND_' + str(6 - t_round)].max()]['TEAM'].iloc[0]
            final_teams = final_teams.append(pd.DataFrame({'rounds': list(range(1, 7 - t_round)), 'teams': [winning_team] * (6 - t_round)}))
    doubling_val += doubling_val

final_teams.sort_values(by='rounds').to_csv('utility_final_team_selections.csv', index=False)
# ------------ #
