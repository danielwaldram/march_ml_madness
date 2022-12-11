import pandas as pd
import numpy as np
import csv

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
