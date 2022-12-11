import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # read in the full dataset
    df = pd.read_csv('march_madness_85-21.csv')
    violin_plot = plt.figure()
    df['A_WIN'] = df['A_WIN'].replace(1, 'Wins')
    df['A_WIN'] = df['A_WIN'].replace(0, 'Losses')
    ax = sns.violinplot(y="A_SEED", data=df, x='A_WIN', cut=0)
    plt.xlabel("")
    plt.ylabel("Team Seed")
    #plt.show()
    plt.savefig('team_seed_violin_plot.pdf')
    plt.figure()
    sns.kdeplot(x="A_SEED", data=df, y='B_SEED', fill=True, alpha=0.5, hue='A_WIN')
    plt.ylabel("Opponent Seed")
    plt.xlabel("Team Seed")
    plt.savefig('team_and_opponent_seed_density_plot.pdf')
    plt.show()
