import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from support_functions import *


def main():
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
    print(test_data)
    # converting to a tensor form that can be input into the model
    train_tensor = tf.convert_to_tensor(shuffled[s].iloc[int(len(shuffled) * 0.2):])
    test_tensor = tf.convert_to_tensor(shuffled[s].iloc[:int(len(shuffled) * 0.2)])
    inp_num = len(train_tensor[0])
    print(inp_num)

    # callbacks save only the best model and stop the model running early if results aren't improving
    callback_a = ModelCheckpoint(filepath='my_best_mode.hdf5', monitor='val_mse', mode='min', save_best_only=True,
                                 save_weights_only=True)
    callback_b = EarlyStopping(monitor='val_mse', mode='min', patience=30, verbose=1)

    # initialization method for the weights: xavier initialization for tanh
    x_initializer = tf.keras.initializers.GlorotNormal()

    # model setup
    model = models.Sequential()
    model.add(layers.Dense(inp_num, input_dim=inp_num, activation='tanh', kernel_initializer=x_initializer))
    model.add(layers.Dense(100, input_dim=inp_num, activation='tanh', kernel_initializer=x_initializer))
    model.add(layers.Dense(100, input_dim=100, activation='tanh', kernel_initializer=x_initializer))
    model.add(layers.Dense(inp_num, input_dim=inp_num, activation='tanh', kernel_initializer=x_initializer))
    model.add(layers.Dense(1, input_dim=inp_num, activation='tanh', kernel_initializer=x_initializer))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse', metrics='mse',
                  optimizer=optimizer)  # loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(train_tensor, tf.convert_to_tensor(train_data['A_WIN']), validation_data=(test_tensor,
        tf.convert_to_tensor(test_data['A_WIN'])), epochs=200, callbacks=[callback_a, callback_b])  # , class_weight=weighting)
    model.load_weights('my_best_mode.hdf5')

    # Testing error
    predictions = model.predict(test_tensor)
    testing_error = average_error(test_data['A_WIN'].tolist(), predictions)
    round_predictions = predictions.round()
    test_data['predictions'] = predictions
    test_data['round_predictions'] = round_predictions

    # Training error
    predictions_train = model.predict(train_tensor)
    round_train_predictions = predictions_train.round()
    training_error = average_error(train_data['A_WIN'].tolist(), predictions_train)
    train_data['predictions'] = predictions_train

    # Output
    print("Testing error: {}".format(testing_error))
    print("Round testing error: {}".format(average_error(test_data['A_WIN'].tolist(), round_predictions)))
    print("Training error: ", training_error)
    print("Round training error: {}".format(average_error(train_data['A_WIN'].tolist(), round_train_predictions)))
    test_data.to_csv('output_combined_data.csv', index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
