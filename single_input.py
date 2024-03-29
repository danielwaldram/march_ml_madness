"""
This script is to train a neural network with only a single input. The model trained is saved as my_best_mode_single_input.hdf5.
For evaluating the model, the script outputs training and testing error with and without rounding.
"""
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from support_functions import *


def main():
    inp_num, train_tensor, test_tensor, train_data, test_data, numeric_cols = generate_test_train_data()
    # converting to a tensor form that can be input into the model
    print(train_data['A_SEED'])
    train_tensor = tf.convert_to_tensor(train_data['A_SEED'])
    test_tensor = tf.convert_to_tensor(test_data['A_SEED'])
    # inp_num = len(train_tensor[0])

    # callbacks save only the best model and stop the model running early if results aren't improving
    callback_a = ModelCheckpoint(filepath='my_best_mode_single_input.hdf5', monitor='val_mse', mode='min', save_best_only=True,
                                 save_weights_only=True)
    callback_b = EarlyStopping(monitor='val_mse', mode='min', patience=30, verbose=1)

    # initialization method for the weights: xavier initialization for tanh
    x_initializer = tf.keras.initializers.GlorotNormal()

    # model setup
    model = models.Sequential()
    model.add(layers.Dense(1, input_dim=1, activation='tanh', kernel_initializer=x_initializer))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse', metrics='mse',
                  optimizer=optimizer)  # loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(train_tensor, tf.convert_to_tensor(train_data['A_WIN']), validation_data=(test_tensor,
        tf.convert_to_tensor(test_data['A_WIN'])), epochs=200, callbacks=[callback_a, callback_b])  # , class_weight=weighting)
    model.load_weights('my_best_mode_single_input.hdf5')

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
