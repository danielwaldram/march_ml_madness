import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

# average error calc for a set of examples
def average_error(y_values, predictions):
    error_sum = 0
    # run through every example
    for example in range(len(y_values)):
        error_sum += abs(y_values[example] - predictions[example][0])
    return error_sum/len(y_values)

def convert_to_binary(float_predictions):
    # run through every example
    for example in range(len(y_values)):
        error_sum += abs(y_values[example] - predictions[example][0])
    return error_sum / len(y_values)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # read in the full dataset
    df = pd.read_csv('march_madness_85-21.csv')
    # shuffle
    shuffled = df.sample(frac=1, random_state=1)
    # split into training and test data
    test_data = shuffled.iloc[:int(len(shuffled)*0.2)]
    train_data = shuffled.iloc[int(len(shuffled)*0.2):]

    tensor_version = tf.convert_to_tensor([train_data['A_SEED'], train_data['B_SEED']])
    print(tensor_version)

    # numeric_input = layers.Input(shape=(1,), dtype=tf.float32)
    callback_a = ModelCheckpoint(filepath='my_best_mode.hdf5', monitor='val_mse', mode='min', save_best_only=True, save_weights_only=True)
    callback_b = EarlyStopping(monitor='val_mse', mode='min', patience=30, verbose=1)

    # initialization method for the weights
    initializer = tf.keras.initializers.HeNormal()  # He initialization for relu
    x_initializer = tf.keras.initializers.GlorotNormal()  # xavier initialization for tanh

    activations = ['relu', 'tanh']
    initializers = [initializer, x_initializer]

    model = models.Sequential()
    model.add(layers.Dense(100, input_dim=2, activation='tanh', kernel_initializer=x_initializer))
    model.add(layers.Dense(1, input_dim=100, activation='tanh', kernel_initializer=x_initializer))
    model.summary()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', metrics='mse') # loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(tf.transpose(tf.convert_to_tensor([train_data['A_SEED'], train_data['B_SEED']])), tf.convert_to_tensor(train_data['A_WIN']), validation_data=(tf.transpose(tf.convert_to_tensor([test_data['A_SEED'], test_data['B_SEED']])),
                tf.convert_to_tensor(test_data['A_WIN'])), epochs=200, callbacks=[callback_a, callback_b])  # , class_weight=weighting)
    model.load_weights('my_best_mode.hdf5')

    # Testing error
    predictions = model.predict(tf.transpose(tf.convert_to_tensor([test_data['A_SEED'], test_data['B_SEED']])))
    testing_error = average_error(test_data['A_WIN'].tolist(), predictions)
    round_predictions = predictions.round()
    test_data['predictions'] = predictions
    test_data['round_predictions'] = round_predictions

    # Training error
    predictions_train = model.predict(tf.transpose(tf.convert_to_tensor([train_data['A_SEED'], train_data['B_SEED']])))
    training_error = average_error(train_data['A_WIN'].tolist(), predictions_train)
    train_data['predictions'] = predictions_train

    # Output
    print("Testing error: {}".format(testing_error))
    print("Round testing error: {}".format(average_error(test_data['A_WIN'].tolist(), round_predictions)))
    print("Training error: ", training_error)
    test_data.to_csv('output.csv', index=False)
