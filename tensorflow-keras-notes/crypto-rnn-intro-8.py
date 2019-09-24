"""
overview
---
here we are dealing with 4 major crypto currencies. we are going to use the last 60 minutes of data from the cryptos,
and based on that, we will try to predict what the price will be in the future.
we will have to take our dataset and take it from one long sequence, and form it into multiple sequences.
we will then have to balance the data, normalize the data, and scale the data.
our dataset is four csv files (one for each crypto)
"""

"""
importing the data
---
"""
import pandas as pd

main_df = pd.DataFrame()

ratios = ['BTC', 'LTC', 'ETH', 'BCH']
for ratio in ratios:
    dataset = f'DataSets/crypto_data/{ratio}-USD.csv'

    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    # print(df.head())
    df.rename(columns={'close': f'{ratio}_close', 'volume': f'{ratio}_volume'},
              inplace=True)  # inplace used to not have to redefine dataframe

    df.set_index('time', inplace=True)

    df = df[[f'{ratio}_close', f'{ratio}_volume']]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

# print(main_df.head())


"""
targets
---
while we have our sequential data, we still need targets.  
we start by defining some constants, the length of time past data we are going to use, the period of time we are
going to try to predict, and the ratio we are going to attempt to predict.
we then need to create the targets, and define the rules for the targets (classify).
we also need to create a future column, with data from 3 minutes ahead of the time from the close column.
we then need to map the classify function to a new column, target.
"""

SEQ_LEN = 60  # take the last 60 minutes
FUTURE_PREIOD_PREDICT = 3  # predict 3 minutes into the future
RATIO_TO_PREDICT = 'LTC'


def classify(current, future):
    # based on future being greater than current, we want to train the network to think that that is good
    # would probably want a hold for equal from here
    if float(future) > float(current):
        return 1
    else:
        return 0


# data import usually here

main_df[f'{RATIO_TO_PREDICT}_future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PREIOD_PREDICT)

main_df[f'{RATIO_TO_PREDICT}_target'] = list(map(classify,
                                                 main_df[f'{RATIO_TO_PREDICT}_close'],
                                                 main_df[f'{RATIO_TO_PREDICT}_future']))

# print(main_df[[f'{RATIO_TO_PREDICT}_close', f'{RATIO_TO_PREDICT}_future', f'{RATIO_TO_PREDICT}_target']])

""" START PART 9 (pt.2 of rnn) """

"""
creating sequences
---
you can't just shuffle the data, and sequence a random 10% because all out of sample examples would have close insample
data. what you have to do with sequential data, you have to take a sequence out of it. here we are taking the last 5%
of our data and will sequence it out.
"""

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

from sklearn import preprocessing
from collections import deque
import numpy as np
import random


def preprocess_df(df):
    # usually up top with other functions
    df = df.drop(f'{RATIO_TO_PREDICT}_future', 1)

    # this is normalizing all of the columns other than the target column
    # then dropping all na values, and scaling between 0 and 1
    for col in df.columns:
        if col != f'{RATIO_TO_PREDICT}_target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        # appending sequence of lists as a list to prev_days, without taking target
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    # train_x, train_y = preprocess_df(main_df)
    # val_x, val_y = preprocess_df(validation_main_df)

    """ START PART 10 (pt.3 of rnn) """

    """
    balancing the data
    ---


    """

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


train_x, train_y = preprocess_df(main_df)
val_x, val_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(val_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {val_y.count(0)}, buys: {val_y.count(1)}")

""" START PART 11 (pt.4 of rnn) """

import time

EPOCHS = 10
BATCH_SIZE = 64
NAME = f'{RATIO_TO_PREDICT}-{SEQ_LEN}-seq-{FUTURE_PREIOD_PREDICT}-pred-{int(time.time())}'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs\{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                      mode='max'))  # saves only the best ones

history = model.fit(train_x, train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_x, val_y),
                    callbacks=[tensorboard, checkpoint])
