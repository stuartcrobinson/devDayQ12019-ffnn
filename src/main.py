import datetime
import json

import random
import numpy

import _3_addArraysToPrepdData as _3
import constants as C

#

# build X for keras

# now load sorted games w/ nn arrays (preppedWithnnArrays)
# iterate through.  with index.
# const:    - MIN_HISTORICAL_GAMES - must b true for home and away
#           - MAX_DAYS_LOOKBACK    - maximum number of prior days to include in X -- 1 year?
#           - RECENCY_BIAS_PARAMETER - 5x for past 2 days, 4x for past week, 3x for past month, 2x for past 2 months, 1x for past MAX_DAYS_LOOKBACK
#           - MIN_PREV_GAMES - irrelevant of teams.  cos models are built per day not per game


MIN_HISTORICAL_GAMES = 5
MIN_PREV_GAMES = 70
MAX_DAYS_LOOKBACK = 365

Y_METRIC = C.y_isOverFinal


# todo don't build X per game!!!! build per DAY
# ugh okay how to determine if new day ...


def setDatetimePythonDate(games):
    for game in games:
        game['pythonDate'] = _3.buildDateForObj(game)


#           - RECENCY_BIAS_PARAMETER - 5x for past 2 days, 4x for past week, 3x for past month, 2x for past 2 months, 1x for past MAX_DAYS_LOOKBACK
def getRecencyBiasFactor(numDaysAgo):
    if numDaysAgo <= 2:
        return 5
    if numDaysAgo <= 7:
        return 4
    if numDaysAgo <= 31:
        return 3
    if numDaysAgo <= 62:
        return 2
    return 1


# print(shuffleTogether([1, 2, 3, 4], [11, 22, 33, 44])) #to check
def shuffleTogether(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def getTrainingData(i, games, maxDaysLookback):
    game = games[i]
    x_arrays = []
    y_values = []
    k = i
    while k >= 0:
        k -= 1
        gameIter = games[k]
        numDaysAgo = _3.daysBetween(game, gameIter)
        if numDaysAgo < 1:
            continue
        if numDaysAgo > maxDaysLookback:
            break
        # okay so gameIter is a valid game, so add it to X
        if gameIter[C.y_isOverFinal] is not None:
            for _ in range(getRecencyBiasFactor(numDaysAgo)):
                x_arrays.append(gameIter[C.x])
                y_values.append(gameIter[Y_METRIC])
    # shuffle doesnt take much extra time.  important so validation data selected randomly as opposed to most recent (MOST IMPORTANT games)
    x_arrays, y_values = shuffleTogether(x_arrays, y_values)
    return x_arrays, y_values


def getTestData(games):
    x_arrays = []
    y_values = []
    for game in games:
        x_arrays.append(game[C.x])
        y_values.append(game[Y_METRIC])
    return x_arrays, y_values


def get_mDateGames(games):
    mDateGames = {}
    for game in games:
        datestr = (game['pythonDate'].date())
        if datestr in mDateGames:
            mDateGames[datestr].append(game)
        else:
            mDateGames[datestr] = [game]
    return mDateGames


def toNumpy(x_arrays, y_values, test_x_list, test_y_list):
    return numpy.array(x_arrays), numpy.array(y_values), numpy.array(test_x_list), numpy.array(test_y_list),


#
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


def doML(X, y, test_X, test_y):
    # create model
    model = Sequential()

    # get number of columns in training data
    n_cols = X.shape[1]

    # add model layers
    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(n_cols, activation='relu'))
    model.add(Dense(1))

    # compile model using mse as a measure of model performance
    model.compile(optimizer='adam', loss='mean_squared_error')

    # set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=3)

    # train model
    model.fit(X, y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

    # example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called 'test_X').
    test_y_predictions = model.predict(test_X)


def doEverything(maxSeason=2019):
    games = json.load(open('data/preparedWithNnArrays.json'))
    setDatetimePythonDate(games)
    mDateGames = get_mDateGames(games)

    datePrev = datetime.datetime(2000, 1, 1).date()
    # trainingData = {}
    # trainingData[date] = {'X': X, 'y': y} #was in loop below

    for i, game in enumerate(games):
        if game[C.season] > maxSeason:
            break
        if i < MIN_PREV_GAMES:
            continue
        date = game['pythonDate'].date()

        if date == datePrev:
            continue
        else:
            datePrev = date

        x_arrays, y_values = getTrainingData(i, games, MAX_DAYS_LOOKBACK)

        currentDayGames = mDateGames[date]
        test_x_list, test_y_list = getTestData(currentDayGames)

        X, y, test_X, test_y = toNumpy(x_arrays, y_values, test_x_list, test_y_list)
        print(date, X.shape, y.shape, test_X.shape, test_y.shape)

        # return trainingData


#####################################################################################################################################
# https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python ?
# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37


def getTrainingDataForFutureDate(date, mDateTrainingData):
    mostRecentDateInTrainingDataKeys = date - datetime.timedelta(days=1)
    while mostRecentDateInTrainingDataKeys not in mDateTrainingData.keys():
        mostRecentDateInTrainingDataKeys = mostRecentDateInTrainingDataKeys - datetime.timedelta(days=1)
        if mostRecentDateInTrainingDataKeys.year < 2014:
            return None
    return mDateTrainingData[mostRecentDateInTrainingDataKeys]


#
# def doMLstuff():
#
#     mDateTrainingData = buildAllTrainingData(2015)
#
#     games = json.load(open('data/preparedWithNnArrays.json'))
#     setDatetimePythonDate(games)
#
#     for i, game in enumerate(games):
#         date = game['pythonDate'].date()
#         trainingData = getTrainingDataForFutureDate(date, mDateTrainingData)
#         if trainingData is None:
#             continue
#
#         X = trainingData['X']
#         y = trainingData['y']
#
#         test_X = game[C.x]
#         test_y = game[Y_METRIC]


mDateTrainingData = doEverything(2015)

# d = datetime.date(2014, 10, 10)

# print(d)
# print(d - datetime.timedelta(days=1))
