import numpy
import constants as C
import json
import _3_addArraysToPrepdData as _3
import datetime

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


# todo don't build X per game!!!! build per DAY
# ugh okay how to determine if new day ...


def setDatetimePythonDate(games):
    for game in games:
        game['pythonDate'] = _3.buildDateForObj(game)


def asdf():
    return 1, 2


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
                y_values.append(gameIter[C.y_isOverFinal])
    return x_arrays, y_values


def buildAllTrainingData(maxSeason=2019):
    games = json.load(open('data/preparedWithNnArrays.json'))
    setDatetimePythonDate(games)

    datePrev = datetime.datetime(2000, 1, 1).date()
    trainingData = {}

    for i, game in enumerate(games):
        if game[C.season] > maxSeason:
            break
        if i < MIN_PREV_GAMES:
            continue
        date = game['pythonDate'].date()
        # print(date)

        # okay now loop back until it's a previous day and then start adding arrays to x_arrays etc

        if date == datePrev:
            continue
        else:
            datePrev = date

        x_arrays, y_values = getTrainingData(i, games, MAX_DAYS_LOOKBACK)

        X = numpy.array(x_arrays)
        y = numpy.array(y_values)
        print(date, X.shape, y.shape)
        trainingData[date] = {'X': X, 'y': y}
    return trainingData


#####################################################################################################################################
# https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python ?

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

mDateTrainingData = buildAllTrainingData(2015)

games = json.load(open('data/preparedWithNnArrays.json'))
setDatetimePythonDate(games)


def getTrainingDataForFutureDate(date, mDateTrainingData):
    mostRecentDateInTrainingDataKeys = date - datetime.timedelta(days=1)
    while mostRecentDateInTrainingDataKeys not in mDateTrainingData.keys():
        mostRecentDateInTrainingDataKeys = mostRecentDateInTrainingDataKeys - datetime.timedelta(days=1)
        if mostRecentDateInTrainingDataKeys.year < 2014:
            return None
    return mDateTrainingData[mostRecentDateInTrainingDataKeys]


for i, game in enumerate(games):
    date = game['pythonDate'].date()
    trainingData = getTrainingDataForFutureDate(date, mDateTrainingData)
    if trainingData is None:
        continue

    X = trainingData['X']
    y = trainingData['y']

    # create model
    model = Sequential()

    # get number of columns in training data
    n_cols = X.shape[1]

    # add model layers
    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(n_cols, activation='relu'))
    model.add(Dense(1))

    # set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=3)

    #train model
    model.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

    
d = datetime.date(2014, 10, 10)

# print(d)
# print(d - datetime.timedelta(days=1))
