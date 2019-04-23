import datetime
import json

import random
import numpy

import _3_addArraysToPrepdData as _3
import constants as C

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras import backend
import gc

from tensorflow.python.keras.utils import plot_model

#


# build X for keras

# now load sorted games w/ nn arrays (preppedWithnnArrays)
# iterate through.  with index.
# const:    - MIN_HISTORICAL_GAMES - must b true for home and away
#           - MAX_DAYS_LOOKBACK    - maximum number of prior days to include in X -- 1 year?
#           - RECENCY_BIAS_PARAMETER - 5x for past 2 days, 4x for past week, 3x for past month, 2x for past 2 months, 1x for past MAX_DAYS_LOOKBACK
#           - MIN_PREV_GAMES - irrelevant of teams.  cos models are built per day not per game


MIN_HISTORICAL_GAMES = 5
MIN_PREV_GAMES = 300
MAX_DAYS_LOOKBACK = 365


# Y_METRIC = C.y_isOverFinal
# Y_METRIC = C.y_homeBeatSpreadFinal


# Y_METRIC = C.y_homeDidWin


# todo don't build X per game!!!! build per DAY
# ugh okay how to determine if new day ...


def setDatetimePythonDate(games):
    for game in games:
        game['pythonDate'] = _3.buildDateForObj(game)


# RECENCY_BIAS_PARAMETER - 5x for past 2 days, 4x for past week, 3x for past month, 2x for past 2 months, 1x for past MAX_DAYS_LOOKBACK
def getRecencyBiasFactor(hp, numDaysAgo):
    if numDaysAgo <= 2:
        return hp.recency_bias[0]
    if numDaysAgo <= 7:
        return hp.recency_bias[1]
    if numDaysAgo <= 31:
        return hp.recency_bias[2]
    if numDaysAgo <= 62:
        return hp.recency_bias[3]
    return hp.recency_bias[4]


# print(shuffleTogether([1, 2, 3, 4], [11, 22, 33, 44])) #to check
def shuffleTogether(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def getTrainingData(hp, i, games):
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
        if numDaysAgo > hp.max_days_lookback:
            break
        # okay so gameIter is a valid game, so add it to X
        if gameIter[C.y_isOverFinal] is not None:
            for _ in range(getRecencyBiasFactor(hp, numDaysAgo)):
                x_arrays.append(gameIter[C.x])
                y_values.append(gameIter[hp.y_metric])
    # shuffle doesnt take much extra time.  important so validation data selected randomly as opposed to most recent (MOST IMPORTANT games)
    x_arrays, y_values = shuffleTogether(x_arrays, y_values)
    return x_arrays, y_values


def getTrainingDataSingleModel(hp, i, games):
    game = games[i]
    x_arrays = []
    y_values = []
    k = i
    prevNumDaysAgo = -1
    while k >= 0:
        k -= 1
        gameIter = games[k]
        numDaysAgo = _3.daysBetween(game, gameIter)
        if numDaysAgo < 1:
            continue
        # if prevNumDaysAgo != -1 and prevNumDaysAgo != numDaysAgo:
        #     break
        if numDaysAgo > hp.max_days_lookback:
            break
        # okay so gameIter is on a diff day from i's game, but same day as other games gotten so far.
        if gameIter[hp.y_metric] is not None:
            for _ in range(getRecencyBiasFactor(hp, numDaysAgo)):
                x_arrays.append(gameIter[C.x])
                # x_arrays.append(gameIter[C.x][30:60] + gameIter[C.x][0:30])
                y_values.append(gameIter[hp.y_metric])
                # y_values.append(0 if gameIter[hp.y_metric] == 1 else 1)
        prevNumDaysAgo = numDaysAgo
    return x_arrays, y_values


def getTestData(hp, games):
    x_arrays = []
    y_values = []
    for game in games:
        x_arrays.append(game[C.x])
        # x_arrays.append(game[C.x][30:60] + game[C.x][0:30])
        y_values.append(game[hp.y_metric])
        # y_values.append(0 if game[hp.y_metric] == 1 else 1)
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


# import numpy
import sys

numpy.set_printoptions(threshold=sys.maxsize)


# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37
def doML(hp, model, X, y, test_X, test_y):
    # https://stackoverflow.com/questions/50331201/memory-leak-keras-tensorflow1-8-0
    # # doesn't help :(
    # backend.clear_session()
    #
    # # create model
    # model = Sequential()
    #
    # # get number of columns in training data
    # n_cols = X.shape[1]
    #
    # if hp.dropout == 0:
    #     model.add(Dense(hp.dim, activation='relu', input_shape=(n_cols,)))
    # else:
    #     model.add(Dropout(hp.dropout, input_shape=(n_cols,)))
    #
    # for _ in range(hp.layers):
    #     model.add(Dense(hp.dim, activation='relu'))
    #
    # model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #
    # # compile model using mse as a measure of model performance
    # model.compile(optimizer='adam', loss='binary_crossentropy')  # mean_squared_error #binary_crossentropy
    #
    # # set early stopping monitor so the model stops training when it won't improve anymore
    # # early_stopping_monitor = EarlyStopping(patience=0)
    early_stopping_monitor = EarlyStopping(monitor='loss', patience=hp.patience)

    # train model
    # model.fit(X, y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
    model.fit(X, y, batch_size=2000, validation_split=hp.validation_split, epochs=hp.epochs,
              callbacks=[early_stopping_monitor])
    #
    # print(model.weights[0])
    # print(model.weights[0][0])
    # print(model.weights)
    # print('wtf')
    # print(model.get_weights())
    # model.reset_metrics()

    # example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called 'test_X').
    test_y_predictions = model.predict(test_X)

    # plot_model(model, to_file='model.png')

    # del model
    # gc.collect()
    print(test_y_predictions, test_y)
    return test_y_predictions, test_y, model


def getResultsStr(allYExp, allYActual):
    if len(allYExp) != len(allYActual):
        raise ValueError("terrible")
    count = 0
    correct = 0
    maxCutoff = 0.55
    minCutoff = 0.45
    for i in range(len(allYActual)):
        exp = allYExp[i]
        act = allYActual[i]
        isOne = exp >= maxCutoff
        isZero = exp < minCutoff
        if isOne:
            if act == 1:
                correct += 1
            count += 1
        if isZero:
            if act == 0:
                correct += 1
            count += 1
        # print("{:.3f}".format(exp), act, count, correct)
    if count > 0:
        return "{0:.1f}%  n: {1}  {2}".format(100 * correct / count, count, correct)
    else:
        return "na"


def removeInvalidGames(games, yMetric):
    return [x for x in games if x[yMetric] is not None]


def useOnlyTeams(games):
    for game in games:
        game[C.x] = game[C.x][0:60]
    return games


def dropStations(games):
    for game in games:
        game[C.x] = game[C.x][0:60] + game[C.x][120:]
    return games


def dropRecency(games):
    for game in games:
        game[C.x] = game[C.x][0:120]
    return games


def keepOnlyFirst30Elements(games):
    for game in games:
        game[C.x] = game[C.x][0:30]
    return games


def makeModel(hp, games):
    backend.clear_session()

    # create model
    model = Sequential()

    # get number of columns in training data
    n_cols = len(games[0][C.x])
    print("ncols:", n_cols)

    if hp.dropout > 0:
        model.add(Dropout(hp.dropout, input_shape=(n_cols,)))

    # if hp.dropout == 0:
    #     model.add(Dense(hp.dim, activation='relu', input_shape=(n_cols,)))
    # else:
    #     model.add(Dropout(hp.dropout, input_shape=(n_cols,)))

    if hp.layers > 0:
        if hp.dropout > 0:
            model.add(Dense(hp.dim, activation='relu'))
        else:
            model.add(Dense(hp.dim, activation='relu', input_shape=(n_cols,)))

        for _ in range(hp.layers - 1):
            model.add(Dense(hp.dim, activation='relu'))

    if hp.layers == 0 and hp.dropout == 0:
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid', input_shape=(n_cols,)))
    else:
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # compile model using mse as a measure of model performance
    model.compile(optimizer='adam', loss='binary_crossentropy')  # mean_squared_error #binary_crossentropy

    # set early stopping monitor so the model stops training when it won't improve anymore
    # early_stopping_monitor = EarlyStopping(patience=0)
    # early_stopping_monitor = EarlyStopping(monitor='loss', patience=hp.patience)
    return model


def doEverything(hp, minSeason=2015, maxSeason=2019):
    games = json.load(open('data/preparedWithNnArrays.json'))
    setDatetimePythonDate(games)
    games = removeInvalidGames(games, hp.y_metric)
    if hp.dropRecencyAndStations:
        games = useOnlyTeams(games)
    elif hp.dropStations:
        games = dropStations(games)
    elif hp.dropRecency:
        games = dropRecency(games)
    elif hp.keepOnlyFirst30Elements:
        games = keepOnlyFirst30Elements(games)

    # exit()
    mDateGames = get_mDateGames(games)

    datePrev = datetime.datetime(2000, 1, 1).date()

    allYExp = []  # numpy.zeros(0)
    allYActual = []  # numpy.zeros(0)

    model = makeModel(hp, games)

    for i, game in enumerate(games):
        # print(game[C.date])
        # print(game[C.x])
        if game[C.season] > maxSeason:
            break
        if game[C.season] < minSeason:
            continue
        if i < hp.min_prev_games:
            continue
        date = game['pythonDate'].date()
        # if date.month > 5: #!= 3:  # and date.month != 3:
        #     continue
        # if date.month != 3:  # or date.day > 1:  # and date.month != 3:
        #     continue

        if date == datePrev:
            continue
        else:
            datePrev = date

        # x_arrays, y_values = getTrainingData(hp, i, games)
        x_arrays, y_values = getTrainingDataSingleModel(hp, i, games)

        if len(x_arrays) == 0:
            continue
        currentDayGames = mDateGames[date]
        test_x_list, test_y_list = getTestData(hp, currentDayGames)

        X, y, test_X, test_y = toNumpy(x_arrays, y_values, test_x_list, test_y_list)

        print(date, X.shape, y.shape, test_X.shape, test_y.shape)

        y_exp, y_act, model = doML(hp, model, X, y, test_X, test_y)
        allYExp += y_exp[:, 0].tolist()
        allYActual += y_act.tolist()
        print('------------------------------------------')
        print(getResultsStr(allYExp, allYActual))
        print()

    # print(model.weights)
    with open("weights.txt", "w") as text_file:
        text_file.write(str(model.weights))

    print("      " + str(hp.toJson()))

    resultstr = getResultsStr(allYExp, allYActual)

    with open("results.txt", "a") as myfile:
        myfile.write("      " + str(hp.toJson()) + '\n')
        myfile.write(resultstr + '\n')


#####################################################################################################################################
# https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python ?

"""

HYPERPARAMETERS

dropout: random float between 0 and 1
hidden layer dimension: random integer between 1 and n_cols*5
num hidden layers: random integer between 1 and 4
RECENCY_BIAS_PARAMETER - 5 values.  random ints between 1 and 15
patience - random int from 0 to 5
include tv stations = boolean
validation_split = float between 0 and 0.5


"""


class Hyperparameters:
    max_days_lookback: 'max_days_lookback'
    min_prev_games: 'min_prev_games'
    dropout: 'dropout'
    dim: 'dim'
    layers: 'layers'
    recency_bias: 'recency_bias'
    patience: 'patience'
    validation_split: 'validation_split'
    y_metric: 'y_metric'
    epochs: 'epochs'
    dropStations: False
    dropRecency: False
    dropRecencyAndStations: False
    keepOnlyFirst30Elements: False

    def toJson(self):
        return {
            'max_days_lookback': self.max_days_lookback,
            'min_prev_games': self.min_prev_games,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'dim': self.dim,
            'layers': self.layers,
            'recency_bias': self.recency_bias,
            'patience': self.patience,
            'validation_split': self.validation_split,
            'y_metric': self.y_metric,
            'dropStations': self.dropStations,
            'dropRecency': self.dropRecency,
            'dropRecencyAndStations': self.dropRecencyAndStations,
            'keepOnlyFirst30Elements': self.keepOnlyFirst30Elements,
        }


hp = Hyperparameters()

hp.max_days_lookback = 365
hp.min_prev_games = 300
hp.dropout = 0.1
hp.dim = 60
hp.layers = 1
hp.recency_bias = [5, 4, 3, 2, 1]
hp.patience = 0
hp.epochs = 30
hp.validation_split = 0
hp.y_metric = C.y_homeDidWin
hp.dropStations = False
hp.dropRecency = False
hp.dropRecencyAndStations = True
hp.keepOnlyFirst30Elements = False

print(hp.toJson())

hp.dropout = 0.3
hp.dim = 30
hp.recency_bias = [10, 9, 8, 1, 1]
hp.layers = 0
hp.max_days_lookback = 300
hp.epochs = 30

# hp.recency_bias = [8, 8, 1, 1, 1]
# hp.max_days_lookback = 17
# hp.epochs = 6
#
hp.dropout = 0.1

# hp.layers = 0
# doEverything(hp, 2015, 2018)

hp.layers = 1
doEverything(hp, 2015, 2017)
#
# hp.dropStations = False
# hp.dropRecency = False
# hp.dropRecencyAndStations = False
# doEverything(hp, 2015, 2015)
#
#
# hp.dim = 100
#
# hp.dropStations = True
# hp.dropRecency = False
# hp.dropRecencyAndStations = False
# doEverything(hp, 2015, 2015)
#
# hp.dropStations = False
# hp.dropRecency = False
# hp.dropRecencyAndStations = False
# doEverything(hp, 2015, 2015)

# doEverything(hp, 2015, 2015)

# hp.max_days_lookback = 100
# doEverything(hp, 2015, 2015)
#
# hp.max_days_lookback = 50
# doEverything(hp, 2015, 2015)

# hp.max_days_lookback = 300
# hp.dropout = 0
# doEverything(hp, 2015, 2015)

# hp.dropout = 0.2
# doEverything(hp, 2015, 2015)
#
# hp.dropout = 0.4
# doEverything(hp, 2015, 2015)
#
# hp.dropout = 0.1
# hp.dim = 120
# doEverything(hp, 2015, 2015)

# still running: next 3:
# hp.dim = 30
# doEverything(hp, 2015, 2015)
#
# hp.dim = 60
# hp.layers = 2
# doEverything(hp, 2015, 2015)
#
# hp.layers = 3
# doEverything(hp, 2015, 2015)

# hp.layers = 4
# doEverything(hp, 2015, 2015)

# hp.layers = 1
# doEverything(hp, 2015, 2015)

# hp.recency_bias = [1, 1, 1, 1, 1]
# doEverything(hp, 2015, 2015)

# hp.layers = 3
# hp.recency_bias = [8, 4, 3, 2, 1]
# doEverything(hp, 2015, 2015)
#
# hp.recency_bias = [8, 7, 2, 2, 1]
# doEverything(hp, 2015, 2015)
#
# hp.recency_bias = [5, 4, 3, 2, 1]
# hp.patience = 2
# doEverything(hp, 2015, 2015)
#
# hp.patience = 4
# doEverything(hp, 2015, 2015)
#
# hp.patience = 6
# doEverything(hp, 2015, 2015)

# 2016-02-28 (1147, 60) (1147,) (7, 60) (7,)
# libc++abi.dylib: terminating with uncaught exception of type std::__1::system_error: thread constructor failed: Resource temporarily unavailable
# fish: 'python3 src/main.py' terminated by signal SIGABRT (Abort)
# stuartrobinson@Stuarts-MacBook-Pro ~/r/devDayQ12019-ffnn>
