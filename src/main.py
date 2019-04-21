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


def incrementHistoricalCount(numTeamPastGames, nickname):
    if nickname in numTeamPastGames:
        numTeamPastGames[nickname] += 1
    else:
        numTeamPastGames[nickname] = 1


#todo don't build X per game!!!! build per DAY
# ugh okay how to determine if new day ...

def main():
    games = json.load(open('data/preparedWithNnArrays.json'))

    numTeamPastGames = {}

    for i, game in enumerate(games):
        # print(i, obj)
        print('hi')
        game['pythonDate'] = _3.buildDateForObj(game)
        away = game[C.away_team_nickname]
        home = game[C.home_team_nickname]
        incrementHistoricalCount(numTeamPastGames, home)
        incrementHistoricalCount(numTeamPastGames, away)
        if numTeamPastGames[home] < MIN_HISTORICAL_GAMES or numTeamPastGames[away] < MIN_HISTORICAL_GAMES:
            continue

        x_arrays = []
        y_values = []

        date = game['pythonDate'].date()


        k = i
        while k >= 0 and _3.daysBetween(game, games[k]) < MAX_DAYS_LOOKBACK:
            k -= 1
            gameIter = games[k]
            if _3.daysBetween(game, gameIter) < 1:
                continue
            # okay so gameIter is a valid game, so add it to X
            if gameIter[C.y_isOverFinal] is not None:
                x_arrays.append(gameIter[C.x])
                y_values.append(gameIter[C.y_isOverFinal])

        X = numpy.array(x_arrays)
        y = numpy.array(y_values)
        # print(game)
        # print(X)
        # print(y)
        print(game[C.date])

# main()


d1 = datetime.datetime(2015, 10, 31, 10)
d2 = datetime.datetime(2015, 10, 30, 11)

print(d1)
print(d2)
print(d1.day)
print(d1.date())
print((d1 - d2).days)
print((d1.date() - d2.date()).days)