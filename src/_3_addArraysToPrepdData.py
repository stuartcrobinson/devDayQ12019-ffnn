import json
import datetime
import constants as C

# now build nn input layer arrays and save with json file.
# first need to get one-hot encoding maps for teams and tv stations.
# so get a set of all elements, then convert to array.   array = index --> value dictionary.  also make map: value --> index

# "away_team_city": "Indiana",
# "away_team_nickname": "Pacers",
# "away_team_statsTeamId": 11,
# "away_team_total_score": 104,
# "date": [
#     2017,
#     10,
#     5,
#     0
# ],
# "home_team_city": "Milwaukee",
# "home_team_nickname": "Bucks",
# "home_team_statsTeamId": 15,
# "home_team_total_score": 86,
# "line_finl_ou": 210.0,
# "line_finl_spread": -4.0,
# "line_init_ou": 205.5,
# "line_init_spread": -6.5,
# "tvStations": [
#     "Fox Sports Net Wisconsin",
#     "Fox Sports Midwest - Indiana"
# ]


def getIsOverFinal(obj):
    try:
        return int(obj['away_team_total_score'] + obj['home_team_total_score'] > obj['line_finl_ou'])
    except TypeError:
        return None


def getIsOverInit(obj):
    try:
        return int(obj['away_team_total_score'] + obj['home_team_total_score'] > obj['line_init_ou'])
    except TypeError:
        return None


def getHomeBeatSpreadFinal(obj):
    try:
        return int(obj['away_team_total_score'] - obj['home_team_total_score'] < obj['line_finl_spread'])
    except TypeError:
        return None


def getHomeBeatSpreadInit(obj):
    try:
        return int(obj['away_team_total_score'] - obj['home_team_total_score'] < obj['line_init_spread'])
    except TypeError:
        return None


def printMaps(m_i_team, m_i_station, m_team_i, m_station_i):
    # visually checking
    for i, value in enumerate(m_i_team):
        print(i, value)
    for i, value in enumerate(m_i_station):
        print(i, value)
    for key in m_team_i.keys():
        print(m_team_i[key], key)
    for key in m_station_i.keys():
        print(m_station_i[key], key)


def buildDateForObj(obj):
    dateAr = obj['date']
    return datetime.datetime(dateAr[0], dateAr[1], dateAr[2], dateAr[3])


def getDateForObj(obj):
    return obj['pythonDate']


def daysBetween(obj1, obj2):
    # print(obj1['pythonDate'], obj2['pythonDate'])
    days = abs((obj1['pythonDate'] - obj2['pythonDate']).days)
    # print(days)
    return days


def getDaysSinceNPriorGame(nickname, i, obj, dataAr, n):
    # output = '\n\n' + 'getDaysSinceNPriorGame' +' ' + nickname+' ' + "i"+' ' + str(i)+' ' + "n"+' ' + str(n)+' ' +'\n'+   str(obj['date'])   + '\n\n' '##############################################################' + '\n'
    results = [5] * n
    k = 0  # prior game; results index
    while i > 0 and k < n:
        i -= 1
        currObj = dataAr[i]
        home = currObj['home_team_nickname']
        away = currObj['away_team_nickname']
        if home == nickname or away == nickname:
            # output += currObj['home_team_nickname']  +'\n'+  currObj['away_team_nickname']   +'\n'+   str(currObj['date'])    +'\n'+  str(currObj['pythonDate'])   +'\n'
            # output += str(i) + ', ' + str(k) + '\n\n'
            days = daysBetween(currObj, obj)
            if k == 0:
                results[k] = min(days, 5)
            else:
                results[k] = min(days - results[k-1], 5)
            k += 1
    # output += '\n' + str(results) + '\n'
    # if 999 in results and nickname == 'Magic':
    #     print(output)
    return results


def buildNnInputArray(i, obj, dataAr, m_i_team, m_i_station, m_team_i, m_station_i):

    xTeams = [0] * len(m_i_team)
    xTeamsIsHome = [0] * len(m_i_team)
    xStations = [0] * len(m_i_station)

    away = obj['away_team_nickname']
    home = obj['home_team_nickname']

    away_i = m_team_i[away]
    home_i = m_team_i[home]
    station_i_ar = []
    stationNames = obj['tvStations']
    for stationName in stationNames:
        station_i_ar.append(m_station_i[stationName])

    xTeams[away_i] = 1
    xTeams[home_i] = 1
    xTeamsIsHome[home_i] = 1
    for j in station_i_ar:
        xStations[j] = 1

    homeDaysSincePrevGamesAr = getDaysSinceNPriorGame(home, i, obj, dataAr, 4)
    awayDaysSincePrevGamesAr = getDaysSinceNPriorGame(away, i, obj, dataAr, 4)

    # now for days since prev games .... how to calculate?  need to figure out python dates
    # need to have array of all events sorted by date
    # we should be looping through that now instead of object keys.  so we have current index, and just walk backwards until we find target team

    # todo 1. python sort.  2.  python date.  3.  python num days between dates
    result = xTeams + xTeamsIsHome + xStations + \
        homeDaysSincePrevGamesAr + awayDaysSincePrevGamesAr

    # print(result)

    return result


def getSeason(date):
    year = date[0]
    month = date[1]
    if month > 7:
        return year
    else:
        return year - 1

def main():

    data = json.load(open('data/prepared.json'))

    teams = set()
    stations = set()

    for eventId in data.keys():
        obj = data[eventId]
        teams.add(obj['away_team_nickname'])
        tvStations = obj['tvStations']
        for station in tvStations:
            stations.add(station)

    m_i_team = list(teams)
    m_i_station = list(stations)
    m_team_i = {}
    m_station_i = {}
    for i, x in enumerate(teams):
        m_team_i[x] = i
    for i, x in enumerate(stations):
        m_station_i[x] = i

    dataAr = [data[key] for key in data.keys()]

    for obj in dataAr:
        obj['pythonDate'] = buildDateForObj(obj)

    # now build inpput array in json

    dataAr.sort(key=getDateForObj)

    for i, obj in enumerate(dataAr):
        nnInputArray = buildNnInputArray(i, obj, dataAr, m_i_team, m_i_station, m_team_i, m_station_i)
        obj['season'] = getSeason(obj['date'])
        obj['y_isOverInit'] = getIsOverInit(obj)
        obj['y_isOverFinal'] = getIsOverFinal(obj)
        obj['y_homeBeatSpreadInit'] = getHomeBeatSpreadInit(obj)
        obj['y_homeBeatSpreadFinal'] = getHomeBeatSpreadFinal(obj)
        obj['y_homeDidWin'] = 1 if obj[C.home_team_total_score] > obj[C.away_team_total_score] else 0
        obj['x'] = nnInputArray

    for obj in dataAr:
        obj.pop('pythonDate', None)  # cos not json serializable

    with open('data/preparedWithNnArrays.json', 'w') as outfile:
        json.dump(obj=dataAr, fp=outfile, indent=4, separators=(',', ': '))


main()