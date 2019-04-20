import json
import datetime

# now build nn input layer array and save with json file.

# TODO that first need to get one-hot encoding maps for teams and tv stations.

# so get a set of all elements, then convert to array.   array = index --> value dictionary.  also make map: value --> index

data = json.load(open('prepared.json'))

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


def printMaps():
    # visually checking
    for i, value in enumerate(m_i_team):
        print(i, value)
    for i, value in enumerate(m_i_station):
        print(i, value)
    for key in m_team_i.keys():
        print(m_team_i[key], key)
    for key in m_station_i.keys():
        print(m_station_i[key], key)

# printMaps()


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


dataAr = [data[key] for key in data.keys()]


for obj in dataAr:
    obj['pythonDate'] = buildDateForObj(obj)


# now build inpput array in json


dataAr.sort(key=getDateForObj)
# dataAr.sort(key=getDateForObj, reverse=True)

# for obj in dataAr:
#     obj.pop('pythonDate', None)

# with open('sorted.json', 'w') as outfile:
#     json.dump(obj=dataAr, fp=outfile, sort_keys=True,
#               indent=4, separators=(',', ': '))

# exit()

# dataAr.sort(key='pythonDate')

# print(len(dataAr))

# print(dataAr[0])

# exit()

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

  #asdf
  #   


def getDaysSinceNPriorGame(nickname, i, obj, dataAr, n):
    # print()
    # print()
    # print('getDaysSinceNPriorGame', nickname, "i", i, "n", n, '##############################################################')
    # print(obj)

    # output = '\n\n' + 'getDaysSinceNPriorGame' +' ' + nickname+' ' + "i"+' ' + str(i)+' ' + "n"+' ' + str(n)+' ' +'\n'+   str(obj['date'])   + '\n\n' '##############################################################' + '\n'

    results = [5] * n
    k = 0  # prior game; results index
    while i > 0 and k < n:
        i -= 1
        currObj = dataAr[i]
        home = currObj['home_team_nickname']
        away = currObj['away_team_nickname']
        if home == nickname or away == nickname:
            # print()
            # print(currObj['home_team_nickname'])
            # print(currObj['away_team_nickname'])
            # print(currObj['date'])
            # print(currObj['pythonDate'])
            # output += currObj['home_team_nickname']  +'\n'+  currObj['away_team_nickname']   +'\n'+   str(currObj['date'])    +'\n'+  str(currObj['pythonDate'])   +'\n'
            # output += str(i) + ', ' + str(k) + '\n\n'
            days = daysBetween(currObj, obj)
            if k == 0:
                results[k] = min(days, 5)
            else:
                results[k] = min(days - results[k-1], 5)
            k += 1
    # print()
    # print(results)
    # output += '\n' + str(results) + '\n'
    # if 999 in results and nickname == 'Magic':
    #     print(output)
    return results

def buildNnInputArray(i, obj, dataAr):

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


for i, obj in enumerate(dataAr):
    # print(i, obj)
    nnInputArray = buildNnInputArray(i, obj, dataAr)
    obj['y_isOverInit'] = getIsOverInit(obj)
    obj['y_isOverFinal'] = getIsOverFinal(obj)
    obj['y_homeBeatSpreadInit'] = getHomeBeatSpreadInit(obj)
    obj['y_homeBeatSpreadFinal'] = getHomeBeatSpreadFinal(obj)
    obj['x'] = nnInputArray



for obj in dataAr:
    obj.pop('pythonDate', None)


#sort_keys=True,
with open('preparedWithNnArrays.json', 'w') as outfile:
    json.dump(obj=dataAr, fp=outfile, 
              indent=4, separators=(',', ': '))
