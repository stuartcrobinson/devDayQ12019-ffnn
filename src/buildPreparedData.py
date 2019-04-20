import os
import json
import getOddsObj

# purpose of this file is to add scores data to existing odds file and save the updated file as prepared.json


def gte(str, num):
    # print(str)
    try:
        return int(str) >= num
    except ValueError:
        return False


# print(gte('2013', 2014))


def get_pbp_files(minSeason):
    filesar = []

    for root, dirs, files in os.walk(pbp_root):
        dirs = [os.path.join(root, dir) for dir in dirs if gte(dir, minSeason)]
        print(dirs)
        for dir in dirs:
            for root2, dirs2, files2 in os.walk(dir):
                files2 = [os.path.join(root2, f)
                          for f in files2 if f.endswith('.json')]
                for filename in files2:
                    # print(filename)
                    filesar.append(filename)
    return filesar


def checkDates(event, odds):
    date = event['startDate'][1]
    if date['dateType'] != "UTC":
        raise ValueError('dateType not UTC !')
    pbpDate = [date['year'], date['month'], date['date'], date['hour']]
    if odds['date'] != pbpDate:
        raise ValueError('date mismatch !')


def checkTeams(event, odds):
    homeObj = event['teams'][0]
    awayObj = event['teams'][1]
    if homeObj['teamId'] != odds['home_team_statsTeamId']:
        raise ValueError('home id mismatch !')
    if homeObj['nickname'] != odds['home_team_nickname']:
        raise ValueError('home id mismatch !')
    if homeObj['location'] != odds['home_team_city']:
        raise ValueError('home id mismatch !')

    if awayObj['teamId'] != odds['away_team_statsTeamId']:
        raise ValueError('away id mismatch !')
    if awayObj['nickname'] != odds['away_team_nickname']:
        raise ValueError('away id mismatch !')
    if awayObj['location'] != odds['away_team_city']:
        raise ValueError('away id mismatch !')

    if homeObj['score'] != odds['home_team_total_score']:
        raise ValueError('home score mismatch !')
    if awayObj['score'] != odds['away_team_total_score']:
        raise ValueError('away score mismatch !')


def buildPreparedData():

    oddsData = getOddsObj.getOddsFromJsonFile('lines.json')

    pbp_root = "data-pond-gitignore/dandata_basketball/nba_1"

    data = {}

    keys = oddsData.keys()

    for file in get_pbp_files(2014):
        print(file)
        pbpFileObj = json.load(open(file))
        event = pbpFileObj['event']
        eventId = event['eventId']

        if str(eventId) not in keys:
            continue
        if not event['isDataConfirmed']['score']:
            continue
        if "All-" in event['teams'][0]['nickname']:
            continue

        odds = oddsData[str(eventId)]

        homeObj = event['teams'][0]
        awayObj = event['teams'][1]

        odds['home_team_total_score'] = event['teams'][0]['score']
        odds['away_team_total_score'] = event['teams'][1]['score']

        checkDates(event, odds)
        checkTeams(event, odds)

        tvStations = event['tvStations']

        stationNames = []

        for station in tvStations:
            stationNames.append(station['name'])

        odds['tvStations'] = stationNames

        data[eventId] = odds

    with open('prepared.json', 'w') as outfile:
        json.dump(obj=data, fp=outfile, sort_keys=True,
                  indent=4, separators=(',', ': '))
