import os
import json

def getOddsFromJsonFile(file):
  return json.load(open(file))


def getOddsFromRaw():
    print('sup')

    data = {}

    lines_root = "data-pond-gitignore/nba_lines_from_cody_ssn2014_to_2018"

    for root, dirs, files in os.walk(lines_root):
        files = [os.path.join(root, f) for f in files if f.endswith('.json')]
        for filename in files:
            # print(root, filename)
            # path = os.path.join(root, filename)
            # print(filename)
            fileObj = json.load(open(filename))
            # print(obj.event.eventId)
            event = fileObj['event']
            eventId = event['eventId']
            o = {}
            # o['event'] = event
            date = event['startDate'][1]
            if date['dateType'] != "UTC":
                raise ValueError('dateType not UTC !')
            o['date'] = [date['year'], date['month'], date['date'], date['hour']]
            # away
            team = event['teams'][0]
            o['home_team_statsTeamId'] = team['teamId']
            o['home_team_city'] = team['location']
            o['home_team_nickname'] = team['nickname']
            team = event['teams'][1]
            o['away_team_statsTeamId'] = team['teamId']
            o['away_team_city'] = team['location']
            o['away_team_nickname'] = team['nickname']
            if team['teamLocationType']['name'] != "away":
                raise ValueError('2nd team not away !')
            lines = event['lines'][0]['line']
            # cos favoritePoints is the negative of the amount the favored team is expected to win by
            # but we want spread
            # spread = away - home  expected points

            # init
            line = lines[0]
            if 'total' in line:
                o['line_init_ou'] = line['total']
            else:
                o['line_init_ou'] = None

            if 'favoritePoints' in line:
                if line['favoriteTeamId'] == o['home_team_statsTeamId']:
                    o['line_init_spread'] = line['favoritePoints']
                else:
                    o['line_init_spread'] = -1 * line['favoritePoints']
            else:
                o['line_init_spread'] = None
            # else:
            #     print('missing init lines: ' + filename)

            # final
            line = lines[1]
            if 'total' in line:
                o['line_finl_ou'] = line['total']
            else:
                o['line_finl_ou'] = None

            if 'favoritePoints' in line:
                if line['favoriteTeamId'] == o['home_team_statsTeamId']:
                    o['line_finl_spread'] = line['favoritePoints']
                else:
                    o['line_finl_spread'] = -1 * line['favoritePoints']
            else:
                o['line_finl_spread'] = None

            data[eventId] = o

    with open('lines.json', 'w') as outfile:
        json.dump(obj=data, fp=outfile, sort_keys=True,
                  indent=4, separators=(',', ': '))
    return data
