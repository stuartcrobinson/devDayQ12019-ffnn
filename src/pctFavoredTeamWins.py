import json
import constants as C

games = json.load(open('data/preparedWithNnArrays.json'))

count = 0
success = 0
for game in games:
    if game[C.line_finl_spread] is not None:
        count += 1
        if game[C.home_team_total_score] > game[C.away_team_total_score] and game[C.line_finl_spread] < 0:
            success += 1
        if game[C.away_team_total_score] > game[C.home_team_total_score] and game[C.line_finl_spread] > 0:
            success += 1

print(count, success, success / count)
