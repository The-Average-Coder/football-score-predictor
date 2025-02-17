import requests

with open("api_keys", "r") as file:
    data = file.read()

API_KEYS = [[key, 0] for key in data.split("\n")]

current_api_key = 0

API_URL = "https://api.football-data.org/v4/competitions/"
MATCHES_URL = "https://api.football-data.org/v4/competitions/PL/matches?season=2023"
STANDINGS_URL = "https://api.football-data.org/v4/competitions/PL/standings?season=2023"

def make_request(url):
    global current_api_key

    if API_KEYS[current_api_key][1] >= 10:
        current_api_key += 1

        if current_api_key >= len(API_KEYS):
            print("ERROR: Ran out of API keys")
            return

    headers = {
        'X-Auth-Token': API_KEYS[current_api_key][0]
    }

    response = requests.get(url, headers=headers)
    API_KEYS[current_api_key][1] += 1

    if response.status_code != 200:
        print(f"Error code {response.status_code} in getting data")
    data = response.json()

    return data

def get_data(competition, season, matchday):
    if matchday < 6 or matchday > 38:
        print("ERROR: Invalid matchday")
        return None, None

    matches_data = make_request(API_URL + f"{competition}/matches?season={season}&matchday={matchday}")
    if matches_data == None:
        return None, None
    matches_data = matches_data.get('matches')
    
    standings_data = make_request(API_URL + f"{competition}/standings?season={season}&matchday={matchday-1}").get('standings')[0]
    if standings_data == None:
        return None, None
    standings_data = standings_data.get('table')

    standings = {}
    for standing in standings_data:
        team_id = standing.get('team').get('id')
        team_points = standing.get('points')
        team_form_string = standing.get('form')
        team_form = team_form_string.count('W') * 3 + team_form_string.count('D')
        
        standings[team_id] = [team_points / (matchday-1), team_form / 5]

    matches, results = [], []

    for match in matches_data:
        home_team_id = match.get('homeTeam').get('id')
        away_team_id = match.get('awayTeam').get('id')
        result = match.get('score').get('winner')

        result_value = {'HOME_TEAM' : 1, 'AWAY_TEAM': 0, 'DRAW': 0.5}

        matches.append(standings[home_team_id] + standings[away_team_id])
        results.append(result_value[result])
    
    return matches, results