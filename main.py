import numpy as np
import neural_network as nn
import data_api as api

neural_network = nn.NeuralNetwork([4, 10, 4, 1])

matches, results = [], []
for matchday in range(6, 39):
    matchday_matches, matchday_results = api.get_data("PL", "2023", matchday)

    if matchday_matches == None:
        break

    matches += matchday_matches
    results += matchday_results

X = np.array(matches)
Y = np.array(results)

    #print(f"Matchday {matchday}: {home_team.get('name')} ({', '.join(map(str, standings[home_team.get('id')]))}) {score.get('home')} - {score.get('away')} ({', '.join(map(str, standings[away_team.get('id')]))}) {away_team.get('name')}")

neural_network.train(X, Y, 10_000, 0.1)
output = neural_network.predict(np.array([[0.87, 0.4, 2.15, 2.4]]))
print(f"{'Home win' if round(output[0]) == 1 else 'Away win'} with {(output[0]*100 if round(output[0]) == 1 else (1-output[0])*100):.2f}% certainty")