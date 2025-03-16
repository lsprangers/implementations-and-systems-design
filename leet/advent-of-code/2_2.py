import os 
path_to_data = os.path.join(os.path.dirname(__name__), 'datum', '2_1.txt')
data = []
with open(path_to_data, 'r') as f:
    data += f.read().splitlines()

game_map = {}
power_of_games = []
for game in data:
    splitz = game.split(':')
    game_idx = int(splitz[0].split(' ')[1])
    game_pull = splitz[1].strip().split(';')
    game_turns = [turn.strip().split(',') for turn in game_pull]
    # red, green, blue
    # 12 red, 13 green, 14 blue
    maxs_of_game = [0, 0, 0]
    game_cats = []
    for turn in game_turns:
        cats = [0, 0, 0]
        for colx in turn:
            splitter = colx.strip().split(' ')
            num_balls = int(splitter[0])
            ball_type = splitter[1]
            if ball_type == 'red':
                cats[0] += num_balls
                if cats[0] > maxs_of_game[0]:
                    maxs_of_game[0] = cats[0]
            elif ball_type == 'green':
                cats[1] += num_balls
                if cats[1] > maxs_of_game[1]:
                    maxs_of_game[1] = cats[1]     
            elif ball_type == 'blue':
                cats[2] += num_balls
                if cats[2] > maxs_of_game[2]:
                    maxs_of_game[2] = cats[2]
        game_cats.append(cats)
    game_map[game_idx] = game_cats
    power_of_games.append(maxs_of_game[0] * maxs_of_game[1] * maxs_of_game[2])

sum(power_of_games)