import os 
path_to_data = os.path.join(os.path.dirname(__name__), 'datum', '2_1.txt')
data = []
with open(path_to_data, 'r') as f:
    data += f.read().splitlines()

game_map = {}
good_game_idxs = []
for game in data:
    splitz = game.split(':')
    game_idx = int(splitz[0].split(' ')[1])
    game_pull = splitz[1].strip().split(';')
    game_turns = [turn.strip().split(',') for turn in game_pull]
    # red, green, blue
    # 12 red, 13 green, 14 blue
    good_game = True
    game_cats = []
    for turn in game_turns:
        cats = [0, 0, 0]
        for colx in turn:
            splitter = colx.strip().split(' ')
            num_balls = int(splitter[0])
            ball_type = splitter[1]
            if ball_type == 'red':
                cats[0] += num_balls
                if cats[0] > 12:
                    good_game = False
            elif ball_type == 'green':
                cats[1] += num_balls
                if cats[1] > 13:
                    good_game = False                
            elif ball_type == 'blue':
                cats[2] += num_balls
                if cats[2] > 14:
                    good_game = False
        game_cats.append(cats)
    game_map[game_idx] = game_cats
    if good_game:
        good_game_idxs.append(game_idx)

sum(good_game_idxs)
    