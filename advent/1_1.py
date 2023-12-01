import os 
path_to_data = os.path.join(os.path.dirname(__name__), 'datum', '1_1.txt')
data = []
with open(path_to_data, 'r') as f:
    data += f.readlines()

good_data = [x.strip('\n') for x in data]
better_data = [''.join(filter(str.isdigit, x)) for x in good_data]
best_data = [int(x[0] + x[-1]) for x in better_data]
