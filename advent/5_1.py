import os 
from collections import defaultdict
path_to_data = os.path.join(os.path.dirname(__name__), 'datum', '5_1.txt')
data = []
with open(path_to_data, 'r') as f:
    data += f.read().splitlines()

seeds = [int(x) for x in data[0].split(': ')[1].split(' ')]
seed2soil = defaultdict(list)

seed_to_soil_list = data[3:49]
for idx, s2s in enumerate(seed_to_soil_list):
    dest_range, start_range, range_len = (int(x) for x in s2s.split(' '))
    for i in range(range_len + 1):
        seed2soil[start_range + i] = dest_range + i