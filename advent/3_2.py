import os
from collections import defaultdict
path_to_data = os.path.join(os.path.dirname(__name__), 'datum', '3_1.txt')
data = []
with open(path_to_data, 'r') as f:
    data += f.read().splitlines()

# data = """467..114..
# ...*......
# .35.......
# .....10*10
# ......#...
# 617*......
# .....+.58.
# ..592*....
# ......10..
# ......755.
# ...$.*....
# .664.598..""".splitlines()

gear_map = defaultdict(list)
def check_adjacent(num, line_idx, char_idx, num_len):
    if char_idx == 0:
        char_left = set()
    else:
        char_left = set(data[line_idx][char_idx - 1])
        if '*' in list(char_left):
            keyy = str(line_idx) + '_' + str(char_idx - 1)
            gear_map[keyy].append(num)
            return True
    if line_idx == 0:
        chars_below = set()
    else:
        char_max_below = min(len(data[line_idx - 1]), char_idx + num_len + 1)
        char_min_below = max(0, char_idx - 1)
        if char_idx > 0:
            offset = -1
        else:
            offset = 0
        the_chars = data[line_idx - 1][char_min_below : char_max_below]
        chars_below = set(the_chars)
        if '*' in list(chars_below):
            star_idx = the_chars.index('*')
            keyy = str(line_idx - 1) + '_' + str(char_idx + star_idx + offset)
            gear_map[keyy].append(num)
            return True
    if char_idx + num_len == len(data[line_idx]):
        char_right = set()
    else:
        char_right = set(data[line_idx][char_idx + num_len])
        if '*' in list(char_right):
            keyy = str(line_idx) + '_' + str(char_idx + num_len)
            gear_map[keyy].append(num)
            return True
    if line_idx == (len(data) - 1):
        chars_above = set()
    else:
        char_max_above = min(len(data[line_idx + 1]), char_idx + num_len + 1)
        char_min_above = max(0, char_idx - 1)
        if char_idx > 0:
            offset = -1
        else:
            offset = 0
        the_chars = data[line_idx + 1][char_min_above : char_max_above]
        chars_above = set(the_chars)
        if '*' in list(chars_above):
            star_idx = the_chars.index('*')
            keyy = str(line_idx + 1) + '_' + str(char_idx + star_idx + offset)
            gear_map[keyy].append(num)
            return True
    all_chars = chars_below.union(chars_above).union(char_right).union(char_left)
    if len(all_chars) == 1 and list(all_chars)[0] == '.':
        return False
    else:
        return True
    


nums_to_use = []
gear_map = defaultdict(list)
for line_idx, line in enumerate(data):
    in_num_thing = False
    for char_idx, char_here in enumerate(line):
        if char_here.isdigit() and in_num_thing:
            continue
        elif not char_here.isdigit() and in_num_thing:
            in_num_thing = False
        num = ''
        if char_here.isdigit():
            in_num_thing = True
            num_idx_start = char_idx
            while line[num_idx_start].isdigit():
                num += line[num_idx_start]
                num_idx_start += 1
                if num_idx_start == len(line):
                    break
            print(line_idx, num, char_idx, len(num))
            if check_adjacent(num, line_idx, char_idx, len(num)):
                nums_to_use.append(num)


real_nums_to_use = []
for k, v in gear_map.items():
    if len(v) == 2:
        print(k, v)
        real_nums_to_use.append(int(v[0]) * int(v[1]))

sum(real_nums_to_use)
