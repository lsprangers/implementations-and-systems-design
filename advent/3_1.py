import os 
path_to_data = os.path.join(os.path.dirname(__name__), 'datum', '3_1.txt')
data = []
with open(path_to_data, 'r') as f:
    data += f.read().splitlines()

def check_adjacent(line_idx, char_idx, num_len):
    if char_idx == 0:
        char_left = set()
    else:
        char_left = set(data[line_idx][char_idx - 1])
    if line_idx == 0:
        chars_below = set()
    else:
        char_max_below = min(len(data[line_idx - 1]), char_idx + num_len + 1)
        char_min_below = max(0, char_idx - 1)
        chars_below = set(data[line_idx - 1][char_min_below : char_max_below])
    if char_idx + num_len == len(data[line_idx]):
        char_right = set()
    else:
        char_right = set(data[line_idx][char_idx + num_len])
    if line_idx == (len(data) - 1):
        chars_above = set()
    else:
        char_max_above = min(len(data[line_idx + 1]), char_idx + num_len + 1)
        char_min_above = max(0, char_idx - 1)
        chars_above = set(data[line_idx + 1][char_min_above : char_max_above])
    all_chars = chars_below.union(chars_above).union(char_right).union(char_left)
    if len(all_chars) == 1 and list(all_chars)[0] == '.':
        return False
    return True
    


nums_to_use = []
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
            if check_adjacent(line_idx, char_idx, len(num)):
                nums_to_use.append(num)


sum([int(x) for x in nums_to_use])