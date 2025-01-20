def process_input(filename):
    with open(filename) as file:
        input = file.read().splitlines()
    grid = {}
    part_grid = {}      # a grid that points to which element of parts a digit belongs to
    parts = []
    part_index = 0
    for y, line in enumerate(input):
        digit = ''
        for x, ch in enumerate(line):
            grid[x,y] = ch
            if ch.isdigit():
                if digit == '':
                    digit_pos = (x, y)
                digit += ch
                part_grid[x,y] = part_index
            else:
                if digit != '':
                    parts.append((digit_pos, digit))
                    part_index += 1
                digit = ''
        if digit != '':
            parts.append((digit_pos, digit))
            part_index += 1
    max_x = len(line)
    max_y = len(input)
    return grid, max_x, max_y, parts, part_grid


def find_gears():
    gear_ratio_sum = 0
    for gx, gy in schematic.keys():
        if schematic[gx,gy] == '*':
            adj_parts = set()
            for y in range(gy-1, gy+2):
                for x in range(gx-1, gx+2):
                    if (x,y) in part_grid:
                        adj_parts.add(part_grid[x,y])
            if len(adj_parts) == 2:
                gear_ratio = 1
                for adj_part in adj_parts:
                    part_pos, part_num = parts[adj_part]
                    gear_ratio *= int(part_num)
                gear_ratio_sum += gear_ratio    
    return gear_ratio_sum


#-----------------------------------------------------------------------------------------

filename = './datum/3_1.txt'

schematic, max_x, max_y, parts, part_grid = process_input(filename)

gear_ratio_sum  = find_gears()

print('Sum of gear ratios is', gear_ratio_sum)