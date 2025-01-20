import os 
path_to_data = os.path.join(os.path.dirname(__name__), 'datum', '1_1.txt')
data = []
with open(path_to_data, 'r') as f:
    data += f.readlines()
    datum = f.read()

number_names = {'one' : '1',
                'two' : '2',
                'three' : '3',
                'four' : '4',
                'five' : '5',
                'six' : '6',
                'seven' : '7',
                'eight' : '8',
                'nine' : '9'}
total = 0

for line in data:
    nums = []
    for n in number_names:
        # replace word with number but keep first and last letter of the word in case of overlapping words
        line = line.replace(n, n[0] + number_names.get(n) + n[-1], line.count(n))
    for char in line:
        if char.isnumeric():
            nums.append(char)
    total += int(nums[0] + nums[-1])
print(total)