# Problem 2244
from collections import Counter
from typing import List
from math import ceil 

class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        frequency = Counter(tasks)
        res = 0
        for freq in frequency.values():
            if freq == 1:
                return -1
            res += ceil(freq/3)
        return res