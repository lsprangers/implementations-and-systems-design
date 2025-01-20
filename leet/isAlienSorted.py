# 953
from typing import List
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        map = {}
        for idx, letter in enumerate(order):
            map[letter] = idx
        for idx in range(1, len(words)):
            first = words[idx - 1]
            second = words[idx]
            min_length = min(len(first), len(second))
            is_ok = False
            for j in range(min_length):
                if map[first[j]] < map[second[j]]:
                    is_ok = True
                    break
                elif map[first[j]] > map[second[j]]:
                    return False
            if not is_ok and len(first) > len(second):
                return False
        return True