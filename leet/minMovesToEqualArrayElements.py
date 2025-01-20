# problem 453
from typing import List

class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x:x[1])
        countt = 1
        current_end = points[0][1]
        for i, j in points:
            # if intervals are no longer overlapping
            if i > current_end:
                countt += 1     # we will need a new dart
                current_end = j # and there will be a new end to compare all else to
        return countt