# problem 7
from itertools import cycle, islice
from collections import defaultdict

class Solution:
    def convert(self, s: str, numRows: int) -> str:
        # [0 1 2 3] + [2 1]
        tmp_idx = [i for i in range(numRows)] + [i for i in range(numRows-2, 0, -1)]

        rrows = list(islice(cycle(tmp_idx), len(s)))

        strdict = defaultdict(list)

        for idx, i in enumerate(rrows):
            strdict[i].append(s[idx])
        
        res = ""
        for r in strdict:
            res += "".join(strdict[r])
        
        return res