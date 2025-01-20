class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        pArr = list(pattern)
        sArr = s.split(' ')
        if len(pArr) < 1 or len(pArr) != len(sArr):
            return False
        matcher = {}
        for idx, charr in enumerate(pArr):
            if charr in matcher:
                if matcher[charr] != sArr[idx]:
                    return False
            else:
                if sArr[idx] in matcher.values():
                    return False
                else:
                    matcher[charr] = sArr[idx]
        return True