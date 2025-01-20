# problem 520
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if all(x.isupper() for x in word) or all(x.islower() for x in word) or (all(x.islower() for x in word[1:]) and word[0].isupper()):
            return True
        return False