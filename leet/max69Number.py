class Solution:
    def maximum69Number (self, num: int) -> int:
        numList = [str(x) for x in str(num)]
        for idx, numChar in enumerate(numList):
            if(numChar == "6"):
                numList[idx] = "9"
                break
        return(int("".join(numList)))