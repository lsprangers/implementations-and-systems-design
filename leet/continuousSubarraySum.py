# 523
# This is disgusting
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        pfxSum = [nums[0]]
        for num in nums[1:]:
            nextSum = num + pfxSum[-1]
            if nextSum % k == 0:
                return(True)
            pfxSum.append(nextSum)
        
        # wtf is prefixMod...that's insane
        # for a num in pfxSum numi = Qi * k + Ri
        # for any other num numj = Qj * k + Rj
        # Someone explains that if Ri = Rj then those two sums are modulo of k?
        # pfxSum is an array of cumulative sums, so pfxSum[i] - pfxSum[j] is the sum
        #   between those 2 indexes, if  pfxSum[i] - pfxSum[j] % k == 0, then we're set
        # They substituted things in and got (Qi * k + Ri) - ( Qj * k + Rj) % k == 0
        #   which resolves to ((Qi - Qj) * k + (Ri - Rj)) % k == 0
        #   they stripped out (Qi - Qj) * k % k because if something is multiplied by it, it's modulo is 0
        #   and they got to if (Ri - Rj) % k == 0 WHICH finally equated to 
        #   Ri % k == Rj % k
        # Just disgusting
        pfxMod = []
        pfxMap = defaultdict(list)
        for idx, num in enumerate(pfxSum):
            mod = num % k
            pfxMod.append(mod)
            pfxMap[mod].append(idx)
            if len(pfxMap[mod]) > 1:
                if pfxMap[mod][-1] - pfxMap[mod][0] > 1:
                    return(True)

        return(False)