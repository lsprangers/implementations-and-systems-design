# 238
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        entire_product = 1
        zeros_included = 0

        for num in nums:
            if num != 0:
                entire_product *= num
            else:
                zeros_included += 1

        resp = []
        for num in nums:
            if num != 0 and zeros_included > 0:
                resp.append(0)

            elif num != 0 and zeros_included == 0:
                resp.append(entire_product // num)
            
            elif num == 0 and zeros_included > 1:
                resp.append(0)
            
            elif num == 0 and zeros_included == 1:
                resp.append(entire_product)
        
        return(resp)