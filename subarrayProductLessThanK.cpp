// 713
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        // Length of a valid subarray is the number
        //  of valid subarrays inside of it
        int left = 0;
        int resp = 0;
        int windowProduct = 1;

        for(int right = 0; right < nums.size(); right++) {
            windowProduct *= nums[right];

            while(windowProduct >= k && left < right) {
                windowProduct /= nums[left];
                left++;
            }

            if(windowProduct < k) {
                // number of valid subarrays in this subarray, + 1 b/c of indexes
                // i.e. if [0,1] is valid, that's 2 numbers, not 1-0 = 1
                resp += right - left + 1;
            }            
        }
        return(resp);
    }
};