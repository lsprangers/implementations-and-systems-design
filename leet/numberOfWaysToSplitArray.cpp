// 2270

class Solution {
public:
    int waysToSplitArray(vector<int>& nums) {
        long long leftCount = 0;
        long long rightCount = 0;
        // some weird int overflow with std::accumulate
        for(int num : nums) {
            rightCount += num;
        }

        long long resp = 0;
        for(int idx = 0; idx < nums.size() - 1; idx++) {
            leftCount += nums[idx];
            rightCount -= nums[idx];
            if(leftCount >= rightCount) {
                resp++;
            }
        }
        return(resp);
    }
};