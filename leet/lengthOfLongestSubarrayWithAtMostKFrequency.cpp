//2958
class Solution {
public:
    int maxSubarrayLength(vector<int>& nums, int k) {
        unordered_map<int, int> freq;
        int resp = 0;

        int left = 0;
        for(int right = 0; right < nums.size(); right++) {
            freq[nums[right]]++;

            while(freq[nums[right]] > k) {
                freq[nums[left]]--;
                left++;
            }
            resp = max(resp, right - left + 1);
        }
        return(resp);
    }
};