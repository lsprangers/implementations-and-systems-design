#include <vector>;
using namespace std;

class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int left = 0;
        int num_zero = 0;
        int max_array = 0;
        for(int right = 0; right < nums.size(); right++) {
            num_zero += (1-nums[right]);
            
            while(num_zero > k) {
                num_zero -= (1-nums[left]);
                left++;
            }
            max_array = max(max_array, (right-left + 1));
        }
        return(max_array);
    }
};