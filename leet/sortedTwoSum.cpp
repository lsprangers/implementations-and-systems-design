#include <vector>;
using namespace std;

class Solution {
public:
    bool checkTarget(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;

        while (left < right) {
            int curr = nums[left] + nums[right];
            if (curr == target) {
                return true;
            }
            if (curr > target) {
                right--;
            } else {
                left++;
            }
        }
    }
};