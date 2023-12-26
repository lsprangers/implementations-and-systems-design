#include <vector>;
using namespace std;

class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int n = nums.size();
        vector<int> newNums(n);
        int left = 0;
        int right = n - 1;
        for(int i = n - 1; i>=0; i--) {
            int res;
            if(abs(nums[left]) < abs(nums[right])) {
                res = nums[right];
                right--;
            }
            else {
                res = nums[left];
                left++;
            }
            newNums[i] = res * res;
        }
        return(newNums);
    }
};