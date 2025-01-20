#include <vector>;
using namespace std;
//1283

class Solution {
public:
    int smallestDivisor(vector<int>& nums, int threshold) {
        int left = 1;
        int right = 1e6;

        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(checkDivisor(mid, nums) <= threshold) {
                right = mid - 1;
            }
            else {
                left = mid + 1;
            }
        }
        // return min, so left
        return(left);
    }
    int checkDivisor(int divisor, vector<int> nums) {
        int resp = 0;
        for(int num : nums) {
            resp += ceil((double)num / (double)divisor); //should be ceiling
        }
        return(resp);
    }
};