#include <vector>;
using namespace std;

// can use the fact that initial sum can be calculated
//  and then after that we just remove the leftmost part
//  of window and include rightmost part

// Find largest sum then convert to double and divide by len
class Solution {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        double summ = 0.0;
        for(int s = 0; s < k; s++) {
            summ += nums[s];
        }
        double resp = summ;
        for(int i = 0; i < nums.size() - k; i++) {
            summ += nums[i+k] - nums[i];
            resp = max(resp, summ);
        }
        return((double)resp / (double)k);
    }
};