#include <set>;
#include <iostream>;
using namespace std;

class Solution {
public:
    int missingNumber(vector<int>& nums) {
        set<int> dis_nums(nums.begin(), nums.end());
        for(int i = 0; i < nums.size() + 1; i++) {
            if (dis_nums.find(i) == dis_nums.end()) {
                return(i);   
            }
        }
        return(-1);
    }
};