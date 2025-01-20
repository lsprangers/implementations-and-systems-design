#include <vector>
using namespace std;
// 2342 leetcode 

class Solution {
public:
    int maximumSum(vector<int>& nums) {
        unordered_map<int, int> umap;
        int resp = -1;
        // hash map (sum_of_digits, {nums in that})
        // for each {nums in that} - sort it, and take sum of top 2
        //  update resp for each
        for(int i = 0; i < nums.size(); i++) {
            int summ = 0;
            int n = nums[i];
            do {
                int digit = n % 10;
                summ += digit;
            } while ((n/=10) > 0);

            if(umap.find(summ) != umap.end()) {
                resp = max(resp, nums[i] + umap.at(summ));
            }
            umap[summ] = max(umap[summ], nums[i]);
        }

        return(resp);
    }
};