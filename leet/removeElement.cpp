// 27
#include <vector>;
#include <set>;
// #include <queue>;
using namespace std;

class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        // doesn't need to be ordered
        unordered_map<int, int> frequency;
        int k = 0;

        // O(n) time and space
        for(int num : nums) {
            if(num == val) {
                // ignore
            }
            else {
                frequency[num]++;
                k++;
            }
        }

        // O(n)
        int idx = 0;
        for(auto [key, val] : frequency) {
            for(int i = 0; i < val; i++) {
                nums[idx] = key;
                idx++;
            }
        }
        return(k);
    }
};