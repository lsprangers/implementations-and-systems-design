// 3005
// O(2n), but can probably reduce to single pass
// O(n) space
class Solution {
public:
    int maxFrequencyElements(vector<int>& nums) {
        unordered_map<int, int> freq;
        int maxx = 0;
        int resp = 0;
        for(int num : nums) {
            freq[num]++;
            maxx = max(freq.at(num), maxx);
        }
        for(auto [key,val] : freq) {
            if(val == maxx) {
                resp += val;
            }
        }
        return(resp);
    }
};

// This is O(n) time and O(n) space
//  but this was slower than double loop...
class Solution {
public:
    int maxFrequencyElements(vector<int>& nums) {
        unordered_map<int, int> freq;
        int maxx = 0;
        int resp = 0;
        for(int num : nums) {
            freq[num]++;
            int here = freq[num];
            if(here > maxx) {
                maxx = here;
                resp = here;
            } else if(here == maxx) {
                resp += here;
            }
        }

        return(resp);
    }
};