// 350
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> freq;
        //O(n) * O(1)
        for(int num : nums1) {
            freq[num]++;
        }

        // O(n)
        vector<int> resp;
        for(int num : nums2) {
            if(freq[num] > 0) {
                freq[num]--;
                resp.push_back(num);
            }
        }

        // O(n) time, O(n) space
        return(resp);
    }
};