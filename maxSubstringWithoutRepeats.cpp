#include <vector>;
#include <iostream>;
using namespace std;
// 3
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> umap;
        int resp = 0;
        int left = 0;
        
        for(int right = 0; right < s.size(); right++) {
            while(umap.find(s[right]) != umap.end()) {
                umap.erase(s[left]);
                left++;
            }
            umap[s[right]] = right;
            resp = max(resp, right - left + 1);
        }
        return(resp);
    }
};