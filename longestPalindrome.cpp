// 409
class Solution {
public:
    int longestPalindrome(string s) {
        unordered_map<char, int> umap;
        for(char c : s) {
            umap[c]++;
        }
        

        int resp = 0;
        bool hasOdd = false;
        for(auto [key, val] : umap) {
            if(val >= 2) {
                // take even number from each value to put on ends
                resp += (val % 2 == 0 ? val : val - 1) ;
            }
            if(val % 2 == 1) {
                hasOdd = true;
            }
        }

        // put an odd in the middle if so
        resp += (hasOdd ? 1 : 0);
        return(resp);
    }
};