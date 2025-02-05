//1790

class Solution {
public:
    bool areAlmostEqual(string s1, string s2) {
        if(s1.size() != s2.size()) {
            return(false);
        }

        int num_diffs = 0;
        set<char> s1_diff;
        set<char> s2_diff;

        for(int i = 0; i < s1.size(); i++) {
            if(s1[i] != s2[i]) {
                s1_diff.insert(s1[i]);
                s2_diff.insert(s2[i]);
                num_diffs++;
            }
        }

        if(
            (num_diffs == 2 or num_diffs == 0)
            &&
            (s1_diff == s2_diff)
         ) {
            return(true);
        }

        return(false);
    }
};