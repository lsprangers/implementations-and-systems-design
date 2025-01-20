// 3042
class Solution {
public:
    int countPrefixSuffixPairs(vector<string>& words) {
        int resp = 0;
        for(int i = 0; i < words.size() - 1; i++) {
            for(int j = i + 1; j < words.size(); j++) {
                if(isPrefixAndSuffix(words[i], words[j])) {
                    resp++;
                }
            }
        }
        return(resp);
    }

    bool isPrefixAndSuffix(string wordi, string wordj) {
        int iIdx = 0;
        while(iIdx < wordi.size()) {
            if(wordi[iIdx] != wordj[iIdx]) {
                return(false);
            }
            iIdx++;
        }

        iIdx = wordi.size() - 1;
        int jIdx = wordj.size() - 1;
        while(iIdx >= 0) {
            if(wordi[iIdx] != wordj[jIdx]) {
                return(false);
            }
            iIdx--;
            jIdx--;
        }        

        return(true);
    }
};