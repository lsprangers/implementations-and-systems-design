// 916
class Solution {
public:
    unordered_map<string, unordered_map<char, int>> worda_count;
    unordered_map<string, unordered_map<char, int>> wordb_count_tmp;
    unordered_map<char, int> wordb_count_max;

    vector<string> wordSubsets(vector<string>& words1, vector<string>& words2) {
        for(string worda : words1) {
            for(char c : worda) {
                worda_count[worda][c]++;
            }
        }
        for(string wordb : words2) {
            // get letters in tmp
            if(wordb_count_tmp.find(wordb) == wordb_count_tmp.end()) {
                for(char c : wordb) {
                    wordb_count_tmp[wordb][c]++;
                }
            }

            // find max
            for(char c : wordb) {
                wordb_count_max[c] = max(wordb_count_max[c], wordb_count_tmp[wordb][c]);
            }
        }


        vector<string> resp;
        for(string worda : words1) {
            if(isUniversal(worda)) {
                resp.push_back(worda);
            }
        }
        return(resp);
    }

    bool isUniversal(string worda) {
            for (const auto& pair : wordb_count_max) {
                if(pair.second > worda_count[worda][pair.first]) {
                    return(false);
                }
            }
            return(true);
    }

    // bool isSubset(string worda, string wordb) {
    //     for(char c : wordb) {
    //         if(wordb_count[wordb][c] > worda_count[worda][c]) {
    //             return(false);
    //         }
    //     }
    //     return(true);
    // }
};