//3295

class Solution {
public:
    bool reportSpam(vector<string>& message, vector<string>& bannedWords) {

        // O(banned)
        unordered_set<string> bannedSet(bannedWords.begin(), bannedWords.end());
        

        // O(message * 1)
        int bannedCount = 0;
        for(string wrd : message) {
            if(bannedSet.find(wrd) != bannedSet.end()) {
                bannedCount++;
            }
            if(bannedCount == 2) {
                return(true);
            }
        }

        // O(banned + message)
        return(false);
    }
};