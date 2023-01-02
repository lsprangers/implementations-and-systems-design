class Solution {
public:
    bool detectCapitalUse(string word) {
        bool allUpper = false;
        bool allLower = false;
        bool camelCase = false;
        if(word.length() < 2){
            return(true);
        }
        if(isupper(word[0])){
            allUpper = true;
            camelCase = true;
            allLower = false;
        }
        else {
            allUpper = false;
            camelCase = false;
            allLower = true;
        }
        for(int i = 1; i < word.length(); i++) {
            if(isupper(word[i])){
                allUpper = allUpper && true;
                allLower = false;
                camelCase = false;
            }
            else {
                allUpper = false;
                camelCase = camelCase && true;
                allLower = allLower && true;
            }
        }
        return (allUpper || allLower || camelCase);
    }
};