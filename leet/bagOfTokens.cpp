#include <vector>;
using namespace std;
//948
class Solution {
public:
    int bagOfTokensScore(vector<int>& tokens, int power) {
        sort(tokens.begin(), tokens.end());
        int score = 0;
        int low = 0;
        int high = tokens.size() - 1;

        while(low <= high) {
            int token = 0;
            if(tokens[low] <= power) {
                // face up
                token = tokens[low];
                low++;
                power -= token;
                score++;
                cout << "low " << token << " " << low << " " << power << " " << score << endl;
            }
            else {
                if(score == 0) {
                    cout << "zero";
                    return(0);
                }
                // No need to play this token
                if(low == high) { break; }
                
                // face down
                token = tokens[high];
                high--;
                score--;
                power += token;
                cout << "high " << token << " " << high << " " << power << " " << score << endl;
            }
        }
        return(score);        
    }
};