#include <set>;
#include <iostream>;
using namespace std;

class Solution {
public:
    bool checkIfPangram(string sentence) {
        set<char> my_name(sentence.begin(), sentence.end());
        return(my_name.size() == 26);
    }
};