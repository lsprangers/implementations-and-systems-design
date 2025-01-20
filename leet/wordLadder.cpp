#include <queue>;
#include <set>;
#include <iostream>;
using namespace std;
//127

class Solution {
public:
    queue<string> q;
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        // O(1) lookup
        set<string> seen;
        set<string> words(wordList.begin(), wordList.end());
        if(words.find(endWord) == words.end()) {
            return(0);
        }

        seen.insert(beginWord);
        q.push(beginWord);
        int resp = 0;

        while(!q.empty()) {
            int q_size = q.size();
            resp++;
            for(int i = 0; i < q_size; i++) {
                string currBeginWord = q.front();
                if(currBeginWord == endWord) {
                    return(resp);
                }
                q.pop();
                for(string compareWord : wordList) {
                    if(checkValid(currBeginWord, compareWord) && seen.find(compareWord) == seen.end()) {
                        seen.insert(compareWord);
                        q.push(compareWord);
                    }
                }
            }
        }
        return(0);
    }
    bool checkValid(string beginWord, string compareWord) {
        int diffs = 0;
        for(int i = 0; i < beginWord.size(); i++) {
            if(beginWord[i] != compareWord[i]) {
                diffs++;
            }
            if(diffs > 1) {
                return(false);
            }
        }
        return(diffs == 1);
    }
};