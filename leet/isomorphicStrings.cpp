// 205
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        if(s.size() != t.size()) {
            return(false);
        }
        unordered_map<char, vector<int>> s_umap;
        unordered_map<char, vector<int>> t_umap;
        for(int i = 0; i < s.size(); i++) {
            s_umap[s[i]].push_back(i);
            t_umap[t[i]].push_back(i);
        }

        for(int i = 0; i < s.size(); i++) {
            if(s_umap[s[i]] != t_umap[t[i]]) {
                return(false);
            }
        }
        return(true);
    }
};

class Solution {
public:
    bool isIsomorphic(string s, string t) {
        if(s.size() != t.size()) {
            return(false);
        }
        unordered_map<char, vector<int>> s_umap;
        unordered_map<char, vector<int>> t_umap;
        for(int i = 0; i < s.size(); i++) {
            if(
                (s_umap[s[i]].size() > 0 || t_umap[t[i]].size() > 0) 
                && (s_umap[s[i]] != t_umap[t[i]])
            ) {
                return(false);
            }
            s_umap[s[i]].push_back(i);
            t_umap[t[i]].push_back(i);
        }
        return(true);
    }
};