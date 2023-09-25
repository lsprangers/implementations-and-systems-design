class Solution {
public:
    void replaceFirstOccurrence(
                    std::string &str,
                    const char &toReplace)
    {
        // Get index position of first occurrence
        // of given character in string
        size_t pos = str.find(toReplace);
        // Check if index is valid,
        // and char toReplace exists in string
        if (pos != std::string::npos)
        {
            // Replace the char at found index position
            str.erase(pos, 1);
        }
    }
    char findTheDifference(string s, string t) {
        for (auto &ch : s) {
            replaceFirstOccurrence(t, ch);
            cout << t << "\n";
        }
        return t[0];
    }
};