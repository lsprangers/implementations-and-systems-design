    struct CompareByIndex {
        int index;

        CompareByIndex(int idx) : index(idx) {}

        bool operator()(const vector<int>& a, const vector<int>& b) const {
            return a[index] > b[index]; // Change to '<' for ascending order
        }
    };

    priority_queue<vector<int>, vector<vector<int>>, CompareByIndex> minHeap(CompareByIndex(2));