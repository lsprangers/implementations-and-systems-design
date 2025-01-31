// In the previous 2 versions, there was a bunch of O(n) operations in the worst case
// These can always be fixed by balancing tree's out 

// So we use a rank[] vector to store the height of the tree which'll ensure
//   there's no worst-case O(n)

// If we have a tree like this:
//    0
//    |    
//    1      2
//   / \    / \
//  3   4  5   6

// Example:
// root     = [0, 0, 2, 1, 1, 2, 2]
// rank     = [1, 2, 1, 2, 2, 2, 2]

// If we connect 2 to 0, we'd have:
//      0000
//     /    \
//    1      2
//   / \    / \
//  3   4  5   6

// Example:
// root     = [0, 0, 0, 1, 1, 2, 2]
// rank     = [1, 2, 2, 3, 3, 3, 3]

#include <vector>;
#include <iostream>;
using namespace std;

class optimizedUnionFind {
public:
    optimizedUnionFind(int n) {
        // Constructor
        root = vector<int>(n);
        rank = vector<int>(n, 0);
        count = n;
        for (int i = 0; i < n; ++i) {
            root[i] = i;
        }
    }

    // If we have a tree like this:
    //    0
    //    |    
    //    1      2
    //   / \    / \
    //  3   4  5   6

    int find(int x) {
        // Worst case O(log n)
        if(x == root[x]) {
            return(x);
        }
        return(find(root[x]));
    }

    void unionSet(int x, int y) {
        // O(log n)
        int rootX = find(x);
        int rootY = find(y);
        if(rootX != rootY) {
            // if rank[X] is higher, i.e. the tree is taller, then put Y under X, keeping it wider
            if(rank[rootX] > rank[rootY]) {
                root[rootY] = rootX;
            }
            else if(rank[rootX] < rank[rootY]) {
                root[rootX] = rootY;
            }
            // rank represents the upper bound of the tree, so we
            //  only need to udpate it in tie's
            else {
                root[rootY] = rootX;
                rank[rootX]++;
            }
            count--;
        }
    }




private:
    vector<int> parent;
    vector<int> root;
    vector<int> rank;
    int count;
}