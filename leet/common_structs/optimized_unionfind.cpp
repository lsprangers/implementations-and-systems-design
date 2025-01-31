// To close it out all, just add parent, root, and rank vectors

#include <vector>;
#include <iostream>;
using namespace std;

class optimizedUnionFind {
public:
    optimizedUnionFind(int n) {
        // Constructor
        parent = vector<int>(n);
        root = vector<int>(n);
        rank = vector<int>(n, 0);
        count = n;
        for (int i = 0; i < n; ++i) {
            root[i] = i;
            rank[i] = i;
        }
    }

    // If we have a tree like this:
    //    0
    //    |    
    //    1      2
    //   / \    / \
    //  3   4  5   6

    // Optimized find() updates path along the way
    //  i.e. pathCompression, so find is O(log n)
    //  and connected() is O(1)
    int find(int x) {
        // O(1)
        if(x == root[x]) {
            return(x);
        }
        // -->>> Changes here! <<<--
        // We update the actual root of the node to this one
        return(root[x] = find(root[x]));
    }

    // Optimized unionSet ensures we don't have a worst-case O(n)
    //  and that we're rebalancing tree's along the way
    void unionSet(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                root[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                root[rootX] = rootY;
            } else {
                root[rootY] = rootX;
                rank[rootX] += 1;
            }
        }
    }
    bool connected(int x, int y) {
        return find(x) == find(y);
    }    

private:
    vector<int> parent;
    vector<int> root;
    vector<int> rank;
    int count;
}