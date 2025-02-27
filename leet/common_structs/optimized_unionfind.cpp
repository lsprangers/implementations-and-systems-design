// To close it out all, just add parent, root, and rank vectors
// Initial state:
// Elements: 0, 1, 2, 3, 4
// Roots:    0, 1, 2, 3, 4
// Ranks:    0, 0, 0, 0, 0

// Union operations:

// unionSet(0, 1)
// 0 and 1 are their own roots with equal rank
// Attach 1 under 0 and increment the rank of 0
// Roots:    0, 0, 2, 3, 4
// Ranks:    1, 0, 0, 0, 0

// unionSet(2, 3)
// 2 and 3 are their own roots with equal rank
// Attach 3 under 2 and increment the rank of 2
// Roots:    0, 0, 2, 2, 4
// Ranks:    1, 0, 1, 0, 0

// unionSet(0, 2)
// Roots of 0 and 2 are 0 and 2 respectively, with equal rank
// Attach 2 under 0 and increment the rank of 0
// Roots:    0, 0, 0, 2, 4
// Ranks:    2, 0, 1, 0, 0

// unionSet(3, 4)
// Root of 3 is 2, root of 4 is 4
// Attach 4 under 2 (root of 3)
// Roots:    0, 0, 0, 2, 2
// Ranks:    2, 0, 1, 0, 0

// unionSet(1, 4)
// Root of 1 is 0, root of 4 is 2
// Attach 2 (root of 4) under 0 (root of 1)
// Roots:    0, 0, 0, 0, 2
// Ranks:    2, 0, 1, 0, 0

// Final state:
// Elements: 0, 1, 2, 3, 4
// Roots:    0, 0, 0, 0, 0
// Ranks:    2, 0, 1, 0, 0


// Elements: 0, 1, 2, 3, 4
// Roots:    0, 1, 2, 3, 4
// Ranks:    0, 0, 0, 0, 0

// 0   1   2   3   4

// 0 and 1 are their own roots with equal rank.
// Attach 1 under 0 and increment the rank of 0.

// Roots:    0, 0, 2, 3, 4
// Ranks:    1, 0, 0, 0, 0

//     0       2   3   4
//    / 
//   1

// 2 and 3 are their own roots with equal rank.
// Attach 3 under 2 and increment the rank of 2.

// Roots:    0, 0, 2, 2, 4
// Ranks:    1, 0, 1, 0, 0

//     0       2       4
//    /       / 
//   1       3


// Union(0, 2)
// Roots of 0 and 2 are 0 and 2 respectively, with equal rank.
// Attach 2 under 0 and increment the rank of 0.

// Roots:    0, 0, 0, 2, 4
// Ranks:    2, 0, 1, 0, 0

//       0       4
//      / \
//     1   2
//          \
//           3


// Union(3, 4)

// Root of 3 is 2, root of 4 is 4.
// Attach 4 under 2 (root of 3).

// Roots:    0, 0, 0, 2, 2
// Ranks:    2, 0, 1, 0, 0

//       0
//      / \
//     1   2
//          \
//           3
//            \
//             4


#include <vector>;
#include <iostream>;
using namespace std;

class optimizedUnionFind {
public:
    optimizedUnionFind(int n) {
        // Constructor
        parent = vector<int>(n);
        root = vector<int>(n);

        // Rank is used as a relative metric of the tree
        //  height , but it isn't necessarily the
        //  exact height of the tree
        // - It's just a way for us to ensure tree is flat-ish
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
    //When performing a union operation, the tree with the 
    //  smaller rank is attached under the root of the tree 
    //  with the larger rank to keep things shallow

    // If larger rank went under smaller, then it would make the tree deeper

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