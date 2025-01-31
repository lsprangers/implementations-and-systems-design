// In quickUnion we store the are able to union things faster, and
//  just swap the parent of the root of another group
//  therefore, the union() operation is O(1) 
//  but the find() operation is O(n)

//      0000
//     /    \
//    1      2
//   / \    / \
//  3   4  5   6

// Example:
// parent   = [0, 0, 0, 1, 1, 2, 2]
// root     = [0, 0, 0, 1, 1, 2, 2]
// So we see here that root just has 0's

// If we have a tree like this:
//    0
//    |    
//    1      2
//   / \    / \
//  3   4  5   6

// Example:
// parent   = [0, 0, 2, 1, 1, 2, 2]
// root     = [0, 0, 2, 1, 1, 2, 2]

// Then we see our find() becomes O(n) since we need to traverse potentially
//  the entire array
// And in union() we just need to change parent[rootY] = rootX and root[rootY] = rootX

// Example:
// parent   = [0, 0, 0, 1, 1, 2, 2]
// root     = [0, 0, 2, 0, 0, 2, 2]
// After this, we need to update 5 and 6 to have
//  root[5] = 0 and root[6] = 0

#include <vector>;
#include <iostream>;
using namespace std;

class quickFind {
public:
    quickFind(int n) {
        // Constructor
        // parent = vector<int>(n);
        root = vector<int>(n, 0);
        count = n;
        for (int i = 0; i < n; ++i) {
            root[i] = i;
        }
    }

    int find(int x) {
        // Could possible be O(n)
        if(x == root[x]) {
            return(x);
        }
        else {
            return(find(root[x]));
        }
    }

    void unionSet(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        // O(1) just need to choose one
        if (rootX != rootY) {
            root[rootY] = rootX;
        }
        
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }

private:
    vector<int> parent;
    vector<int> root;
    int count;
};


// Test Case
int main() {
    // for displaying booleans as literal strings, instead of 0 and 1
    cout << boolalpha;
    quickFind uf(10);
    // 1-2-5-6-7 3-8-9 4
    uf.unionSet(1, 2);
    uf.unionSet(2, 5);
    uf.unionSet(5, 6);
    uf.unionSet(6, 7);
    uf.unionSet(3, 8);
    uf.unionSet(8, 9);
    cout << uf.connected(1, 5) << endl;  // true
    cout << uf.connected(5, 7) << endl;  // true
    cout << uf.connected(4, 9) << endl;  // false
    // 1-2-5-6-7 3-8-9-4
    uf.unionSet(9, 4);
    cout << uf.connected(4, 9) << endl;  // true

    return 0;
}