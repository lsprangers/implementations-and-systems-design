#include <queue>;
#include <unordered_set>;
#include <iostream>;
using namespace std;
//433
class Solution {
public:
    
    
    int minMutation(string startGene, string endGene, vector<string>& bank) {
        queue<pair<string, int>> q;
        q.push(pair(startGene, 0));
        unordered_set<string> the_bank(bank.begin(), bank.end());   
        unordered_set<string> seen;

        while(!q.empty()) {
            int q_size = q.size();
            for(int i = 0; i < q_size; i++){
                auto [gene, n_mutations] = q.front();
                q.pop();
                if(gene == endGene) {
                    return(n_mutations);
                }
                for(string good_neighbor : get_neighbors(gene, the_bank, seen)) {
                    seen.insert(good_neighbor);
                    q.push(pair(good_neighbor, n_mutations + 1));
                }
            }

        }
        return(-1); 
    }

    vector<string> get_neighbors(string gene, unordered_set<string> the_bank, unordered_set<string> seen) {
        vector<string> resp;
        for(int i = 0; i < gene.size(); i++) {
            for(char new_gene_char : "ACGT") {
                string neighbor = gene;
                neighbor[i] = new_gene_char;
                if(the_bank.find(neighbor) != the_bank.end() && seen.find(neighbor) == seen.end()) {
                    resp.push_back(neighbor);
                }
            }
        }
        return(resp);
    }
};