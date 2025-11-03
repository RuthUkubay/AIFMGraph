#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <cassert>

class Graph {
public:
    Graph(int n, bool directed = false)
        : n_(n), directed_(directed), adj_(n) {}

    // add an edge u -> v
    void add_edge(int u, int v) {
        assert(u >= 0 && u < n_ && v >= 0 && v < n_);
        adj_[u].push_back(v);
        if (!directed_) adj_[v].push_back(u);
    }

    // BFS over unweighted edges; fills parent if provided
    std::vector<int> bfs(int src, std::vector<int>* parent = nullptr) const {
        std::vector<int> dist(n_, -1);
        if (parent) parent->assign(n_, -1);

        std::queue<int> q;
        dist[src] = 0;
        if (parent) (*parent)[src] = src;
        q.push(src);

        while (!q.empty()) {
            int u = q.front(); q.pop();

            // In AIFM, prefetch this contiguous neighbor range once:
            // aifm_prefetch(adj_[u].data(), adj_[u].size() * sizeof(int));??

            for (int v : adj_[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    if (parent) (*parent)[v] = u;
                    q.push(v);
                }
            }
        }
        return dist;
    }

    // reconstruct src -> dst path using parent from bfs()
    static std::vector<int> build_path(int src, int dst, const std::vector<int>& parent) {
        std::vector<int> path;
        if (src < 0 || dst < 0 || src >= (int)parent.size() || dst >= (int)parent.size()) return path;
        for (int v = dst; v != -1 && v != parent[v]; v = parent[v]) path.push_back(v);
        if (src >= 0) path.push_back(src);
        std::reverse(path.begin(), path.end());
        if (path.front() != src) return {};
        return path;
    }

    int size() const { return n_; }

private:
    int n_;
    bool directed_;
    std::vector<std::vector<int>> adj_; // contiguous neighbors per node
};

// --- demo ---
int main() {
    Graph g(6, /*directed=*/false);
    g.add_edge(0,1);
    g.add_edge(0,2);
    g.add_edge(1,3);
    g.add_edge(2,3);
    g.add_edge(3,4);
    g.add_edge(4,5);

    std::vector<int> parent;
    auto dist = g.bfs(0, &parent);
    auto path = Graph::build_path(0, 5, parent);

    std::cout << "BFS distance to 5 = " << dist[5] << "\nPath: ";
    for (int v : path) std::cout << v << " ";
    std::cout << "\n";
}
