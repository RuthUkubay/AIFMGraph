#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <queue>

struct CSRLocal {
  int32_t n = 0;
  std::vector<int32_t> off;   // size n
  std::vector<int32_t> deg;   // size n
  std::vector<int32_t> nbr;   // size sum(deg)

  // builder before finalize
  std::vector<std::vector<int32_t>> adj_tmp;

  explicit CSRLocal(int32_t n_) : n(n_), off(n_), deg(n_), adj_tmp(n_) {}

  inline void add_edge(int32_t u, int32_t v) { adj_tmp[u].push_back(v); }

  inline void finalize() {
    size_t total = 0;
    for (int i=0;i<n;i++){ off[i]=(int32_t)total; deg[i]=(int32_t)adj_tmp[i].size(); total += deg[i]; }
    nbr.resize(total);
    for (int i=0;i<n;i++){
      std::copy(adj_tmp[i].begin(), adj_tmp[i].end(), nbr.begin()+off[i]);
      std::vector<int32_t>().swap(adj_tmp[i]); // free tmp[i]
    }
    std::vector<std::vector<int32_t>>().swap(adj_tmp); // free all tmp
  }
};

// Local BFS for reference
inline std::vector<int> bfs_local(const CSRLocal& G, int src, std::vector<int>* parent=nullptr){
  const int n = G.n;
  std::vector<int> dist(n, -1);
  if (parent) parent->assign(n, -1);
  std::queue<int> q; dist[src]=0; if(parent)(*parent)[src]=src; q.push(src);
  while(!q.empty()){
    int u=q.front(); q.pop();
    int off=G.off[u], d=G.deg[u];
    for(int i=0;i<d;i++){
      int v=G.nbr[off+i];
      if(dist[v]==-1){ dist[v]=dist[u]+1; if(parent)(*parent)[v]=u; q.push(v); }
    }
  }
  return dist;
}
