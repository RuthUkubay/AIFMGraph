#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include <queue>

#include "manager.hpp"       // FarMemManager, Factory
#include "pointer.hpp"       // far_memory::Pointer<T>
#include "deref_scope.hpp"   // far_memory::DerefScope

namespace fargraph {
using i32 = int32_t;

struct CSRFar {
  int n = 0;

  // far arrays (use the repo's pointer type)
  far_memory::Pointer<i32> off;
  far_memory::Pointer<i32> deg;
  far_memory::Pointer<i32> nbr;

  // local builder before upload
  std::vector<std::vector<i32>> adj_tmp;

  explicit CSRFar(int n_) : n(n_), adj_tmp(n_) {}
  inline void add_edge(i32 u, i32 v) { adj_tmp[u].push_back(v); }

  inline void finalize(far_memory::FarMemManager* mm) {
    std::vector<i32> off_l(n), deg_l(n);
    size_t E = 0;
    for (int u=0; u<n; ++u) { off_l[u]=(i32)E; deg_l[u]=(i32)adj_tmp[u].size(); E += deg_l[u]; }
    std::vector<i32> nbr_l; nbr_l.reserve(E);
    for (int u=0; u<n; ++u) nbr_l.insert(nbr_l.end(), adj_tmp[u].begin(), adj_tmp[u].end());

    // allocate in far memory (this repo uses manager->allocate<T>(count))
    off = mm->allocate<i32>(n);
    deg = mm->allocate<i32>(n);
    nbr = mm->allocate<i32>((i32)E);

    // ONE live scope per thread
    far_memory::DerefScope scope;
    std::memcpy(off.deref(scope), off_l.data(), n*sizeof(i32));
    std::memcpy(deg.deref(scope), deg_l.data(), n*sizeof(i32));
    std::memcpy(nbr.deref(scope), nbr_l.data(), E*sizeof(i32));

    adj_tmp.clear(); adj_tmp.shrink_to_fit();
  }

  inline void neighbors(int u, far_memory::DerefScope& scope, const i32*& ptr, i32& len) const {
    const i32* p_off = off.deref(scope);
    const i32* p_deg = deg.deref(scope);
    ptr = nbr.deref(scope) + p_off[u];
    len = p_deg[u];
  }
};

inline std::vector<int> bfs_far(const CSRFar& G, int src, std::vector<int>* parent=nullptr) {
  const int n = G.n;
  std::vector<int> dist(n, -1);
  if (parent) parent->assign(n, -1);
  std::queue<int> q; dist[src]=0; if(parent)(*parent)[src]=src; q.push(src);

  far_memory::DerefScope scope; // exactly one live scope

  while(!q.empty()){
    int u=q.front(); q.pop();

    const int* ptr=nullptr; int len=0;
    G.neighbors(u, scope, ptr, len);
    for (int i=0;i<len;i++){
      int v = ptr[i];
      if (dist[v]==-1) { dist[v]=dist[u]+1; if(parent)(*parent)[v]=u; q.push(v); }
    }
  }
  return dist;
}

} 
