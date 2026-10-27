// Pure speed benchmark: random U/V stream, scores all +1 (everything stays active).
#include "api.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <x86intrin.h>
#include <chrono>
int main(int argc,char**argv){
  int nu=atoi(argv[1]), per=atoi(argv[2]); long N=atol(argv[3]);
  std::mt19937 rng(1); std::uniform_real_distribution<double> U(0,1);
  ISolution* s=createSolution();
  for(int u=0;u<nu;u++) for(int j=0;j<per;j++) s->registerOption(Option{u*per+j,u,80+j%40,0.1+0.01*j});
  std::vector<double> S(nu,100);
  std::vector<Score> prev; std::vector<Valuation> v;
  for(int u=0;u<nu;u++){ s->process({'U',100,u},{}); }
  for(int i=0;i<nu*per;i++){ s->process({'V',0.3,i},{}); }
  auto t0=std::chrono::steady_clock::now(); uint64_t c0=__rdtsc(); double tot=0; long nv=0;
  for(long t=0;t<N;t++){
    ParameterUpdate up = U(rng)<0.05 ? ParameterUpdate{'V',0.2+0.2*U(rng),(int)(rng()%(nu*per))} : ParameterUpdate{'U',100*(1+0.01*(U(rng)-0.5)),(int)(rng()%nu)};
    uint64_t a=__rdtsc(); v=s->process(up,prev); tot+=__rdtsc()-a; nv+=v.size();
    prev.resize(v.size()); for(size_t j=0;j<v.size();j++) prev[j]={v[j].optionId,1.0};
  }
  double hz=(__rdtsc()-c0)/std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  printf("%d opts: %.1f ns/call, %.1f vals/call, %.3f ns/val\n", nu*per, tot/hz/N*1e9, (double)nv/N, tot/hz/nv*1e9);
}
