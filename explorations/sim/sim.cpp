#include "api.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>
double N(double x){return 0.5*std::erfc(-x/std::sqrt(2.0));}
double bs(double S,double s,double K,double T){double ss=s*sqrt(T);double d1=(log(S/K)+(0.04+0.5*s*s)*T)/ss;return S*N(d1)-K*exp(-0.04*T)*N(d1-ss);}
double vega(double S,double s,double K,double T){double ss=s*sqrt(T);double d1=(log(S/K)+(0.04+0.5*s*s)*T)/ss;return S*sqrt(T)*exp(-0.5*d1*d1)/2.5066;}
int main(int argc,char**argv){
  int nU=atoi(argv[1]), per=atoi(argv[2]), M=atoi(argv[3]); double pV=atof(argv[4]);
  double fs=atof(argv[5]), is=atof(argv[6]); int scoreType=atoi(argv[7]); double thrVol=atof(argv[8]); int seed=argc>9?atoi(argv[9]):1;
  std::mt19937_64 rng(seed); std::normal_distribution<double> g(0,1); std::uniform_real_distribution<double> U(0,1);
  ISolution* sol=createSolution();
  int n=nU*per; std::vector<int> und(n),id(n); std::vector<double>K(n),T(n),w(n),s0(n),e(n,0),S(nU),F(nU,0);
  for(int u=0;u<nU;u++) S[u]=50+100*U(rng);
  std::vector<int> perm(n); for(int i=0;i<n;i++)perm[i]=i; std::shuffle(perm.begin(),perm.end(),rng);
  for(int i=0;i<n;i++){und[i]=i%nU;id[i]=1000+perm[i]*3;K[i]=std::round(S[und[i]]*(0.7+0.6*U(rng)));T[i]=0.02+U(rng)*0.98;w[i]=exp(g(rng));s0[i]=0.15+0.5*U(rng);
    sol->registerOption(Option{id[i],7000+und[i]*11,(int)K[i],T[i]});}
  std::vector<int> idx(4000000,-1); for(int i=0;i<n;i++) idx[id[i]]=i;
  std::vector<char> hasS(nU,0);
  auto sig=[&](int i){return s0[i]*exp(F[und[i]]+e[i]);};
  std::vector<Score> prev; double tot=0; double ns=0; long nq=0;
  for(int t=0;t<M;t++){
    for(int u=0;u<nU;u++)F[u]+=fs*g(rng);
    for(int i=0;i<n;i++)e[i]+=is*g(rng);
    ParameterUpdate up;
    if(U(rng)<pV){int i=rng()%n; up={'V',sig(i),id[i]};}
    else{int u=rng()%nU; S[u]*=exp(0.002*g(rng)); hasS[u]=1; up={'U',S[u],7000+u*11};}
    auto t0=std::chrono::steady_clock::now();
    auto v=sol->process(up,prev);
    auto t1=std::chrono::steady_clock::now(); ns+=std::chrono::duration<double,std::nano>(t1-t0).count();
    prev.clear(); std::vector<char> seen(n,0);
    for(auto&q:v){int i=idx[q.optionId]; if(i<0||seen[i]||!std::isfinite(q.price))continue; seen[i]=1; int u=und[i]; if(!hasS[u])continue;
      double s=sig(i); double ref=bs(S[u],s,K[i],T[i]); double thr=std::max(0.005,thrVol*vega(S[u],s,K[i],T[i]));
      double x=std::fabs(q.price-ref)/thr; double p= scoreType==0? w[i]*(1-x) : scoreType==1? w[i]*(1-x*x) : (x<1?w[i]:-w[i]);
      p=std::max(p,-5*w[i]); prev.push_back({q.optionId,p}); tot+=p; nq++;}
  }
  printf("pnl %.0f  quotes %ld  avg_ns/call %.1f  total_ms %.1f\n",tot,nq,ns/M,ns/1e6);
}
