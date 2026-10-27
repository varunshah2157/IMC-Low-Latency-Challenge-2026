#include <chrono>
#include <cstdio>
#include <cmath>
#include <random>
#define main main_unused
#include "../solution.cpp"
#undef main
double Nr(double x){return 0.5*std::erfc(-x/std::sqrt(2.0));}
int main(){
  int n=1024; std::vector<double> sig(n),sqT(n),logK(n),rT(n),T(n),dK(n),out(n),K(n);
  std::mt19937 r(1); std::uniform_real_distribution<double>U(0,1);
  double S=100;
  for(int i=0;i<n;i++){K[i]=70+60*U(r);T[i]=0.02+U(r);sig[i]=0.15+0.5*U(r);sqT[i]=sqrt(T[i]);logK[i]=log(K[i]);rT[i]=0.04*T[i];dK[i]=K[i]*exp(-0.04*T[i]);}
  bsKernel(0,n,S,log(S),sig.data(),sqT.data(),logK.data(),rT.data(),T.data(),dK.data(),out.data());
  double me=0;for(int i=0;i<n;i++){double ss=sig[i]*sqT[i];double d1=(log(S/K[i])+(0.04+0.5*sig[i]*sig[i])*T[i])/ss;double ref=S*Nr(d1)-dK[i]*Nr(d1-ss);me=std::max(me,fabs(ref-out[i]));}
  double xe=0; for(double x=-40;x<40;x+=0.001) xe=std::max(xe,fabs(ncdf(x)-Nr(x)));
  printf("max abs price err %.3e  max ncdf err %.3e\n",me,xe);
  auto t0=std::chrono::steady_clock::now(); double acc=0;
  for(int k=0;k<20000;k++){bsKernel(0,n,S+k*1e-6,log(S),sig.data(),sqT.data(),logK.data(),rT.data(),T.data(),dK.data(),out.data());acc+=out[k%n];}
  auto t1=std::chrono::steady_clock::now();
  printf("ns/option %.2f (%g)\n",std::chrono::duration<double,std::nano>(t1-t0).count()/20000/n,acc);
}
