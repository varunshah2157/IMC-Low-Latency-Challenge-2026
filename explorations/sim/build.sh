# run from this folder after: cp ../../template/api.hpp .
for cfg in "A -DLLC_REQUOTE_ALL=0 -DLLC_PROBE=0 -DLLC_SIBLING=0" "B -DLLC_PROBE=0 -DLLC_SIBLING=0" "C -DLLC_SIBLING=0" "D -DLLC_PROBE=0" "E "; do set -- $cfg; nm=$1; shift; g++-13 -std=c++20 -O3 -static $@ -I. ../solution.cpp sim.cpp -o sim_$nm & done; wait
