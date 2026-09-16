// Regression of the actual alp-fermion source projector, before secondary decays.
#include "Pythia8/Pythia.h"
#include "symmetry_filter.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc!=2) return 2;
  Pythia8::Pythia p(argv[1],false);
  p.readString("ProcessLevel:all = off");
  p.readString("Print:quiet = on");
  if (!p.init()) throw std::runtime_error("Pythia initialization failed");
  SourceQN q;
  q.sourceJ2=0; q.expectedP=-1; q.expectedC=+1;
  q.expectedG=+1; q.enforceGProxy=true;
  DiscreteSymmetryFilter filter(p.particleData,q);
  int checked=0;
  auto check=[&](int a,int b,bool allowed) {
    p.event.reset();
    const double m1=p.particleData.m0(a),m2=p.particleData.m0(b),w=5.;
    const double e1=(w*w+m1*m1-m2*m2)/(2*w),z=std::sqrt(e1*e1-m1*m1);
    p.event.append(a,1,0,0,0.,0.,z,e1,m1);
    p.event.append(b,1,0,0,0.,0.,-z,w-e1,m2);
    if (filter.supports(p.event)!=allowed)
      throw std::runtime_error("Wrong alp-fermion support for "+std::to_string(a)+","+std::to_string(b));
    ++checked;
  };
  for (int d : {411,421,431}) check(d,-d,false); // DDbar and DsDsbar: 0- -> 0- 0- forbidden.
  for (int d : {411,421,431}) {
    check(d,-(d+2),true);      // D D*bar: S=L=1 can give J=0, P=-.
    check(-d,d+2,true);        // Charge conjugate.
    check(d+2,-(d+2),true);    // D*D*bar is also allowed.
  }
  std::cout<<checked<<" alp-fermion charm quantum-number checks passed\n";
}
