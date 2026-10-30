#include "MPM.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace teaching;
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
int main() {
 try {
    M2 a; a<<2,3,3,-1;
    require((unpack(pack(a))-a).norm()<1e-12,"tensor roundtrip");
    require(std::abs(pack(a).squaredNorm()-a.squaredNorm())<1e-12,"Frobenius isometry");
    V2 g(.8,-.3),v(.2,.6); M2 raw=v*g.transpose();
    require((strainBlock(g)*v-pack(0.5*(raw+raw.transpose()))).norm()<1e-12,"B symmetric gradient");
    V3 r(2,.4,-.2);
    require(std::abs(r.dot(strainBlock(g)*v)-v.dot(strainBlock(g).transpose()*r))<1e-12,"adjoint virtual work");
    for(int i=-4;i<=4;++i) for(int j=-4;j<=4;++j) {
        V3 p=projectCone(V3(i,j,1),.5);
        require(p[0]>=-1e-12 && p.tail<2>().norm()<=.5*p[0]+1e-12,"cone feasibility");
        require((projectCone(p,.5)-p).norm()<1e-12,"projection idempotence");
    }
    // Analytic sliding: W=I, q=(-1,2,0), mu=.5 -> lambda=(1,-.5,0), gamma=(0,1.5,0).
    V3 stress=V3::Zero(),q(-1,2,0);
    for(int i=0;i<1000;++i) stress=projectCone(stress-.2*modifiedRate(stress+q,.5),.5);
    require((stress-V3(1,-.5,0)).norm()<1e-8,"Coulomb sliding without dilatancy");
    Config cfg; cfg.gravity=0; Simulation still(cfg); auto before=still.particles(); still.step();
    for(size_t i=0;i<before.size();++i) require((still.particles()[i].x-before[i].x).norm()<1e-12,"zero-force equilibrium");
    double integratedVolume=0, particleVolume=0;
    for(const auto& p:still.particles()) particleVolume+=p.volume;
    for(const auto& c:still.stressNodes()) {
        integratedVolume+=c.solidVolume;
        V3 rigidRate=V3::Zero();
        for(const auto& b:c.B) rigidRate+=b*V2(.7,-.2);
        require(rigidRate.norm()<1e-12,"assembled B annihilates translation");
    }
    require(std::abs(integratedVolume-particleVolume)<1e-12,"Q1 volume partition");
    auto st=still.stencil(V2(.413,.527)); double sum=0; V2 grad=V2::Zero();
    for(int k=0;k<4;++k) { sum+=st.w[k]; grad+=st.grad[k]; }
    require(std::abs(sum-1)<1e-12 && grad.norm()<1e-12,"partition of unity");
    Simulation sim; double mass=0; for(const auto& p:sim.particles()) mass+=p.mass;
    for(int i=0;i<120;++i) {
        sim.step();
        require(sim.stats().converged,"default solve converges");
        require(sim.stats().safetyClamps==0,"default dynamics need no geometric guards");
    }
    double massAfter=0; for(const auto& p:sim.particles()) {
        massAfter+=p.mass; require(p.x.allFinite() && p.v.allFinite(),"finite simulation");
    }
    require(std::abs(massAfter-mass)<1e-10,"particle mass conservation");
    std::cout<<"PASS: tensor mapping, adjoint, cone, sliding, equilibrium, interpolation, finite dynamics, mass\n";
    return 0;
 } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n';return 1; }
}
