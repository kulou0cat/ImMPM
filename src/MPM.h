#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>

namespace teaching {
	using V2 = Eigen::Vector2d;
	using V3 = Eigen::Vector3d;
	using M2 = Eigen::Matrix2d;
	using B32 = Eigen::Matrix<double, 3, 2>;

	struct Config
	{
		int nx = 40;
		int ny = 30;
		double h = 0.04;
		double dt = 0.001;
		double density = 1500.0;
		double phiMax = 0.60;
		double friction = 0.5;
		double gravity = -9.81;
		int maxIterations = 5000;
		double tolerance = 2e-5;
		double relaxation = 0.65;
		bool APIC = false;
	};

	struct Particle     //粒子
	{
		V2 x = V2::Zero();
		V2 v = V2::Zero();
		M2 C = M2::Zero();
		double volume = 0.0;
		double mass = 0.0;
	};

	struct Node {         //网格节点
		V2 velocity = V2::Zero();
		V2 inverseMass = V2::Zero();
		double mass = 0.0;
	};

	struct StressNode {  //应力节点
		std::array<int, 9> nodes{};
		std::array<B32, 9> B;
		StressNode() {
			for (int i = 0; i < 9; i++) {
				B[i].setZero();
			}
		}
		V3 stress = V3::Zero();
		V3 offset = V3::Zero();
		double solidVolume = 0.0;
		double stepSize = 0.0;
	};

	struct StenCil {
		int cell = 0;
		std::array<int, 4> nodes{};
		std::array<double, 4> w{};
		std::array<V2, 4> grad{};
	};

	struct Stats {
		int iterations = 0;
		int safetyClamps = 0;
		double residual = 0.0;
		double maxSpeed = 0.0;
		double maxFraction = 0.0;
		bool converged = false;
	};

	V3 pack(const M2& tensor);
	M2 unpack(const V3& tensor);
	B32 strainBlock(const V2& gradient);
	V3 projectCone(const V3& r, double mu);
	V3 modifiedRate(const V3& gamma, double mu);

	class Simulation {
	public:
		explicit Simulation(Config config = {});
		void reset();
		void step();
		StenCil stencil(const V2& x) const;
		const Config& config() const { return cfg; }
		const std::vector<Particle>& particles() const { return ps; }
		const std::vector<StressNode>& stressNodes() const { return cs; }
		const Stats& stats() const { return diagnostic; }
		double time() const { return elapsed; }
	private:
		Config cfg;
		std::vector<Particle> ps;
		std::vector<Node> ns;
		std::vector<StressNode> cs;
		Stats diagnostic;
		double elapsed = 0.0;
		int nodeIndex(int x, int y) const { return y * (cfg.nx + 1) + x; }
		void particlesToGrid(); // paper 5.3: APIC and free velocity
		void buildConstraints(); // paper 5.1/5.2: B, k, local Delassus block
		V3 rate(const StressNode& c) const;
		void applyStress(StressNode& c, const V3& delta);
		void solveStress(); // paper Eq.21, simplified projected Gauss-Seidel
		void gridToParticles(); // paper Eq.19/23
	};
}