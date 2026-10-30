#include "MPM.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace teaching {
	constexpr double s2 = 0.7071067811865475244;

	V3 pack(const M2& a) {
		return V3(
			(a(0, 0) + a(1, 1)) * s2,
			(a(0, 0) - a(1, 1)) * s2,
			(a(0, 1) + a(1, 0)) * s2
		);
	}

	M2 unpack(const V3& r) {
		M2 a;
		a << (r[0] + r[1]) * s2,
			r[2] * s2,
			r[2] * s2,
			(r[0] - r[1])* s2;
		return a;
	}

	B32 strainBlock(const V2& g) {
		B32 b;
		b << g.x() * s2, g.y()* s2,
			g.x()* s2, -g.y() * s2,
			g.y()* s2, g.x()* s2;
		return b;
	}

	V3 projectCone(const V3& r, double mu) {
		const double t = r.tail<2>().norm();
		if (mu == 0.0) {
			return V3(std::max(0.0, r[0]), 0, 0);
		}
		if (r[0] >= 0.0 && t <= mu * r[0]){
			return r;
		}
		const double n = (r[0] + mu * t) / (1.0 + mu * mu);
		if (n <= 0.0) {
			return V3::Zero();
		}
		V3 out = V3::Zero();
		out[0] = n;
		if (t > 0.0) {
			out.tail<2>() = (mu * n / t) * r.tail<2>();
		}
		return out;
	}

	V3 Simulation::rate(const StressNode& c)const {
		V3 g = c.offset;
		for (int k = 0; k < 9; ++k) {
			g += c.B[k] * ns[c.nodes[k]].velocity;
		}
		return g;
	}

	V3 modifiedRate(const V3& gamma, double mu) {
		V3 q = gamma;
		q[0] += mu * gamma.tail<2>().norm();
		return q;
	}

	void Simulation::applyStress(StressNode& c, const V3& delta) {
		for (int k = 0; k < 9; ++k) {
			auto& n = ns[c.nodes[k]];
			n.velocity += cfg.dt * n.inverseMass.asDiagonal() * c.B[k].transpose() * delta;
		}
	}

	StenCil Simulation::stencil(const V2& x)const{
		const V2 q = x / cfg.h;
		const int ix = std::clamp(int(std::floor(q.x())), 0, cfg.nx - 1);
		const int iy = std::clamp(int(std::floor(q.y())), 0, cfg.ny - 1);
		const double a = q.x() - ix;
		const double b = q.y() - iy;
		StenCil s;
		s.cell = iy * cfg.nx + ix;
		s.nodes = {
			nodeIndex(ix,iy),
			nodeIndex(ix + 1,iy),
			nodeIndex(ix,iy + 1),
			nodeIndex(ix + 1,iy + 1)
		};
		s.w = {
			(1 - a) * (1 - b),
			a * (1 - b),
			(1 - a) * b,
			a*b
		};
		s.grad = {
			V2(-(1 - b),-(1 - a)) / cfg.h,
			V2(1 - b,-a) / cfg.h,
			V2(-b,1 - a) / cfg.h,
			V2(b,a) / cfg.h
		};
		return s;
	}

	Simulation::Simulation(Config config) :cfg(config) {
		if (cfg.nx < 28 || cfg.ny < 23 ||
			cfg.h <= 0 || cfg.dt <= 0 ||
			cfg.density <= 0 ||
			cfg.phiMax <= 0 || cfg.phiMax > 1 ||
			cfg.friction < 0 ||
			cfg.tolerance <= 0 ||
			cfg.relaxation <= 0 ||
			cfg.relaxation > 1 ||
			cfg.maxIterations < 1 ||
			!std::isfinite(
				cfg.dt + cfg.h + cfg.density +
				cfg.phiMax + cfg.friction +
				cfg.gravity + cfg.relaxation +
				cfg.tolerance
			)) {
			throw std::invalid_argument(
				"Invalid simulation configuration"
			);
		}
		ns.resize((cfg.nx + 1) * (cfg.ny + 1));
		cs.resize(ns.size());
		reset();
	}

	void Simulation::reset() {
		ps.clear();
		elapsed = 0.0;
		diagnostic = {};
		for (int i = 0; i < ns.size(); ++i) {
			ns[i] = Node{};
		}
		for (int i = 0; i < cs.size(); ++i) {
			cs[i] = StressNode{};
		}
		const int particlePerAxis = 4;
		std::mt19937 gen(12345);
		std::uniform_real_distribution<double>
			jitter(-0.45, 0.45);

		for (int y = 1; y < 18; ++y) {
			for (int x = 9; x < 19; ++x) {
				for (int b = 0; b < particlePerAxis; ++b) {
					for (int a = 0; a < particlePerAxis; ++a) {
						Particle p;
						p.x = cfg.h * V2(x + (a + 0.5 + jitter(gen))/ particlePerAxis,
										y + (b + 0.5 + jitter(gen))/ particlePerAxis
										);
						p.volume =cfg.phiMax * cfg.h * cfg.h /(particlePerAxis * particlePerAxis);
						p.mass = cfg.density * p.volume;
						ps.push_back(p);
					}
				}
			}
		}
	}

	void Simulation::particlesToGrid() {   //P2G 
		for (int i = 0; i < static_cast<int>(ns.size()); ++i) {
			ns[i].inverseMass.setZero();
			ns[i].mass = 0.0;
			ns[i].velocity.setZero();
		}
		for (int i = 0; i < static_cast<int>(ps.size()); ++i) {
			const Particle& p = ps[i];
			const StenCil s = stencil(p.x);
			for (int j = 0; j < 4; ++j) {
				const int id = s.nodes[j];
				auto& n = ns[id];
				V2 v = p.v;
				if (cfg.APIC == true) {
					const V2 xi = cfg.h * V2(id % (cfg.nx + 1), id / (cfg.nx + 1));
					v += p.C*(xi - p.x);
				}
				const double weight = p.mass * s.w[j];
				n.mass += weight;
				n.velocity += weight * v;
			}
		}
		for (int y = 0; y <= cfg.ny; ++y) {
			for (int x = 0; x <= cfg.nx; ++x) {
				auto& n = ns[nodeIndex(x, y)];

				if (n.mass <= 1e-12) {
					continue;
				}

				n.velocity /= n.mass;
				n.velocity.y() += cfg.dt * cfg.gravity;
				n.inverseMass = V2(1.0 / n.mass, 1.0 / n.mass); 

				if (y <= 1) {
					n.velocity.setZero();
					n.inverseMass.setZero();
				}
				if (x <= 1 || x >= cfg.nx - 1) {
					n.velocity.x() = 0.0;
					n.inverseMass.x() = 0.0;
				}
				if (y >= cfg.ny - 1) {
					n.velocity.y() = 0.0;
					n.inverseMass.y() = 0.0;
				}
			}
		}

	}

	void Simulation::buildConstraints() {  //组装应力—速度关系与堆积余量
		for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
			cs[i] = StressNode{};
		}
		for (int y = 0; y <= cfg.ny; ++y) {
			for (int x = 0; x <= cfg.nx; ++x) {
				auto& c = cs[nodeIndex(x, y)];
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) {
						c.nodes[(dy + 1) * 3 + dx + 1] = nodeIndex(std::clamp(x + dx, 0, cfg.nx), std::clamp(y + dy, 0, cfg.ny));
					}
				}
			}
		}
		
		for (int i = 0; i < static_cast<int>(ps.size()); ++i) {
			const Particle& p = ps[i];
			const StenCil s = stencil(p.x);
			for (int j = 0; j < 4; ++j) { //应力节点
				const int id = s.nodes[j];
				const int x = id % (cfg.nx + 1);
				const int y = id / (cfg.nx + 1);
				auto& c = cs[id];
				c.solidVolume += p.volume * s.w[j];
				for (int k = 0; k < 4; ++k) {  //速度节点
					const int dx = s.nodes[k] % (cfg.nx + 1) - x;
					const int dy = s.nodes[k] / (cfg.nx + 1) - y;
					const int index = (dy + 1) * 3 + dx + 1;
					c.B[index] += p.volume * s.w[j] * strainBlock(s.grad[k]);
				}
			}
		}

		for (int id = 0; id < static_cast<int>(cs.size()); ++id) {
			auto& c = cs[id];
			if (c.solidVolume == 0.0) {
				continue;
			}
			const int x = id % (cfg.nx + 1);
			const int y = id / (cfg.nx + 1);
			const double are = cfg.h * cfg.h * ((x == 0 || x == cfg.nx) ? 0.5 : 1.0) * ((y == 0 || y == cfg.ny) ? 0.5 : 1.0);
			const double gap = cfg.phiMax * are - c.solidVolume;
			const double slack = std::max(0.0, gap);
			c.offset[0] = slack * s2 / cfg.dt;

			Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
			for (int k = 0; k < 9; ++k) {
				const auto& n = ns[c.nodes[k]];
				W += cfg.dt * c.B[k] * n.inverseMass.asDiagonal() * c.B[k].transpose();
			}
			if (W.trace() > 1e-20) {
				c.stepSize = cfg.relaxation / ((1.0 + cfg.friction) * W.trace());
			}
			diagnostic.maxFraction = std::max(diagnostic.maxFraction, c.solidVolume / are);
		}

	} 
	void Simulation::solveStress() {    //求出与速度相协调的摩擦应力
		for (int iteration = 0; iteration < cfg.maxIterations; ++iteration) {
			for (int j = 0; j < static_cast<int>(cs.size()); ++j) {
				const int i = ((iteration % 2 == 0) ? j : int(cs.size() - 1 - j));

				auto& c = cs[i];
				if (c.stepSize == 0.0) {
					continue;
				}

				const V3 gamma = rate(c);

				const V3 next = projectCone(c.stress - c.stepSize * modifiedRate(gamma, cfg.friction), cfg.friction);
				const V3 delta = next - c.stress;
				applyStress(c, delta);
				c.stress = next;
			}
			double error = 0.0;
			double scale = 1.0;
			for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
				auto& c = cs[i];

				if (c.stepSize == 0.0) {
					continue;
				}

				const V3 projected = projectCone(c.stress - c.stepSize * modifiedRate(rate(c), cfg.friction), cfg.friction);
				error = std::max(error, (c.stress - projected).norm());
				scale = std::max(scale, c.stress.norm());
			}
			diagnostic.iterations = iteration + 1;
			diagnostic.residual = error / scale;
			if (diagnostic.residual < cfg.tolerance) {
				diagnostic.converged = true;
				break;
			}
		}
	}   
	void Simulation::gridToParticles() {  //G2P
		for (int i = 0; i < static_cast<int>(ps.size()); ++i) {
			auto& p = ps[i];
			const StenCil s = stencil(p.x);
			p.v.setZero();
			p.C.setZero();
			for (int k = 0; k < 4; ++k) {
				const V2 v = ns[s.nodes[k]].velocity;
				p.v += v * s.w[k];
				if (cfg.APIC == true) {
					p.C += v * s.grad[k].transpose();
				}
			}
			p.x += p.v * cfg.dt;

			for (int a = 0; a < 2; ++a) {
				const double lo = cfg.h + 1e-8;
				const double hi = ((a == 0 ? cfg.nx : cfg.ny) - 1) * cfg.h - 1e-8;
				if (p.x[a]<lo || p.x[a]>hi) {
					p.x[a] = std::clamp(p.x[a], lo, hi);
					p.v[a] = 0.0;
					p.C.setZero();
					++diagnostic.safetyClamps;
				}
			}
			if (!p.x.allFinite() || !p.v.allFinite() || !p.C.allFinite()) {
				throw std::runtime_error("Non-finite state: reduce dt or check solver");
			}
			diagnostic.maxSpeed = std::max(diagnostic.maxSpeed, p.v.norm());
		}
	} 

	void Simulation::step() {
		diagnostic = {};

		particlesToGrid();
		buildConstraints();
		solveStress();
		gridToParticles();
		elapsed += cfg.dt;
	}
}

