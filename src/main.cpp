#include "MPM.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace teaching;

cv::Mat draw(const Simulation& sim) {
    const auto& c = sim.config();

    const int width = 1000;
    const int header = 110;

    const double scale =
        width / (c.nx * c.h);

    cv::Mat img(
        int(c.ny * c.h * scale) + header,
        width,
        CV_8UC3,
        cv::Scalar(245, 241, 235)
    );

    auto pixel = [&](V2 x) {
        return cv::Point(
            int(x.x() * scale),
            header + int((c.ny * c.h - x.y()) * scale)
        );
        };

    // 绘制网格
    for (int x = 0; x <= c.nx; ++x) {
        cv::line(
            img,
            pixel(V2(x * c.h, 0)),
            pixel(V2(x * c.h, c.ny * c.h)),
            cv::Scalar(225, 222, 216)
        );
    }

    for (int y = 0; y <= c.ny; ++y) {
        cv::line(
            img,
            pixel(V2(0, y * c.h)),
            pixel(V2(c.nx * c.h, y * c.h)),
            cv::Scalar(225, 222, 216)
        );
    }

    // 绘制容器边界
    cv::rectangle(
        img,
        pixel(V2(c.h, (c.ny - 1) * c.h)),
        pixel(V2((c.nx - 1) * c.h, c.h)),
        cv::Scalar(110, 90, 65),
        2
    );

    // 绘制粒子
    for (const auto& p : sim.particles()) {

        const double speed =
            std::min(p.v.norm() / 2.0, 1.0);

        cv::circle(
            img,
            pixel(p.x),
            2,
            cv::Scalar(
                45 + 150 * speed,
                150 - 60 * speed,
                210 - 120 * speed
            ),
            -1,
            cv::LINE_AA
        );
    }

    // 信息栏
    cv::rectangle(
        img,
        { 0, 0 },
        { width, header },
        cv::Scalar(55, 43, 30),
        -1
    );

    std::ostringstream a, b;

    a << std::fixed << std::setprecision(3)
        << "Implicit MPM 2D | t=" << sim.time()
        << "  mu=" << c.friction
        << "  APIC=" << c.APIC;

    b << std::scientific << std::setprecision(2)
        << "GS=" << sim.stats().iterations
        << " residual=" << sim.stats().residual
        << " converged=" << sim.stats().converged
        << " guards=" << sim.stats().safetyClamps;

    cv::putText(
        img, a.str(), { 18, 30 },
        cv::FONT_HERSHEY_SIMPLEX,
        0.66, { 240, 240, 240 }, 1, cv::LINE_AA
    );

    cv::putText(
        img, b.str(), { 18, 61 },
        cv::FONT_HERSHEY_SIMPLEX,
        0.58, { 210, 220, 230 }, 1, cv::LINE_AA
    );

    cv::putText(
        img,
        "SPACE: pause | N: step | R: reset | "
        "S: screenshot | ESC: quit",
        { 18, 91 },
        cv::FONT_HERSHEY_SIMPLEX,
        0.55, { 190, 200, 210 }, 1, cv::LINE_AA
    );

    return img;
}

int main(int argc, char** argv) {
    try {
        Config cfg;

        bool headless = false;
        int steps = 1000;

        std::string out = "results";

        for (int i = 1; i < argc; ++i) {

            const std::string a = argv[i];

            auto val = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(
                        "Missing value after " + a
                    );
                }
                return argv[++i];
                };

            if (a == "--headless")
                headless = true;

            else if (a == "--steps")
                steps = std::stoi(val());

            else if (a == "--mu")
                cfg.friction = std::stod(val());

            else if (a == "--dt")
                cfg.dt = std::stod(val());

            else if (a == "--iterations")
                cfg.maxIterations = std::stoi(val());

            else if (a == "--apic")
                cfg.APIC = true;

            else if (a == "--pic")
                cfg.APIC = false;

            else if (a == "--out")
                out = val();

            else if (a == "--help") {
                std::cout
                    << "--headless --steps 1000 "
                    << "--mu 0.5 --dt 0.001 "
                    << "--iterations 3000 "
                    << "--apic --pic --out results\n";
                return 0;
            }

            else {
                throw std::runtime_error(
                    "Unknown option: " + a
                );
            }
        }

        if (steps < 1) {
            throw std::runtime_error(
                "steps must be positive"
            );
        }

        std::filesystem::create_directories(out);

        Simulation sim(cfg);

        // 无图形界面模式
        if (headless) {

            std::ofstream csv(
                out + "/diagnostics.csv"
            );

            if (!csv) {
                throw std::runtime_error(
                    "Cannot write diagnostics.csv"
                );
            }

            csv
                << "step,time,iterations,residual,"
                << "converged,max_speed,max_fraction,guards\n";

            if (!cv::imwrite(
                out + "/initial.png",
                draw(sim)
            )) {
                throw std::runtime_error(
                    "Cannot write initial.png"
                );
            }

            int failures = 0;
            int guards = 0;

            for (int s = 1; s <= steps; ++s) {
                sim.step();

                const auto& q = sim.stats();

                failures += !q.converged;
                guards += q.safetyClamps;

                csv
                    << s << ','
                    << sim.time() << ','
                    << q.iterations << ','
                    << q.residual << ','
                    << q.converged << ','
                    << q.maxSpeed << ','
                    << q.maxFraction << ','
                    << q.safetyClamps << '\n';

                if (s % 100 == 0) {
                    std::cout
                        << "step " << s
                        << " residual " << q.residual
                        << " speed " << q.maxSpeed
                        << " fraction " << q.maxFraction
                        << '\n';
                }
            }

            if (!cv::imwrite(
                out + "/final.png",
                draw(sim)
            )) {
                throw std::runtime_error(
                    "Cannot write final.png"
                );
            }

            std::cout
                << "Finished: unconverged steps="
                << failures
                << ", geometric guards="
                << guards << "\n";
        }

        // 图形界面模式
        else {
            bool paused = false;

            cv::namedWindow(
                "Implicit MPM 2D",
                cv::WINDOW_AUTOSIZE
            );

            while (true) {

                if (!paused) {
                    for (int k = 0; k < 2; ++k)
                        sim.step();
                }

                cv::imshow(
                    "Implicit MPM 2D",
                    draw(sim)
                );

                const int key =
                    cv::waitKey(1) & 255;

                if (key == 27 ||
                    cv::getWindowProperty(
                        "Implicit MPM 2D",
                        cv::WND_PROP_VISIBLE
                    ) < 1) {
                    break;
                }

                if (key == ' ')
                    paused = !paused;

                if (key == 'n' || key == 'N')
                    sim.step();

                if (key == 'r' || key == 'R')
                    sim.reset();

                if (key == 's' || key == 'S') {
                    cv::imwrite(
                        out + "/screenshot.png",
                        draw(sim)
                    );
                }
            }
        }

        return 0;

    }
    catch (const std::exception& e) {
        std::cerr
            << "ERROR: " << e.what() << '\n';

        return 1;
    }
}
