// Guards the normal-grid loss of GrowPatch (NormalConstraintPlane::calculate_normal_snapping_loss).
// Its per-segment sum is folded into one Jet product for speed; these tests check that the folded
// sum keeps the value and the derivatives of the plain per-segment formula it replaced, so the
// speed-up cannot silently change the surfaces GrowPatch traces.

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/tracer/CostFunctions.hpp"

// ceres pulls in glog, whose CHECK would replace the test framework's; the framework comes last.
#undef CHECK
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;
using vc::core::util::GridStore;
using vc::core::util::NormalGridVolume;
using J = ceres::Jet<double, 4>;  // derivatives with respect to p1.x, p1.y, p2.x, p2.y

namespace {

// Wavy, partly crossing lines with vertices a few pixels apart, like traced sheet contours.
void fillGrid(GridStore& grid)
{
    std::mt19937 rng(3);
    std::normal_distribution<float> wobble(0.0f, 0.6f);
    for (int off = 0; off < 400; off += 13) {
        std::vector<cv::Point> line;
        for (int t = 0; t < 400; t += 3) {
            const float y = off + 6.0f * std::sin(t / 37.0f) + 0.15f * t + wobble(rng);
            line.emplace_back(t, static_cast<int>(std::lround(y)));
        }
        grid.add(line);
    }
    std::vector<cv::Point> diagonal;
    for (int t = 0; t < 400; t += 4) {
        diagonal.emplace_back(t, 399 - t);
    }
    grid.add(diagonal);
}

// A one-slice normal-grid store, only needed to construct NormalConstraintPlane.
fs::path makeStoreDir()
{
    std::mt19937_64 rng(std::random_device{}());
    const fs::path dir = fs::temp_directory_path() / ("vc_ncp_" + std::to_string(rng()));
    fs::create_directories(dir);
    std::ofstream(dir / "metadata.json") << "{\"sparse-volume\": 1}";
    return dir;
}

// The per-segment formula as it was before the folded sum: every in-range segment adds
// w * (1 - |e . n|) on T, with the same filtering (query radius, then ROI radius) and weights.
template <typename T>
T referenceNormalLoss(const T* p1, const T* p2, const GridStore& grid, bool direction_aware,
                      float roi_radius, float query_radius)
{
    using NCP = NormalConstraintPlane;
    const T ex = p2[0] - p1[0];
    const T ey = p2[1] - p1[1];
    const T len = ceres::sqrt(ex * ex + ey * ey);
    const T nx = ey / len;
    const T ny = -ex / len;
    const cv::Point2f p1_cv(val(p1[0]), val(p1[1]));
    const cv::Point2f p2_cv(val(p2[0]), val(p2[1]));

    T weighted = T(0.0);
    T total = T(0.0);
    for (const auto& path_ptr : grid.get(0.5f * (p1_cv + p2_cv), query_radius)) {
        const std::vector<cv::Point>& path = *path_ptr;
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            const cv::Point2f a = path[i];
            const cv::Point2f b = path[i + 1];
            float dist_sq = NCP::seg_dist_sq_appx(p1_cv, p2_cv, a, b);
            if (dist_sq > query_radius * query_radius || dist_sq > roi_radius * roi_radius) {
                continue;
            }
            dist_sq = std::max(10.0f, dist_sq);
            const T w = T(1.0 / dist_sq);
            cv::Point2f tangent = b - a;
            const float length = cv::norm(tangent);
            if (length > 0) {
                tangent /= length;
            }
            T dot = nx * T(-tangent.y) + ny * T(tangent.x);
            if (!direction_aware) {
                dot = ceres::abs(dot);
            }
            weighted += w * (T(1.0) - dot);
            total += w;
        }
    }
    return total > T(1e-9) ? weighted / total : T(0.0);
}

void checkMatchesReference(bool direction_aware)
{
    const fs::path dir = makeStoreDir();
    const NormalGridVolume ngv(dir.string());
    GridStore grid(cv::Rect(0, 0, 400, 400), 64);
    fillGrid(grid);

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> pos(60.0, 340.0);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * CV_PI);
    int compared = 0;
    for (int k = 0; k < 300; ++k) {
        // Fresh cost per edge (as every GrowPatch Problem has), so each call fills its own path cache.
        const NormalConstraintPlane cost(ngv, 0, /*w_normal=*/1.0, /*w_snap=*/0.0, nullptr, direction_aware);
        const double x = pos(rng), y = pos(rng), a = angle(rng);
        J p1[2] = {J(x, 0), J(y, 1)};
        J p2[2] = {J(x + 20.0 * std::cos(a), 2), J(y + 20.0 * std::sin(a), 3)};

        const J got = cost.calculate_normal_snapping_loss(p1, p2, grid, 0);
        const J want = referenceNormalLoss(p1, p2, grid, direction_aware, cost.roi_radius_, cost.query_radius_);
        CHECK(got.a == doctest::Approx(want.a).epsilon(1e-12));
        for (int d = 0; d < 4; ++d) {
            CHECK(got.v[d] == doctest::Approx(want.v[d]).epsilon(1e-9).scale(1e-12));
        }

        // The double path (residual-only evaluation) must agree with the Jet value.
        const double p1d[2] = {x, y};
        const double p2d[2] = {p2[0].a, p2[1].a};
        CHECK(cost.calculate_normal_snapping_loss(p1d, p2d, grid, 1) == doctest::Approx(want.a).epsilon(1e-12));
        compared += want.a != 0.0;
    }
    // The edges must actually see segments, or the comparison above proves nothing.
    CHECK(compared > 250);
    fs::remove_all(dir);
}

}  // namespace

TEST_CASE("normal loss keeps the per-segment formula's value and derivatives (sign-blind)")
{
    checkMatchesReference(false);
}

TEST_CASE("normal loss keeps the per-segment formula's value and derivatives (direction-aware)")
{
    checkMatchesReference(true);
}
