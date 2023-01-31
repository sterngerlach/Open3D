// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/Feature.h"

#include <Eigen/Dense>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace pipelines {
namespace registration {

static Eigen::Vector4d ComputePairFeatures(const Eigen::Vector3d &p1,
                                           const Eigen::Vector3d &n1,
                                           const Eigen::Vector3d &p2,
                                           const Eigen::Vector3d &n2) {
    Eigen::Vector4d result;
    Eigen::Vector3d dp2p1 = p2 - p1;
    result(3) = dp2p1.norm();
    if (result(3) == 0.0) {
        return Eigen::Vector4d::Zero();
    }
    auto n1_copy = n1;
    auto n2_copy = n2;
    double angle1 = n1_copy.dot(dp2p1) / result(3);
    double angle2 = n2_copy.dot(dp2p1) / result(3);
    if (acos(fabs(angle1)) > acos(fabs(angle2))) {
        n1_copy = n2;
        n2_copy = n1;
        dp2p1 *= -1.0;
        result(2) = -angle2;
    } else {
        result(2) = angle1;
    }
    auto v = dp2p1.cross(n1_copy);
    double v_norm = v.norm();
    if (v_norm == 0.0) {
        return Eigen::Vector4d::Zero();
    }
    v /= v_norm;
    auto w = n1_copy.cross(v);
    result(1) = v.dot(n2_copy);
    result(0) = atan2(w.dot(n2_copy), n1_copy.dot(n2_copy));
    return result;
}

static std::shared_ptr<Feature> ComputeSPFHFeature(
        const geometry::PointCloud &input,
        const geometry::KDTreeFlann &kdtree,
        const geometry::KDTreeSearchParam &search_param) {
    auto feature = std::make_shared<Feature>();
    feature->Resize(33, (int)input.points_.size());
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < (int)input.points_.size(); i++) {
        const auto &point = input.points_[i];
        const auto &normal = input.normals_[i];
        std::vector<int> indices;
        std::vector<double> distance2;
        if (kdtree.Search(point, search_param, indices, distance2) > 1) {
            // only compute SPFH feature when a point has neighbors
            double hist_incr = 100.0 / (double)(indices.size() - 1);
            for (size_t k = 1; k < indices.size(); k++) {
                // skip the point itself, compute histogram
                auto pf = ComputePairFeatures(point, normal,
                                              input.points_[indices[k]],
                                              input.normals_[indices[k]]);
                int h_index = (int)(floor(11 * (pf(0) + M_PI) / (2.0 * M_PI)));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index, i) += hist_incr;
                h_index = (int)(floor(11 * (pf(1) + 1.0) * 0.5));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index + 11, i) += hist_incr;
                h_index = (int)(floor(11 * (pf(2) + 1.0) * 0.5));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index + 22, i) += hist_incr;
            }
        }
    }
    return feature;
}

std::shared_ptr<Feature> ComputeFPFHFeature(
        const geometry::PointCloud &input,
        const geometry::KDTreeSearchParam
                &search_param /* = geometry::KDTreeSearchParamKNN()*/) {
    auto feature = std::make_shared<Feature>();
    feature->Resize(33, (int)input.points_.size());
    if (!input.HasNormals()) {
        utility::LogError(
                "[ComputeFPFHFeature] Failed because input point cloud has no "
                "normal.");
    }
    geometry::KDTreeFlann kdtree(input);
    auto spfh = ComputeSPFHFeature(input, kdtree, search_param);
    if (spfh == nullptr) {
        utility::LogError("Internal error: SPFH feature is nullptr.");
    }
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < (int)input.points_.size(); i++) {
        const auto &point = input.points_[i];
        std::vector<int> indices;
        std::vector<double> distance2;
        if (kdtree.Search(point, search_param, indices, distance2) > 1) {
            double sum[3] = {0.0, 0.0, 0.0};
            for (size_t k = 1; k < indices.size(); k++) {
                // skip the point itself
                double dist = distance2[k];
                if (dist == 0.0) continue;
                for (int j = 0; j < 33; j++) {
                    double val = spfh->data_(j, indices[k]) / dist;
                    sum[j / 11] += val;
                    feature->data_(j, i) += val;
                }
            }
            for (int j = 0; j < 3; j++)
                if (sum[j] != 0.0) sum[j] = 100.0 / sum[j];
            for (int j = 0; j < 33; j++) {
                feature->data_(j, i) *= sum[j / 11];
                // The commented line is the fpfh function in the paper.
                // But according to PCL implementation, it is skipped.
                // Our initial test shows that the full fpfh function in the
                // paper seems to be better than PCL implementation. Further
                // test required.
                feature->data_(j, i) += spfh->data_(j, i);
            }
        }
    }
    return feature;
}

// Compute SPFH features for a point cloud (useful for profiling)
// Eigen uses the column-major matrices
// `point_cloud` is of size [3, N]
// `knn_indices` is of size [K, N]
// Output is of size [33, N]
// N is the number of points
// K is the number of neighbors
std::shared_ptr<Feature> ComputeSPFHFeatureEx(
    const geometry::PointCloud& point_cloud,
    const Eigen::MatrixXi& knn_indices)
{
    const int num_points = static_cast<int>(point_cloud.points_.size());

    if (!point_cloud.HasNormals()) {
        utility::LogError("Point cloud has no normal");
        return nullptr;
    }

    if (knn_indices.cols() != num_points) {
        utility::LogError("Number of points in `knn_indices` should be the "
                          "same as the number of points in `point_cloud`");
        return nullptr;
    }

    if (knn_indices.rows() <= 1) {
        utility::LogError("Number of neighbors should be greater than 1");
        return nullptr;
    }

    auto feature = std::make_shared<Feature>();
    feature->Resize(33, num_points);

    const double hist_incr = 100.0 / static_cast<double>(
        knn_indices.rows() - 1);

#pragma omp parallel for schedule(static) \
    num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < num_points; ++i) {
        const auto& point = point_cloud.points_[i];
        const auto& normal = point_cloud.normals_[i];
        // Skip the point itself
        for (int k = 1; k < knn_indices.rows(); ++k) {
            const auto pair_feature = ComputePairFeatures(
                point, normal,
                point_cloud.points_[knn_indices(k, i)],
                point_cloud.normals_[knn_indices(k, i)]);

            int h_index = static_cast<int>(std::floor(
                11 * (pair_feature(0) + M_PI) / (2.0 * M_PI)));
            h_index = std::max(0, std::min(10, h_index));
            feature->data_(h_index, i) += hist_incr;

            h_index = static_cast<int>(std::floor(
                11 * (pair_feature(1) + 1.0) * 0.5));
            h_index = std::max(0, std::min(10, h_index));
            feature->data_(h_index + 11, i) += hist_incr;

            h_index = static_cast<int>(std::floor(
                11 * (pair_feature(2) + 1.0) * 0.5));
            h_index = std::max(0, std::min(10, h_index));
            feature->data_(h_index + 22, i) += hist_incr;
        }
    }

    return feature;
}

// Compute FPFH features for a point cloud (useful for profiling)
// Eigen uses the column-major matrices
// `spfh_features` is of size [33, N]
// `knn_indices` is of size [K, N]
// `knn_distances` is of size [K, N]
// Output is of size [33, N]
// N is the number of points
// K is the number of neighbors
std::shared_ptr<Feature> ComputeFPFHFeatureEx(
    const Feature& spfh_features,
    const Eigen::MatrixXi& knn_indices,
    const Eigen::MatrixXd& knn_distances)
{
    const int num_points = static_cast<int>(spfh_features.Num());

    if (spfh_features.Dimension() != 33) {
        utility::LogError("Number of SPFH feature dimensions should be 33");
        return nullptr;
    }

    if (knn_indices.cols() != num_points) {
        utility::LogError("Number of points in `knn_indices` should be the "
                          "same as the number of points in `point_cloud`");
        return nullptr;
    }

    if (knn_indices.rows() <= 1) {
        utility::LogError("Number of neighbors should be greater than 1");
        return nullptr;
    }

    if (knn_distances.cols() != num_points) {
        utility::LogError("Number of points in `knn_distances` should be the "
                          "same as the number of points in `point_cloud`");
        return nullptr;
    }

    if (knn_distances.rows() != knn_indices.rows()) {
        utility::LogError("Number of neighbors in `knn_distances` should be "
                          "the same as the number of neighbors in "
                          "`knn_indices`");
        return nullptr;
    }

    auto feature = std::make_shared<Feature>();
    feature->Resize(33, num_points);

#pragma omp parallel for schedule(static) \
    num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < num_points; ++i) {
        double sum[3] = { 0.0, 0.0, 0.0 };

        // Skip the point itself
        for (int k = 1; k < knn_indices.rows(); ++k) {
            const double dist = knn_distances(k, i);
            if (dist == 0.0)
                continue;

            const int knn_idx = knn_indices(k, i);
            for (int j = 0; j < 33; ++j) {
                const double spfh_val = spfh_features.data_(j, knn_idx);
                const double val = spfh_val / dist;
                sum[j / 11] += val;
                feature->data_(j, i) += val;
            }
        }

        for (int j = 0; j < 3; ++j)
            if (sum[j] != 0.0)
                sum[j] = 100.0 / sum[j];

        for (int j = 0; j < 33; ++j) {
            feature->data_(j, i) *= sum[j / 11];
            feature->data_(j, i) += spfh_features.data_(j, i);
        }
    }

    return feature;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
