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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)
#endif

#include "open3d/geometry/KDTreeFlann.h"

#include <nanoflann.hpp>

#include "open3d/geometry/HalfEdgeTriangleMesh.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace geometry {

KDTreeFlann::KDTreeFlann() {}

KDTreeFlann::KDTreeFlann(const Eigen::MatrixXd &data) { SetMatrixData(data); }

KDTreeFlann::KDTreeFlann(const Geometry &geometry) { SetGeometry(geometry); }

KDTreeFlann::KDTreeFlann(const pipelines::registration::Feature &feature) {
    SetFeature(feature);
}

KDTreeFlann::~KDTreeFlann() {}

bool KDTreeFlann::SetMatrixData(const Eigen::MatrixXd &data) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            data.data(), data.rows(), data.cols()));
}

bool KDTreeFlann::SetGeometry(const Geometry &geometry) {
    switch (geometry.GetGeometryType()) {
        case Geometry::GeometryType::PointCloud:
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                    (const double *)((const PointCloud &)geometry)
                            .points_.data(),
                    3, ((const PointCloud &)geometry).points_.size()));
        case Geometry::GeometryType::TriangleMesh:
        case Geometry::GeometryType::HalfEdgeTriangleMesh:
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                    (const double *)((const TriangleMesh &)geometry)
                            .vertices_.data(),
                    3, ((const TriangleMesh &)geometry).vertices_.size()));
        case Geometry::GeometryType::Image:
        case Geometry::GeometryType::Unspecified:
        default:
            utility::LogWarning(
                    "[KDTreeFlann::SetGeometry] Unsupported Geometry type.");
            return false;
    }
}

bool KDTreeFlann::SetFeature(const pipelines::registration::Feature &feature) {
    return SetMatrixData(feature.data_);
}

template <typename T>
int KDTreeFlann::Search(const T &query,
                        const KDTreeSearchParam &param,
                        std::vector<int> &indices,
                        std::vector<double> &distance2) const {
    switch (param.GetSearchType()) {
        case KDTreeSearchParam::SearchType::Knn:
            return SearchKNN(query, ((const KDTreeSearchParamKNN &)param).knn_,
                             indices, distance2);
        case KDTreeSearchParam::SearchType::Radius:
            return SearchRadius(
                    query, ((const KDTreeSearchParamRadius &)param).radius_,
                    indices, distance2);
        case KDTreeSearchParam::SearchType::Hybrid:
            return SearchHybrid(
                    query, ((const KDTreeSearchParamHybrid &)param).radius_,
                    ((const KDTreeSearchParamHybrid &)param).max_nn_, indices,
                    distance2);
        default:
            return -1;
    }
    return -1;
}

template <typename T>
int KDTreeFlann::SearchKNN(const T &query,
                           int knn,
                           std::vector<int> &indices,
                           std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Other flann::Index::knnSearch() implementations lose performance due to
    // memory allocation/deallocation.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_ || knn < 0) {
        return -1;
    }
    indices.resize(knn);
    distance2.resize(knn);
    std::vector<Eigen::Index> indices_eigen(knn);
    int k = nanoflann_index_->index->knnSearch(
            query.data(), knn, indices_eigen.data(), distance2.data());
    indices.resize(k);
    distance2.resize(k);
    std::copy_n(indices_eigen.begin(), k, indices.begin());
    return k;
}

template <typename T>
int KDTreeFlann::SearchRadius(const T &query,
                              double radius,
                              std::vector<int> &indices,
                              std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Since max_nn is not given, we let flann to do its own memory management.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory management and CPU caching.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_) {
        return -1;
    }
    std::vector<std::pair<Eigen::Index, double>> indices_dists;
    int k = nanoflann_index_->index->radiusSearch(
            query.data(), radius * radius, indices_dists,
            nanoflann::SearchParams(-1, 0.0));
    indices.resize(k);
    distance2.resize(k);
    for (int i = 0; i < k; ++i) {
        indices[i] = indices_dists[i].first;
        distance2[i] = indices_dists[i].second;
    }
    return k;
}

template <typename T>
int KDTreeFlann::SearchHybrid(const T &query,
                              double radius,
                              int max_nn,
                              std::vector<int> &indices,
                              std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // It is also the recommended setting for search.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory allocation/deallocation.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_ || max_nn < 0) {
        return -1;
    }
    distance2.resize(max_nn);
    std::vector<Eigen::Index> indices_eigen(max_nn);
    int k = nanoflann_index_->index->knnSearch(
            query.data(), max_nn, indices_eigen.data(), distance2.data());
    k = std::distance(distance2.begin(),
                      std::lower_bound(distance2.begin(), distance2.begin() + k,
                                       radius * radius));
    indices.resize(k);
    distance2.resize(k);
    std::copy_n(indices_eigen.begin(), k, indices.begin());
    return k;
}

// Batched K-nearest neighbor search for efficiency
// Eigen uses the column-major matrices
// `queries` is of size [D, N]
// `indices` is of size [K, N]
// `distance2` is of size [K, N]
// N is the number of queries
// D is the number of dimensions
// K is the number of neighbors
template <typename T>
bool KDTreeFlann::BatchedSearchKNN(const T& queries,
                                   const int knn,
                                   Eigen::MatrixXi& indices,
                                   Eigen::MatrixXd& distance2) const
{
    using IndexVector = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    const auto data_dim = static_cast<std::size_t>(queries.rows());

    if (this->data_.empty() || this->dataset_size_ <= 0 ||
        data_dim != this->dimension_ || knn < 0) {
        return false;
    }

    indices.resize(knn, queries.cols());
    distance2.resize(knn, queries.cols());

    std::vector<Eigen::Index> query_indices;
    std::vector<double> query_distance2;
    query_indices.resize(knn);
    query_distance2.resize(knn);

    for (int i = 0; i < queries.cols(); ++i) {
        const int k = this->nanoflann_index_->index->knnSearch(
            queries.col(i).data(), knn,
            query_indices.data(), query_distance2.data());
        indices.col(i).head(k) = Eigen::Map<IndexVector>(
            query_indices.data(), k).cast<int>();
        distance2.col(i).head(k) = Eigen::Map<Eigen::VectorXd>(
            query_distance2.data(), k);
    }

    return true;
}

// Batched K-nearest neighbor search for efficiency (with OpenMP)
// Eigen uses the column-major matrices
// `queries` is of size [D, N]
// `indices` is of size [K, N]
// `distance2` is of size [K, N]
// N is the number of queries
// D is the number of dimensions
// K is the number of neighbors
template <typename T>
bool KDTreeFlann::BatchedSearchKNNOpenMP(const T& queries,
                                         const int knn,
                                         Eigen::MatrixXi& indices,
                                         Eigen::MatrixXd& distance2) const
{
    using IndexVector = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    const auto data_dim = static_cast<std::size_t>(queries.rows());

    if (this->data_.empty() || this->dataset_size_ <= 0 ||
        data_dim != this->dimension_ || knn < 0) {
        return false;
    }

    indices.resize(knn, queries.cols());
    distance2.resize(knn, queries.cols());

    std::vector<Eigen::Index> query_indices;
    std::vector<double> query_distance2;
    query_indices.resize(knn);
    query_distance2.resize(knn);

    // Copy constructor is called when initializing `query_indices` and
    // `query_distance2` in each thread

#pragma omp parallel for schedule(static) \
    num_threads(utility::EstimateMaxThreads()) \
    firstprivate(query_indices, query_distance2)
    for (int i = 0; i < queries.cols(); ++i) {
        const int k = this->nanoflann_index_->index->knnSearch(
            queries.col(i).data(), knn,
            query_indices.data(), query_distance2.data());
        indices.col(i).head(k) = Eigen::Map<IndexVector>(
            query_indices.data(), k).cast<int>();
        distance2.col(i).head(k) = Eigen::Map<Eigen::VectorXd>(
            query_distance2.data(), k);
    }

    return true;
}

bool KDTreeFlann::SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
    dimension_ = data.rows();
    dataset_size_ = data.cols();
    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning("[KDTreeFlann::SetRawData] Failed due to no data.");
        return false;
    }
    data_.resize(dataset_size_ * dimension_);
    memcpy(data_.data(), data.data(),
           dataset_size_ * dimension_ * sizeof(double));
    data_interface_.reset(new Eigen::Map<const Eigen::MatrixXd>(data));
    nanoflann_index_.reset(
            new KDTree_t(dimension_, std::cref(*data_interface_), 15));
    nanoflann_index_->index->buildIndex();
    return true;
}

template int KDTreeFlann::Search<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTreeFlann::Search<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template bool KDTreeFlann::BatchedSearchKNN<Eigen::Matrix3Xd>(
    const Eigen::Matrix3Xd& queries,
    const int knn,
    Eigen::MatrixXi& indices,
    Eigen::MatrixXd& distance2) const;
template bool KDTreeFlann::BatchedSearchKNN<Eigen::MatrixXd>(
    const Eigen::MatrixXd& queries,
    const int knn,
    Eigen::MatrixXi& indices,
    Eigen::MatrixXd& distance2) const;

template bool KDTreeFlann::BatchedSearchKNNOpenMP<Eigen::Matrix3Xd>(
    const Eigen::Matrix3Xd& queries,
    const int knn,
    Eigen::MatrixXi& indices,
    Eigen::MatrixXd& distance2) const;
template bool KDTreeFlann::BatchedSearchKNNOpenMP<Eigen::MatrixXd>(
    const Eigen::MatrixXd& queries,
    const int knn,
    Eigen::MatrixXi& indices,
    Eigen::MatrixXd& distance2) const;

}  // namespace geometry
}  // namespace open3d

#ifdef _MSC_VER
#pragma warning(pop)
#endif
