#pragma once

#include <algorithm>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <kiss_matcher/points/fast_floor.hpp>
#include <kiss_matcher/points/point_cloud.hpp>
#include <kiss_matcher/points/traits.hpp>
#include <kiss_matcher/points/vector3i_hash.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace kiss_matcher {

// Finds the start index of every run of equal keys in a sorted array, e.g. for
// keys = [A, A, B, B, B, C], returns [0, 2, 5] (plus `keys.size()` as an implicit
// end sentinel). Each run corresponds to exactly one output voxel, so partitioning
// work by run (rather than by a fixed block size) guarantees that no voxel is ever
// split across two parallel tasks, and that iterating runs in order reproduces a
// deterministic output order regardless of TBB scheduling.
template <typename KeyFn>
std::vector<size_t> FindRunStarts(size_t num_elements, KeyFn &&key_of) {
  std::vector<size_t> run_starts;
  if (num_elements == 0) {
    return run_starts;
  }

  std::vector<uint8_t> is_run_start(num_elements, 0);
  is_run_start[0] = 1;
  tbb::parallel_for(tbb::blocked_range<size_t>(1, num_elements, 4096),
                    [&](const tbb::blocked_range<size_t> &range) {
                      for (size_t i = range.begin(); i != range.end(); i++) {
                        if (key_of(i - 1) != key_of(i)) {
                          is_run_start[i] = 1;
                        }
                      }
                    });

  run_starts.reserve(num_elements);
  for (size_t i = 0; i < num_elements; i++) {
    if (is_run_start[i]) {
      run_starts.push_back(i);
    }
  }
  return run_starts;
}

/**
 * @brief Voxel grid downsampling with TBB backend.
 * @param points     Input points
 * @param leaf_size  Downsampling resolution
 * @return           Downsampled points
 */
template <typename InputPointCloud, typename OutputPointCloud = InputPointCloud>
std::shared_ptr<OutputPointCloud> VoxelgridSampling(const InputPointCloud& points,
                                                    const double leaf_size) {
  if (traits::size(points) == 0) {
    return std::make_shared<OutputPointCloud>();
  }

  const double inv_leaf_size = 1.0 / leaf_size;

  constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
  constexpr int coord_bit_size          = 21;
  constexpr size_t coord_bit_mask       = (1 << 21) - 1;
  constexpr int coord_offset            = 1 << (coord_bit_size - 1);

  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(traits::size(points));
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, traits::size(points), 64),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
          const Eigen::Array4i coord =
              fast_floor(traits::point(points, i) * inv_leaf_size) + coord_offset;
          if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
            std::cerr << "warning: voxel coord is out of range!!" << std::endl;
            coord_pt[i] = {invalid_coord, i};
            continue;
          }

          const std::uint64_t bits =
              (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |
              (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |
              (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
          coord_pt[i] = {bits, i};
        }
      });

  tbb::parallel_sort(coord_pt,
                     [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  // Invalid (out-of-range) coords all share the key `invalid_coord`, so they form a
  // single trailing run after the sort. That run is dropped entirely rather than
  // averaged, since those points don't belong to a real voxel.
  const size_t num_coords = coord_pt.size();
  const size_t num_valid_coords =
      num_coords > 0 && coord_pt.back().first == invalid_coord
          ? std::lower_bound(coord_pt.begin(),
                             coord_pt.end(),
                             std::make_pair(invalid_coord, size_t(0)),
                             [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }) -
                coord_pt.begin()
          : num_coords;

  const std::vector<size_t> run_starts =
      FindRunStarts(num_valid_coords, [&](size_t i) { return coord_pt[i].first; });
  const size_t num_voxels = run_starts.size();

  auto downsampled = std::make_shared<OutputPointCloud>();
  traits::resize(*downsampled, num_voxels);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_voxels),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t voxel = range.begin(); voxel != range.end(); voxel++) {
                        const size_t begin = run_starts[voxel];
                        const size_t end =
                            (voxel + 1 < num_voxels) ? run_starts[voxel + 1] : num_valid_coords;

                        Eigen::Vector4d sum_pt = traits::point(points, coord_pt[begin].second);
                        for (size_t i = begin + 1; i != end; i++) {
                          sum_pt += traits::point(points, coord_pt[i].second);
                        }
                        traits::set_point(*downsampled, voxel, sum_pt / sum_pt.w());
                      }
                    });

  return downsampled;
}

inline std::vector<Eigen::Vector3f> VoxelgridSampling(const std::vector<Eigen::Vector3f>& points,
                                                      const double leaf_size) {
  if (points.empty()) {
    return {};
  }

  size_t num_raw_points      = points.size();
  const double inv_leaf_size = 1.0 / leaf_size;

  constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
  constexpr int coord_bit_size          = 21;
  constexpr size_t coord_bit_mask       = (1 << 21) - 1;
  constexpr int coord_offset            = 1 << (coord_bit_size - 1);

  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(num_raw_points);
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_raw_points, 64),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
          const Eigen::Array3i coord =
              fast_floor_vector3f(points[i] * inv_leaf_size) + coord_offset;
          if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
            std::cerr << "warning: voxel coord is out of range!!" << std::endl;
            coord_pt[i] = {invalid_coord, i};
            continue;
          }

          const std::uint64_t bits =
              (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |
              (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |
              (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
          coord_pt[i] = {bits, i};
        }
      });

  tbb::parallel_sort(coord_pt,
                     [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  // Invalid (out-of-range) coords all share the key `invalid_coord`, so they form a
  // single trailing run after the sort. That run is dropped entirely rather than
  // averaged, since those points don't belong to a real voxel.
  const size_t num_coords = coord_pt.size();
  const size_t num_valid_coords =
      num_coords > 0 && coord_pt.back().first == invalid_coord
          ? std::lower_bound(coord_pt.begin(),
                             coord_pt.end(),
                             std::make_pair(invalid_coord, size_t(0)),
                             [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }) -
                coord_pt.begin()
          : num_coords;

  const std::vector<size_t> run_starts =
      FindRunStarts(num_valid_coords, [&](size_t i) { return coord_pt[i].first; });
  const size_t num_voxels = run_starts.size();

  std::vector<Eigen::Vector3f> downsampled;
  downsampled.resize(num_voxels);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_voxels),
                    [&](const tbb::blocked_range<size_t>& range) {
                      for (size_t voxel = range.begin(); voxel != range.end(); voxel++) {
                        const size_t begin = run_starts[voxel];
                        const size_t end =
                            (voxel + 1 < num_voxels) ? run_starts[voxel + 1] : num_valid_coords;

                        Eigen::Vector3d sum_pt = points[coord_pt[begin].second].cast<double>();
                        for (size_t i = begin + 1; i != end; i++) {
                          sum_pt += points[coord_pt[i].second].cast<double>();
                        }
                        downsampled[voxel] = (sum_pt / static_cast<double>(end - begin)).cast<float>();
                      }
                    });

  return downsampled;
}

}  // namespace kiss_matcher
