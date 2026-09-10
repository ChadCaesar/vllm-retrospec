// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/Parallel.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

std::tuple<std::vector<torch::Tensor>, torch::Tensor>
retrospec_order_prefetch_misses(
    const std::vector<torch::Tensor>& cluster_id_records,
    const std::vector<torch::Tensor>& position_records,
    const std::vector<torch::Tensor>& count_records,
    const std::vector<int64_t>& num_groups,
    const std::vector<int64_t>& num_ranks) {
  const int64_t num_records = static_cast<int64_t>(cluster_id_records.size());
  TORCH_CHECK(position_records.size() == cluster_id_records.size(),
              "Prefetch position-record count differs from cluster IDs");
  TORCH_CHECK(count_records.size() == cluster_id_records.size(),
              "Prefetch count-record count differs from cluster IDs");
  TORCH_CHECK(num_groups.size() == cluster_id_records.size(),
              "Prefetch group-count record count differs from cluster IDs");
  TORCH_CHECK(num_ranks.size() == cluster_id_records.size(),
              "Prefetch rank-count record count differs from cluster IDs");

  std::vector<int64_t> raw_counts(num_records);
  std::vector<std::vector<std::pair<int64_t, int64_t>>> commands(num_records);
  for (int64_t record_index = 0; record_index < num_records; ++record_index) {
    const torch::Tensor& cluster_ids = cluster_id_records[record_index];
    const torch::Tensor& positions = position_records[record_index];
    const torch::Tensor& count = count_records[record_index];

    TORCH_CHECK(cluster_ids.device().is_cpu(),
                "Prefetch cluster IDs must reside on CPU");
    TORCH_CHECK(positions.device().is_cpu(),
                "Prefetch positions must reside on CPU");
    TORCH_CHECK(count.device().is_cpu(), "Prefetch counts must reside on CPU");
    TORCH_CHECK(cluster_ids.scalar_type() == at::kLong,
                "Prefetch cluster IDs must use int64");
    TORCH_CHECK(positions.scalar_type() == at::kLong,
                "Prefetch positions must use int64");
    TORCH_CHECK(count.scalar_type() == at::kInt,
                "Prefetch counts must use int32");
    TORCH_CHECK(cluster_ids.is_contiguous() && positions.is_contiguous() &&
                    count.is_contiguous(),
                "Prefetch command records must be contiguous");
    TORCH_CHECK(cluster_ids.sizes() == positions.sizes(),
                "Prefetch positions must match cluster IDs");
    TORCH_CHECK(count.numel() == 1,
                "Each prefetch count record must contain one value");
    TORCH_CHECK(num_groups[record_index] > 0 && num_ranks[record_index] > 0,
                "Prefetch group and rank counts must be positive");

    const int64_t valid_count = count.data_ptr<int32_t>()[0];
    TORCH_CHECK(valid_count >= 0 && valid_count <= cluster_ids.numel(),
                "Prefetch count exceeds command-record capacity");
    raw_counts[record_index] = valid_count;
    commands[record_index].reserve(valid_count);

    const auto* cluster_id_data = cluster_ids.data_ptr<int64_t>();
    const auto* position_data = positions.data_ptr<int64_t>();
    const int64_t layout_size =
        num_groups[record_index] * num_ranks[record_index];
    for (int64_t command_index = 0; command_index < valid_count;
         ++command_index) {
      const int64_t cluster_id = cluster_id_data[command_index];
      const int64_t position = position_data[command_index];
      TORCH_CHECK(cluster_id >= 0,
                  "Prefetch command contains an invalid cluster handle");
      TORCH_CHECK(position >= 0 && position < layout_size,
                  "Prefetch command position is outside its layout");

      const int64_t group_index = position / num_ranks[record_index];
      const int64_t rank = position % num_ranks[record_index];
      const int64_t priority = rank * num_groups[record_index] + group_index;
      commands[record_index].emplace_back(priority, cluster_id);
    }
  }

  std::vector<std::vector<int64_t>> ordered_ids(num_records);
  at::parallel_for(0, num_records, 1, [&](int64_t begin, int64_t end) {
    for (int64_t record_index = begin; record_index < end; ++record_index) {
      auto& record_commands = commands[record_index];
      std::stable_sort(record_commands.begin(), record_commands.end(),
                       [](const auto& lhs, const auto& rhs) {
                         return lhs.first < rhs.first;
                       });

      auto& output = ordered_ids[record_index];
      output.reserve(record_commands.size());
      std::unordered_set<int64_t> observed;
      for (const auto& command : record_commands) {
        if (observed.insert(command.second).second) {
          output.push_back(command.second);
        }
      }
    }
  });

  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  std::vector<torch::Tensor> outputs;
  outputs.reserve(num_records);
  for (const auto& record : ordered_ids) {
    torch::Tensor output =
        torch::empty({static_cast<int64_t>(record.size())}, options);
    if (!record.empty()) {
      std::copy(record.begin(), record.end(), output.data_ptr<int64_t>());
    }
    outputs.push_back(std::move(output));
  }

  torch::Tensor raw_count_tensor = torch::empty({num_records}, options);
  if (!raw_counts.empty()) {
    std::copy(raw_counts.begin(), raw_counts.end(),
              raw_count_tensor.data_ptr<int64_t>());
  }
  return std::make_tuple(std::move(outputs), raw_count_tensor);
}
