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

namespace {

constexpr uint8_t kPrefetchAbsent = 0;
constexpr uint8_t kPrefetchPending = 1;
constexpr uint8_t kPrefetchResident = 2;

constexpr int64_t kRawCommands = 0;
constexpr int64_t kUniqueCommands = 1;
constexpr int64_t kStaleCommands = 2;
constexpr int64_t kPendingCommands = 3;
constexpr int64_t kResidentCommands = 4;
constexpr int64_t kSelectedClusters = 5;
constexpr int64_t kSelectedPages = 6;
constexpr int64_t kBudgetStops = 7;
constexpr int64_t kNumStatistics = 8;

void validate_cpu_contiguous(const torch::Tensor& tensor, at::ScalarType dtype,
                             const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must reside on CPU");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, torch::Tensor>
retrospec_plan_prefetch_admissions(
    const std::vector<torch::Tensor>& cluster_id_records,
    const std::vector<torch::Tensor>& position_records,
    const std::vector<torch::Tensor>& count_records,
    const std::vector<int64_t>& num_groups,
    const std::vector<int64_t>& num_ranks,
    const std::vector<torch::Tensor>& descriptor_page_ids,
    const std::vector<torch::Tensor>& descriptor_page_counts,
    const std::vector<torch::Tensor>& descriptor_group_ids,
    const std::vector<torch::Tensor>& resident_states,
    const std::vector<int64_t>& page_capacities) {
  const int64_t num_records = static_cast<int64_t>(cluster_id_records.size());
  auto require_record_count = [num_records](size_t size, const char* name) {
    TORCH_CHECK(static_cast<int64_t>(size) == num_records, name,
                " record count differs from cluster IDs");
  };

  require_record_count(position_records.size(), "Position");
  require_record_count(count_records.size(), "Count");
  require_record_count(num_groups.size(), "Group");
  require_record_count(num_ranks.size(), "Rank");
  require_record_count(descriptor_page_ids.size(), "Page descriptor");
  require_record_count(descriptor_page_counts.size(), "Page-count descriptor");
  require_record_count(descriptor_group_ids.size(), "Group descriptor");
  require_record_count(resident_states.size(), "Resident-state");
  require_record_count(page_capacities.size(), "Page-capacity");

  const auto int64_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  std::vector<torch::Tensor> selected_cluster_ids(num_records);
  std::vector<torch::Tensor> selected_page_ids(num_records);
  std::vector<torch::Tensor> selected_staging_page_ids(num_records);
  std::vector<torch::Tensor> selected_unique_page_ids(num_records);
  std::vector<torch::Tensor> selected_group_ids(num_records);
  torch::Tensor statistics =
      torch::zeros({num_records, kNumStatistics}, int64_options);

  at::parallel_for(0, num_records, 1, [&](int64_t begin, int64_t end) {
    for (int64_t record_index = begin; record_index < end; ++record_index) {
      const torch::Tensor& cluster_ids = cluster_id_records[record_index];
      const torch::Tensor& positions = position_records[record_index];
      const torch::Tensor& count = count_records[record_index];
      const torch::Tensor& page_ids = descriptor_page_ids[record_index];
      const torch::Tensor& page_counts = descriptor_page_counts[record_index];
      const torch::Tensor& group_ids = descriptor_group_ids[record_index];
      const torch::Tensor& states = resident_states[record_index];

      validate_cpu_contiguous(cluster_ids, at::kLong, "Cluster IDs");
      validate_cpu_contiguous(positions, at::kLong, "Positions");
      validate_cpu_contiguous(count, at::kInt, "Command count");
      validate_cpu_contiguous(page_ids, at::kLong, "Descriptor page IDs");
      validate_cpu_contiguous(page_counts, at::kInt, "Descriptor page counts");
      validate_cpu_contiguous(group_ids, at::kLong, "Descriptor group IDs");
      validate_cpu_contiguous(states, at::kByte, "Resident states");

      TORCH_CHECK(cluster_ids.sizes() == positions.sizes(),
                  "Prefetch positions must match cluster IDs");
      TORCH_CHECK(count.numel() == 1,
                  "Each prefetch count must contain one value");
      TORCH_CHECK(page_ids.dim() == 2,
                  "Descriptor page IDs must be two-dimensional");
      TORCH_CHECK(
          page_counts.dim() == 1 && page_counts.size(0) == page_ids.size(0),
          "Descriptor page-count shape is invalid");
      TORCH_CHECK(group_ids.sizes() == page_counts.sizes(),
                  "Descriptor group IDs have an invalid shape");
      TORCH_CHECK(states.sizes() == page_counts.sizes(),
                  "Resident-state shape differs from descriptors");
      TORCH_CHECK(num_groups[record_index] > 0 && num_ranks[record_index] > 0,
                  "Prefetch layout must be positive");
      TORCH_CHECK(page_capacities[record_index] >= 0,
                  "Prefetch page capacity must be non-negative");

      const int64_t valid_count = count.data_ptr<int32_t>()[0];
      TORCH_CHECK(valid_count >= 0 && valid_count <= cluster_ids.numel(),
                  "Prefetch count exceeds command capacity");
      int64_t* record_stats =
          statistics.data_ptr<int64_t>() + record_index * kNumStatistics;
      record_stats[kRawCommands] = valid_count;

      const auto* cluster_data = cluster_ids.data_ptr<int64_t>();
      const auto* position_data = positions.data_ptr<int64_t>();
      const int64_t layout_size =
          num_groups[record_index] * num_ranks[record_index];
      std::vector<std::pair<int64_t, int64_t>> commands;
      commands.reserve(valid_count);

      for (int64_t command_index = 0; command_index < valid_count;
           ++command_index) {
        const int64_t handle = cluster_data[command_index];
        const int64_t position = position_data[command_index];
        TORCH_CHECK(handle >= 0, "Prefetch command contains an invalid handle");
        TORCH_CHECK(position >= 0 && position < layout_size,
                    "Prefetch command position is out of range");
        const int64_t group_index = position / num_ranks[record_index];
        const int64_t rank = position % num_ranks[record_index];
        const int64_t priority = rank * num_groups[record_index] + group_index;
        commands.emplace_back(priority, handle);
      }

      std::stable_sort(commands.begin(), commands.end(),
                       [](const auto& lhs, const auto& rhs) {
                         return lhs.first < rhs.first;
                       });
      std::vector<int64_t> ordered_handles;
      ordered_handles.reserve(commands.size());
      std::unordered_set<int64_t> observed_handles;
      for (const auto& command : commands) {
        if (observed_handles.insert(command.second).second) {
          ordered_handles.push_back(command.second);
        }
      }
      record_stats[kUniqueCommands] = ordered_handles.size();

      const auto* descriptor_counts = page_counts.data_ptr<int32_t>();
      const auto* descriptor_groups = group_ids.data_ptr<int64_t>();
      const auto* descriptor_pages = page_ids.data_ptr<int64_t>();
      const auto* resident_state_data = states.data_ptr<uint8_t>();
      const int64_t descriptor_capacity = page_counts.numel();
      const int64_t page_width = page_ids.size(1);
      std::vector<int64_t> output_handles;
      std::vector<int64_t> output_groups;
      std::vector<int64_t> output_pages;
      std::vector<int64_t> output_staging_pages;
      std::vector<int64_t> unique_pages;
      std::unordered_set<int64_t> observed_pages;
      int64_t selected_page_count = 0;

      for (const int64_t handle : ordered_handles) {
        if (handle >= descriptor_capacity) {
          ++record_stats[kStaleCommands];
          continue;
        }
        const uint8_t state = resident_state_data[handle];
        if (state == kPrefetchPending) {
          ++record_stats[kPendingCommands];
          continue;
        }
        if (state == kPrefetchResident) {
          ++record_stats[kResidentCommands];
          continue;
        }
        TORCH_CHECK(state == kPrefetchAbsent,
                    "Resident state contains an invalid value");

        const int64_t cluster_page_count = descriptor_counts[handle];
        const int64_t group_id = descriptor_groups[handle];
        if (cluster_page_count <= 0 || group_id < 0 ||
            cluster_page_count > page_width) {
          ++record_stats[kStaleCommands];
          continue;
        }
        if (selected_page_count + cluster_page_count >
            page_capacities[record_index]) {
          ++record_stats[kBudgetStops];
          break;
        }

        output_handles.push_back(handle);
        output_groups.push_back(group_id);
        const int64_t row_start = handle * page_width;
        for (int64_t page_index = 0; page_index < page_width; ++page_index) {
          if (page_index >= cluster_page_count) {
            output_pages.push_back(-1);
            output_staging_pages.push_back(-1);
            continue;
          }
          const int64_t page_id = descriptor_pages[row_start + page_index];
          TORCH_CHECK(page_id >= 0,
                      "Valid descriptor contains an invalid page ID");
          TORCH_CHECK(observed_pages.insert(page_id).second,
                      "Logical page belongs to multiple selected clusters");
          output_pages.push_back(page_id);
          output_staging_pages.push_back(unique_pages.size());
          unique_pages.push_back(page_id);
        }
        selected_page_count += cluster_page_count;
      }

      const int64_t selected_count = output_handles.size();
      record_stats[kSelectedClusters] = selected_count;
      record_stats[kSelectedPages] = selected_page_count;
      torch::Tensor output_handle_tensor =
          torch::empty({selected_count}, int64_options);
      torch::Tensor output_group_tensor =
          torch::empty({selected_count}, int64_options);
      torch::Tensor output_page_tensor =
          torch::full({selected_count, page_width}, -1, int64_options);
      torch::Tensor output_staging_tensor =
          torch::full({selected_count, page_width}, -1, int64_options);
      torch::Tensor unique_page_tensor =
          torch::empty({selected_page_count}, int64_options);

      std::copy(output_handles.begin(), output_handles.end(),
                output_handle_tensor.data_ptr<int64_t>());
      std::copy(output_groups.begin(), output_groups.end(),
                output_group_tensor.data_ptr<int64_t>());
      std::copy(output_pages.begin(), output_pages.end(),
                output_page_tensor.data_ptr<int64_t>());
      std::copy(output_staging_pages.begin(), output_staging_pages.end(),
                output_staging_tensor.data_ptr<int64_t>());
      std::copy(unique_pages.begin(), unique_pages.end(),
                unique_page_tensor.data_ptr<int64_t>());

      selected_cluster_ids[record_index] = std::move(output_handle_tensor);
      selected_page_ids[record_index] = std::move(output_page_tensor);
      selected_staging_page_ids[record_index] =
          std::move(output_staging_tensor);
      selected_unique_page_ids[record_index] = std::move(unique_page_tensor);
      selected_group_ids[record_index] = std::move(output_group_tensor);
    }
  });

  return std::make_tuple(
      std::move(selected_cluster_ids), std::move(selected_page_ids),
      std::move(selected_staging_page_ids), std::move(selected_unique_page_ids),
      std::move(selected_group_ids), statistics);
}
