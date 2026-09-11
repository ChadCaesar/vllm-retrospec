// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kPageOffsetBits = 32;
constexpr int64_t kPageOffsetMask = (int64_t{1} << kPageOffsetBits) - 1;

struct CompactRange {
  int64_t head_index;
  int64_t slab_id;
  int64_t token_offset;
  int64_t token_count;
  int64_t head_token_offset;
};

void validate_integer_tensor(const torch::Tensor& tensor, const char* name,
                             int64_t expected_dimensions) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must reside on CPU");
  TORCH_CHECK(tensor.dim() == expected_dimensions, name,
              " has an invalid rank");
  TORCH_CHECK(
      tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong,
      name, " must use int32 or int64");
}

torch::Tensor to_contiguous_int64(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == at::kLong && tensor.is_contiguous()) {
    return tensor;
  }
  return tensor.to(torch::kInt64).contiguous();
}

template <typename Function>
void parallel_for_heads(int64_t num_heads, int64_t num_workers,
                        Function&& function) {
  if (num_heads == 0) {
    return;
  }

  const int64_t worker_count =
      std::max<int64_t>(1, std::min(num_heads, num_workers));
  if (worker_count == 1) {
    for (int64_t head_index = 0; head_index < num_heads; ++head_index) {
      function(head_index);
    }
    return;
  }

  std::atomic<int64_t> next_head{0};
  std::exception_ptr failure;
  std::mutex failure_mutex;

  auto worker = [&]() {
    while (true) {
      const int64_t head_index = next_head.fetch_add(1);
      if (head_index >= num_heads) {
        return;
      }

      try {
        function(head_index);
      } catch (...) {
        {
          std::lock_guard<std::mutex> guard(failure_mutex);
          if (failure == nullptr) {
            failure = std::current_exception();
          }
        }
        next_head.store(num_heads);
        return;
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(worker_count - 1);
  for (int64_t worker_index = 1; worker_index < worker_count; ++worker_index) {
    threads.emplace_back(worker);
  }

  worker();
  for (std::thread& thread : threads) {
    thread.join();
  }

  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
}

void validate_slab_pair(const torch::Tensor& key_slab,
                        const torch::Tensor& value_slab,
                        const torch::Tensor& token_keys, int64_t page_size,
                        int64_t head_size) {
  TORCH_CHECK(key_slab.device().is_cpu(),
              "RetroSpec key slab must reside on CPU");
  TORCH_CHECK(value_slab.device().is_cpu(),
              "RetroSpec value slab must reside on CPU");
  TORCH_CHECK(key_slab.is_contiguous() && value_slab.is_contiguous(),
              "RetroSpec cluster-page slabs must be contiguous");
  TORCH_CHECK(key_slab.sizes() == value_slab.sizes(),
              "RetroSpec key/value slab shapes differ");
  TORCH_CHECK(key_slab.dim() == 3,
              "RetroSpec slabs must have shape [pages, page_size, head_size]");
  TORCH_CHECK(key_slab.size(1) == page_size && key_slab.size(2) == head_size,
              "RetroSpec slab geometry differs from the cluster-page layout");
  TORCH_CHECK(key_slab.scalar_type() == token_keys.scalar_type() &&
                  value_slab.scalar_type() == token_keys.scalar_type(),
              "RetroSpec slab dtype differs from staged token KV");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
retrospec_build_cluster_pages(const std::vector<torch::Tensor>& key_slabs,
                              const std::vector<torch::Tensor>& value_slabs,
                              const torch::Tensor& allocated_page_ids,
                              const torch::Tensor& token_keys,
                              const torch::Tensor& token_values,
                              const torch::Tensor& assignments,
                              const torch::Tensor& cluster_token_counts,
                              const torch::Tensor& token_offsets_in_cluster,
                              int64_t page_size, int64_t num_workers) {
  TORCH_CHECK(page_size > 0, "RetroSpec page size must be positive");
  TORCH_CHECK(num_workers > 0,
              "RetroSpec CPU page-build worker count must be positive");
  TORCH_CHECK(key_slabs.size() == value_slabs.size(),
              "RetroSpec key/value slab counts differ");

  TORCH_CHECK(token_keys.device().is_cpu() && token_values.device().is_cpu(),
              "RetroSpec staged token KV must reside on CPU");
  TORCH_CHECK(token_keys.dim() == 3,
              "RetroSpec token KV must have shape "
              "[num_kv_heads, num_tokens, head_size]");
  TORCH_CHECK(token_keys.sizes() == token_values.sizes(),
              "RetroSpec staged key/value shapes differ");
  TORCH_CHECK(token_keys.scalar_type() == token_values.scalar_type(),
              "RetroSpec staged key/value dtypes differ");
  TORCH_CHECK(token_keys.is_contiguous() && token_values.is_contiguous(),
              "RetroSpec staged token KV must be contiguous");

  validate_integer_tensor(assignments, "assignments", 2);
  validate_integer_tensor(cluster_token_counts, "cluster_token_counts", 2);
  validate_integer_tensor(token_offsets_in_cluster, "token_offsets_in_cluster",
                          2);

  TORCH_CHECK(assignments.size(0) == token_keys.size(0) &&
                  assignments.size(1) == token_keys.size(1),
              "RetroSpec assignments do not match token KV");
  TORCH_CHECK(token_offsets_in_cluster.sizes() == assignments.sizes(),
              "RetroSpec token offsets do not match assignments");
  TORCH_CHECK(cluster_token_counts.size(0) == token_keys.size(0),
              "RetroSpec cluster counts changed KV-head count");

  TORCH_CHECK(allocated_page_ids.device().is_cpu(),
              "RetroSpec allocated page IDs must reside on CPU");
  TORCH_CHECK(allocated_page_ids.scalar_type() == at::kLong,
              "RetroSpec allocated page IDs must use int64");
  TORCH_CHECK(allocated_page_ids.is_contiguous(),
              "RetroSpec allocated page IDs must be contiguous");

  const int64_t num_heads = token_keys.size(0);
  const int64_t num_tokens = token_keys.size(1);
  const int64_t head_size = token_keys.size(2);
  const int64_t num_clusters = cluster_token_counts.size(1);

  for (size_t slab_index = 0; slab_index < key_slabs.size(); ++slab_index) {
    validate_slab_pair(key_slabs[slab_index], value_slabs[slab_index],
                       token_keys, page_size, head_size);
  }

  const torch::Tensor assignments_int64 = to_contiguous_int64(assignments);
  const torch::Tensor counts_int64 = to_contiguous_int64(cluster_token_counts);
  const torch::Tensor offsets_int64 =
      to_contiguous_int64(token_offsets_in_cluster);

  const auto* assignment_data = assignments_int64.data_ptr<int64_t>();
  const auto* count_data = counts_int64.data_ptr<int64_t>();
  const auto* offset_data = offsets_int64.data_ptr<int64_t>();
  const auto* allocated_data = allocated_page_ids.data_ptr<int64_t>();

  const int64_t num_cluster_rows = num_heads * num_clusters;
  std::vector<int64_t> cluster_page_starts(num_cluster_rows);
  std::vector<int64_t> cluster_page_counts(num_cluster_rows);

  int64_t total_pages = 0;
  int64_t max_pages_per_cluster = 0;
  for (int64_t row_index = 0; row_index < num_cluster_rows; ++row_index) {
    const int64_t token_count = count_data[row_index];
    TORCH_CHECK(token_count >= 0,
                "RetroSpec cluster token counts must be non-negative");

    const int64_t page_count = (token_count + page_size - 1) / page_size;
    cluster_page_starts[row_index] = total_pages;
    cluster_page_counts[row_index] = page_count;
    total_pages += page_count;
    max_pages_per_cluster = std::max(max_pages_per_cluster, page_count);
  }

  TORCH_CHECK(allocated_page_ids.numel() == total_pages,
              "RetroSpec allocated page count differs from cluster layout");
  TORCH_CHECK(total_pages == 0 || !key_slabs.empty(),
              "RetroSpec cluster pages require at least one backing slab");

  const auto long_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  const auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

  torch::Tensor page_ids = torch::full(
      {num_heads, num_clusters, max_pages_per_cluster}, -1, long_options);
  torch::Tensor page_token_counts = torch::zeros(
      {num_heads, num_clusters, max_pages_per_cluster}, int_options);
  torch::Tensor head_token_counts = torch::zeros({num_heads}, int_options);

  auto* page_id_output = page_ids.data_ptr<int64_t>();
  auto* page_count_output = page_token_counts.data_ptr<int32_t>();
  auto* head_count_output = head_token_counts.data_ptr<int32_t>();

  const size_t token_bytes =
      static_cast<size_t>(head_size) * token_keys.element_size();
  const size_t page_bytes = static_cast<size_t>(page_size) * token_bytes;
  const auto* key_source = reinterpret_cast<const char*>(token_keys.data_ptr());
  const auto* value_source =
      reinterpret_cast<const char*>(token_values.data_ptr());

  std::vector<std::vector<CompactRange>> ranges_by_head(num_heads);

  parallel_for_heads(num_heads, num_workers, [&](int64_t head_index) {
    const int64_t cluster_row_start = head_index * num_clusters;
    int64_t expected_tokens = 0;
    std::vector<int64_t> cluster_token_starts(num_clusters);

    for (int64_t cluster_index = 0; cluster_index < num_clusters;
         ++cluster_index) {
      cluster_token_starts[cluster_index] = expected_tokens;
      expected_tokens += count_data[cluster_row_start + cluster_index];
    }

    TORCH_CHECK(
        expected_tokens == num_tokens,
        "RetroSpec assignment count does not match cluster_token_counts");
    TORCH_CHECK(
        num_tokens <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "RetroSpec per-head token count exceeds int32");
    head_count_output[head_index] = static_cast<int32_t>(num_tokens);

    std::vector<uint8_t> occupied(num_tokens, uint8_t{0});
    std::vector<CompactRange>& head_ranges = ranges_by_head[head_index];
    int64_t head_token_offset = 0;

    for (int64_t cluster_index = 0; cluster_index < num_clusters;
         ++cluster_index) {
      const int64_t row_index = cluster_row_start + cluster_index;
      const int64_t cluster_tokens = count_data[row_index];
      const int64_t page_count = cluster_page_counts[row_index];
      const int64_t page_start = cluster_page_starts[row_index];

      for (int64_t page_index = 0; page_index < page_count; ++page_index) {
        const int64_t page_id = allocated_data[page_start + page_index];
        TORCH_CHECK(page_id >= 0,
                    "RetroSpec allocated page handle is negative");

        const int64_t slab_id = page_id >> kPageOffsetBits;
        const int64_t page_offset = page_id & kPageOffsetMask;
        TORCH_CHECK(
            slab_id >= 0 && slab_id < static_cast<int64_t>(key_slabs.size()),
            "RetroSpec page handle references an unknown slab");
        TORCH_CHECK(
            page_offset >= 0 && page_offset < key_slabs[slab_id].size(0),
            "RetroSpec page handle exceeds its slab");

        const int64_t metadata_index =
            row_index * max_pages_per_cluster + page_index;
        const int64_t valid_tokens =
            std::min(page_size, cluster_tokens - page_index * page_size);
        TORCH_CHECK(valid_tokens > 0, "RetroSpec valid page has no tokens");

        page_id_output[metadata_index] = page_id;
        page_count_output[metadata_index] = static_cast<int32_t>(valid_tokens);

        char* key_destination =
            reinterpret_cast<char*>(key_slabs[slab_id].data_ptr()) +
            static_cast<size_t>(page_offset) * page_bytes;
        char* value_destination =
            reinterpret_cast<char*>(value_slabs[slab_id].data_ptr()) +
            static_cast<size_t>(page_offset) * page_bytes;

        std::memset(key_destination, 0, page_bytes);
        std::memset(value_destination, 0, page_bytes);

        const int64_t source_token_start = page_offset * page_size;
        if (!head_ranges.empty()) {
          CompactRange& previous = head_ranges.back();
          const int64_t previous_end =
              previous.token_offset + previous.token_count;
          if (previous.slab_id == slab_id &&
              previous_end == source_token_start) {
            previous.token_count += valid_tokens;
            head_token_offset += valid_tokens;
            continue;
          }
        }

        head_ranges.push_back(CompactRange{
            head_index,
            slab_id,
            source_token_start,
            valid_tokens,
            head_token_offset,
        });
        head_token_offset += valid_tokens;
      }
    }

    TORCH_CHECK(head_token_offset == num_tokens,
                "RetroSpec compact descriptor omitted tokens");

    for (int64_t token_index = 0; token_index < num_tokens; ++token_index) {
      const int64_t source_index = head_index * num_tokens + token_index;
      const int64_t cluster_index = assignment_data[source_index];
      TORCH_CHECK(
          cluster_index >= 0 && cluster_index < num_clusters,
          "RetroSpec assignment count does not match cluster_token_counts");

      const int64_t row_index = cluster_row_start + cluster_index;
      const int64_t token_offset = offset_data[source_index];
      const int64_t cluster_tokens = count_data[row_index];
      TORCH_CHECK(token_offset >= 0 && token_offset < cluster_tokens,
                  "RetroSpec cluster token offsets exceed cluster boundaries");

      const int64_t compact_position =
          cluster_token_starts[cluster_index] + token_offset;
      TORCH_CHECK(!occupied[compact_position],
                  "RetroSpec cluster token offsets must be unique");
      occupied[compact_position] = uint8_t{1};

      const int64_t page_linear_index =
          cluster_page_starts[row_index] + token_offset / page_size;
      const int64_t page_id = allocated_data[page_linear_index];
      const int64_t slab_id = page_id >> kPageOffsetBits;
      const int64_t page_offset = page_id & kPageOffsetMask;
      const int64_t token_in_page = token_offset % page_size;

      const size_t destination_token_index =
          static_cast<size_t>(page_offset * page_size + token_in_page);
      const size_t destination_byte_offset =
          destination_token_index * token_bytes;
      const size_t source_byte_offset =
          static_cast<size_t>(source_index) * token_bytes;

      char* key_destination =
          reinterpret_cast<char*>(key_slabs[slab_id].data_ptr()) +
          destination_byte_offset;
      char* value_destination =
          reinterpret_cast<char*>(value_slabs[slab_id].data_ptr()) +
          destination_byte_offset;

      std::memcpy(key_destination, key_source + source_byte_offset,
                  token_bytes);
      std::memcpy(value_destination, value_source + source_byte_offset,
                  token_bytes);
    }
  });

  int64_t num_ranges = 0;
  for (const auto& head_ranges : ranges_by_head) {
    num_ranges += static_cast<int64_t>(head_ranges.size());
  }

  torch::Tensor range_table = torch::empty({num_ranges, 5}, long_options);
  auto* range_output = range_table.data_ptr<int64_t>();

  int64_t range_index = 0;
  for (const auto& head_ranges : ranges_by_head) {
    for (const CompactRange& range : head_ranges) {
      int64_t* row = range_output + range_index * 5;
      row[0] = range.head_index;
      row[1] = range.slab_id;
      row[2] = range.token_offset;
      row[3] = range.token_count;
      row[4] = range.head_token_offset;
      ++range_index;
    }
  }

  return std::make_tuple(std::move(page_ids), std::move(page_token_counts),
                         std::move(range_table), std::move(head_token_counts));
}
