// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct CopySpan {
  const char* key_source;
  const char* value_source;
  char* key_destination;
  char* value_destination;
  size_t num_bytes;
};

void copy_span(const CopySpan& span) {
  std::memcpy(span.key_destination, span.key_source, span.num_bytes);
  std::memcpy(span.value_destination, span.value_source, span.num_bytes);
}

void parallel_copy_spans(const std::vector<CopySpan>& spans,
                         int64_t num_workers) {
  TORCH_CHECK(num_workers > 0, "Compact gather worker count must be positive");
  if (spans.empty()) {
    return;
  }

  const int64_t worker_count =
      std::min<int64_t>(num_workers, static_cast<int64_t>(spans.size()));
  if (worker_count == 1) {
    for (const CopySpan& span : spans) {
      copy_span(span);
    }
    return;
  }

  std::atomic<int64_t> next_span{0};
  auto worker = [&]() {
    while (true) {
      const int64_t span_index =
          next_span.fetch_add(1, std::memory_order_relaxed);
      if (span_index >= static_cast<int64_t>(spans.size())) {
        return;
      }
      copy_span(spans[span_index]);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(worker_count - 1);
  try {
    for (int64_t worker_index = 1; worker_index < worker_count;
         ++worker_index) {
      threads.emplace_back(worker);
    }
  } catch (...) {
    for (std::thread& thread : threads) {
      thread.join();
    }
    throw;
  }

  worker();
  for (std::thread& thread : threads) {
    thread.join();
  }
}

void validate_slab_pair(const torch::Tensor& key_slab,
                        const torch::Tensor& value_slab,
                        const torch::Tensor& key_output) {
  TORCH_CHECK(key_slab.device().is_cpu(), "Key slab must reside on CPU");
  TORCH_CHECK(value_slab.device().is_cpu(), "Value slab must reside on CPU");
  TORCH_CHECK(key_slab.is_contiguous(), "Key slab must be contiguous");
  TORCH_CHECK(value_slab.is_contiguous(), "Value slab must be contiguous");
  TORCH_CHECK(key_slab.sizes() == value_slab.sizes(),
              "Key/value slab shapes differ");
  TORCH_CHECK(key_slab.scalar_type() == key_output.scalar_type(),
              "Key slab dtype differs from output dtype");
  TORCH_CHECK(value_slab.scalar_type() == key_output.scalar_type(),
              "Value slab dtype differs from output dtype");
}

}  // namespace

void retrospec_gather_compact_kv(const std::vector<torch::Tensor>& key_slabs,
                                 const std::vector<torch::Tensor>& value_slabs,
                                 const std::vector<torch::Tensor>& range_tables,
                                 const torch::Tensor& token_offsets,
                                 int64_t destination_token_start,
                                 torch::Tensor& key_output,
                                 torch::Tensor& value_output,
                                 int64_t num_workers) {
  TORCH_CHECK(key_slabs.size() == value_slabs.size(),
              "Key/value slab counts differ");
  TORCH_CHECK(range_tables.size() == static_cast<size_t>(token_offsets.size(0)),
              "Range-table request count differs from token offsets");
  TORCH_CHECK(!key_slabs.empty(), "Compact gather requires source slabs");
  TORCH_CHECK(destination_token_start >= 0,
              "Destination token start must be non-negative");
  TORCH_CHECK(num_workers > 0, "Compact gather worker count must be positive");

  TORCH_CHECK(token_offsets.device().is_cpu(),
              "Token offsets must reside on CPU");
  TORCH_CHECK(token_offsets.scalar_type() == at::kLong,
              "Token offsets must use int64");
  TORCH_CHECK(token_offsets.dim() == 2,
              "Token offsets must have shape [requests, heads]");
  TORCH_CHECK(token_offsets.is_contiguous(),
              "Token offsets must be contiguous");

  TORCH_CHECK(key_output.device().is_cpu(), "Key output must reside on CPU");
  TORCH_CHECK(value_output.device().is_cpu(),
              "Value output must reside on CPU");
  TORCH_CHECK(key_output.dim() == 2 && value_output.dim() == 2,
              "Compact outputs must have shape [tokens, head_size]");
  TORCH_CHECK(key_output.sizes() == value_output.sizes(),
              "Key/value output shapes differ");
  TORCH_CHECK(key_output.scalar_type() == value_output.scalar_type(),
              "Key/value output dtypes differ");
  TORCH_CHECK(key_output.is_contiguous() && value_output.is_contiguous(),
              "Compact outputs must be contiguous");

  const int64_t num_requests = token_offsets.size(0);
  const int64_t num_heads = token_offsets.size(1);
  const int64_t head_size = key_output.size(1);
  const int64_t destination_token_end =
      destination_token_start + key_output.size(0);
  const size_t token_bytes =
      static_cast<size_t>(head_size) * key_output.element_size();
  const auto* token_offsets_data = token_offsets.data_ptr<int64_t>();

  for (size_t slab_index = 0; slab_index < key_slabs.size(); ++slab_index) {
    validate_slab_pair(key_slabs[slab_index], value_slabs[slab_index],
                       key_output);
    TORCH_CHECK(key_slabs[slab_index].size(-1) == head_size,
                "Source slab head size differs from output head size");
  }

  size_t num_ranges = 0;
  for (const torch::Tensor& range_table : range_tables) {
    TORCH_CHECK(range_table.device().is_cpu(),
                "Compact range table must reside on CPU");
    TORCH_CHECK(range_table.scalar_type() == at::kLong,
                "Compact range table must use int64");
    TORCH_CHECK(range_table.dim() == 2 && range_table.size(1) == 5,
                "Compact range table must have shape [ranges, 5]");
    TORCH_CHECK(range_table.is_contiguous(),
                "Compact range table must be contiguous");
    num_ranges += range_table.size(0);
  }

  std::vector<CopySpan> spans;
  spans.reserve(num_ranges);
  char* key_destination = reinterpret_cast<char*>(key_output.data_ptr());
  char* value_destination = reinterpret_cast<char*>(value_output.data_ptr());

  for (int64_t request_index = 0; request_index < num_requests;
       ++request_index) {
    const torch::Tensor& range_table = range_tables[request_index];
    const auto* ranges = range_table.data_ptr<int64_t>();

    for (int64_t range_index = 0; range_index < range_table.size(0);
         ++range_index) {
      const int64_t* row = ranges + range_index * 5;
      const int64_t head_index = row[0];
      const int64_t slab_id = row[1];
      const int64_t source_token_start = row[2];
      const int64_t token_count = row[3];
      const int64_t head_token_offset = row[4];

      TORCH_CHECK(head_index >= 0 && head_index < num_heads,
                  "Compact range contains an invalid KV-head index");
      TORCH_CHECK(
          slab_id >= 0 && slab_id < static_cast<int64_t>(key_slabs.size()),
          "Compact range references an unknown slab");
      TORCH_CHECK(source_token_start >= 0 && token_count > 0,
                  "Compact source range is invalid");
      TORCH_CHECK(head_token_offset >= 0,
                  "Compact destination offset is invalid");

      const int64_t head_destination_start =
          token_offsets_data[request_index * num_heads + head_index];
      const int64_t range_destination_start =
          head_destination_start + head_token_offset;
      const int64_t range_destination_end =
          range_destination_start + token_count;
      const int64_t copy_start =
          std::max(range_destination_start, destination_token_start);
      const int64_t copy_end =
          std::min(range_destination_end, destination_token_end);
      if (copy_start >= copy_end) {
        continue;
      }

      const torch::Tensor& key_slab = key_slabs[slab_id];
      const torch::Tensor& value_slab = value_slabs[slab_id];
      const int64_t source_capacity = key_slab.numel() / head_size;
      const int64_t source_copy_start =
          source_token_start + copy_start - range_destination_start;
      const int64_t copy_tokens = copy_end - copy_start;
      TORCH_CHECK(source_copy_start + copy_tokens <= source_capacity,
                  "Compact range exceeds its source slab");

      const size_t source_byte_offset =
          static_cast<size_t>(source_copy_start) * token_bytes;
      const size_t destination_byte_offset =
          static_cast<size_t>(copy_start - destination_token_start) *
          token_bytes;
      const size_t copy_bytes = static_cast<size_t>(copy_tokens) * token_bytes;

      spans.push_back(CopySpan{
          reinterpret_cast<const char*>(key_slab.data_ptr()) +
              source_byte_offset,
          reinterpret_cast<const char*>(value_slab.data_ptr()) +
              source_byte_offset,
          key_destination + destination_byte_offset,
          value_destination + destination_byte_offset,
          copy_bytes,
      });
    }
  }

  parallel_copy_spans(spans, num_workers);
}

void retrospec_gather_cluster_pages(
    const std::vector<torch::Tensor>& key_slabs,
    const std::vector<torch::Tensor>& value_slabs,
    const torch::Tensor& page_ids, int64_t page_size, torch::Tensor& key_output,
    torch::Tensor& value_output, int64_t num_workers) {
  TORCH_CHECK(key_slabs.size() == value_slabs.size(),
              "Key/value slab counts differ");
  TORCH_CHECK(page_ids.device().is_cpu(),
              "Cluster page IDs must reside on CPU");
  TORCH_CHECK(page_ids.scalar_type() == at::kLong,
              "Cluster page IDs must use int64");
  TORCH_CHECK(page_ids.dim() == 1 && page_ids.is_contiguous(),
              "Cluster page IDs must be contiguous and one-dimensional");
  TORCH_CHECK(key_output.device().is_cpu() && value_output.device().is_cpu(),
              "Cluster page outputs must reside on CPU");
  TORCH_CHECK(key_output.sizes() == value_output.sizes(),
              "Cluster page output shapes differ");
  TORCH_CHECK(key_output.dim() == 3,
              "Cluster page outputs must have shape [pages, page_size, head]");
  TORCH_CHECK(key_output.size(0) == page_ids.numel(),
              "Cluster page output count differs from page IDs");
  TORCH_CHECK(key_output.size(1) == page_size,
              "Cluster page output page size differs");
  TORCH_CHECK(key_output.scalar_type() == value_output.scalar_type(),
              "Cluster page output dtypes differ");
  TORCH_CHECK(key_output.is_contiguous() && value_output.is_contiguous(),
              "Cluster page outputs must be contiguous");
  TORCH_CHECK(num_workers > 0,
              "Cluster page gather worker count must be positive");

  constexpr uint64_t kPageOffsetMask = (uint64_t{1} << 32) - 1;
  const int64_t head_size = key_output.size(2);
  const size_t page_bytes =
      static_cast<size_t>(page_size) * head_size * key_output.element_size();

  for (size_t slab_index = 0; slab_index < key_slabs.size(); ++slab_index) {
    validate_slab_pair(key_slabs[slab_index], value_slabs[slab_index],
                       key_output);
    TORCH_CHECK(key_slabs[slab_index].dim() == 3,
                "Cluster page slab must be three-dimensional");
    TORCH_CHECK(key_slabs[slab_index].size(1) == page_size &&
                    key_slabs[slab_index].size(2) == head_size,
                "Cluster page slab layout differs from output");
  }

  const auto* page_id_data = page_ids.data_ptr<int64_t>();
  char* key_destination = reinterpret_cast<char*>(key_output.data_ptr());
  char* value_destination = reinterpret_cast<char*>(value_output.data_ptr());
  std::vector<CopySpan> spans;
  spans.reserve(page_ids.numel());

  for (int64_t destination_page = 0; destination_page < page_ids.numel();
       ++destination_page) {
    const int64_t encoded_page_id = page_id_data[destination_page];
    TORCH_CHECK(encoded_page_id >= 0,
                "Cluster page gather received an invalid page ID");
    const uint64_t page_id = static_cast<uint64_t>(encoded_page_id);
    const int64_t slab_id = page_id >> 32;
    const int64_t page_offset = page_id & kPageOffsetMask;
    TORCH_CHECK(
        slab_id >= 0 && slab_id < static_cast<int64_t>(key_slabs.size()),
        "Cluster page references an unknown slab");

    const torch::Tensor& key_slab = key_slabs[slab_id];
    const torch::Tensor& value_slab = value_slabs[slab_id];
    TORCH_CHECK(page_offset < key_slab.size(0),
                "Cluster page offset exceeds slab capacity");
    const size_t source_byte_offset =
        static_cast<size_t>(page_offset) * page_bytes;
    const size_t destination_byte_offset =
        static_cast<size_t>(destination_page) * page_bytes;
    CopySpan next{
        reinterpret_cast<const char*>(key_slab.data_ptr()) + source_byte_offset,
        reinterpret_cast<const char*>(value_slab.data_ptr()) +
            source_byte_offset,
        key_destination + destination_byte_offset,
        value_destination + destination_byte_offset,
        page_bytes,
    };

    if (!spans.empty()) {
      CopySpan& previous = spans.back();
      const bool contiguous =
          previous.key_source + previous.num_bytes == next.key_source &&
          previous.value_source + previous.num_bytes == next.value_source &&
          previous.key_destination + previous.num_bytes ==
              next.key_destination &&
          previous.value_destination + previous.num_bytes ==
              next.value_destination;
      if (contiguous) {
        previous.num_bytes += next.num_bytes;
        continue;
      }
    }
    spans.push_back(next);
  }

  parallel_copy_spans(spans, num_workers);
}
