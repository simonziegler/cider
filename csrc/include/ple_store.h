// SSD-backed row store for an affine-quantized, sharded embedding table
// (Qwen3.8-Flash-Next's 32 GB n-gram "PLE" table), and the MLX primitive that
// gathers rows from it lazily on the CPU stream.
//
// Each row is three byte ranges in the checkpoint's safetensors files: packed
// codes (uint32), bf16 scales and bf16 biases. The store copies those bytes,
// unchanged, into unified-memory MLX arrays; dequantization stays a normal
// mx.dequantize on the GPU, so the result is bit-identical to reading the
// table any other way.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cider {

namespace mx = mlx::core;

// Blocking parallel-for over a fixed set of workers; the caller participates.
// Tasks are claimed under the mutex, so a late-waking worker can never run a
// task of a finished call.
class ThreadPool {
 public:
  explicit ThreadPool(int workers);
  ~ThreadPool();
  void run(int tasks, const std::function<void(int)>& fn);
  int size() const { return static_cast<int>(workers_.size()) + 1; }

 private:
  void loop();
  std::vector<std::thread> workers_;
  std::mutex mutex_, run_mutex_;
  std::condition_variable wake_, done_;
  const std::function<void(int)>* fn_ = nullptr;
  int tasks_ = 0, next_ = 0, finished_ = 0;
  bool stop_ = false;
};

enum class PLEBackend {
  Mmap,         // memcpy from a read-only mapping (page faults; page cache)
  Pread,        // pread per byte range (page cache)
  PreadNoCache  // pread with F_NOCACHE: bypasses the page cache entirely
};

struct PLEShard {
  int file[3];         // weight, scales, biases
  uint64_t offset[3];  // absolute byte offset of each tensor's data
};

class PLEStore {
 public:
  PLEStore(
      const std::vector<std::string>& files,
      const std::vector<int64_t>& row_starts,
      const std::vector<PLEShard>& shards,
      int w_row_bytes,
      int sb_row_bytes,
      PLEBackend backend,
      int threads,
      int parallel_min_rows);
  ~PLEStore();

  // Copies rows ids[0..n) into w (n x w_row_bytes), s and b (n x sb_row_bytes).
  // Out-of-range ids and failed reads zero their row and count as errors;
  // it never throws, because it runs inside an MLX CPU-stream task.
  void gather(const int64_t* ids, size_t n, uint8_t* w, uint8_t* s, uint8_t* b);

  std::map<std::string, uint64_t> stats() const;
  void reset_stats();

  int w_row_bytes() const { return w_row_bytes_; }
  int sb_row_bytes() const { return sb_row_bytes_; }
  int64_t num_rows() const { return row_starts_.back(); }

 private:
  void gather_range(const int64_t* ids, size_t lo, size_t hi, uint8_t* w,
                    uint8_t* s, uint8_t* b);
  bool read(int file, uint64_t offset, size_t bytes, uint8_t* dst);

  std::vector<int> fds_;
  std::vector<const uint8_t*> maps_;
  std::vector<size_t> map_sizes_;
  std::vector<int64_t> row_starts_;
  std::vector<PLEShard> shards_;
  int w_row_bytes_, sb_row_bytes_;
  PLEBackend backend_;
  int parallel_min_rows_;
  std::unique_ptr<ThreadPool> pool_;

  std::atomic<uint64_t> calls_{0}, rows_{0}, errors_{0}, nanos_{0},
      parallel_calls_{0};
};

// ── Custom Primitive ─────────────────────────────────────────────
// Inputs:  [ids (N,) int64, row-contiguous]
// Outputs: [w (N, w_row_bytes/4) uint32, s (N, sb_row_bytes/2) bfloat16,
//           b (N, sb_row_bytes/2) bfloat16]
// CPU stream only. Holds the store alive until every queued gather has run.
class PLEGather : public mx::Primitive {
 public:
  PLEGather(mx::Stream stream, std::shared_ptr<PLEStore> store)
      : mx::Primitive(stream), store_(std::move(store)) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    throw std::runtime_error("PLEGather: runs on the CPU stream only");
  }

  const char* name() const override { return "PLEGather"; }

  bool is_equivalent(const mx::Primitive& other) const override {
    return store_ == static_cast<const PLEGather&>(other).store_;
  }

 private:
  std::shared_ptr<PLEStore> store_;
};

// Lazy gather of the rows named by ids (any shape, any integer dtype).
std::vector<mx::array> ple_gather(const std::shared_ptr<PLEStore>& store,
                                  const mx::array& ids);

}  // namespace cider
