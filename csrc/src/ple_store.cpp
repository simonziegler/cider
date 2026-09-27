// SSD-backed PLE row store and its lazy CPU-stream gather primitive.
// See ple_store.h for the contract.

#include "ple_store.h"

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace cider {

// ── ThreadPool ───────────────────────────────────────────────────

ThreadPool::ThreadPool(int workers) {
  for (int i = 0; i < workers; ++i) {
    workers_.emplace_back([this] { loop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::loop() {
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    wake_.wait(lock, [this] { return stop_ || next_ < tasks_; });
    if (stop_) return;
    int task = next_++;
    auto* fn = fn_;
    lock.unlock();
    (*fn)(task);
    lock.lock();
    if (++finished_ == tasks_) done_.notify_all();
  }
}

void ThreadPool::run(int tasks, const std::function<void(int)>& fn) {
  if (tasks <= 0) return;
  std::lock_guard<std::mutex> serial(run_mutex_);
  std::unique_lock<std::mutex> lock(mutex_);
  fn_ = &fn;
  tasks_ = tasks;
  next_ = 0;
  finished_ = 0;
  wake_.notify_all();
  while (next_ < tasks_) {
    int task = next_++;
    lock.unlock();
    fn(task);
    lock.lock();
    ++finished_;
  }
  done_.wait(lock, [this] { return finished_ == tasks_; });
  tasks_ = 0;
  next_ = 0;
  fn_ = nullptr;
}

// ── PLEStore ─────────────────────────────────────────────────────

PLEStore::PLEStore(const std::vector<std::string>& files,
                   const std::vector<int64_t>& row_starts,
                   const std::vector<PLEShard>& shards, int w_row_bytes,
                   int sb_row_bytes, PLEBackend backend, int threads,
                   int parallel_min_rows)
    : row_starts_(row_starts),
      shards_(shards),
      w_row_bytes_(w_row_bytes),
      sb_row_bytes_(sb_row_bytes),
      backend_(backend),
      parallel_min_rows_(std::max(1, parallel_min_rows)) {
  if (row_starts_.size() != shards_.size() + 1 || shards_.empty()) {
    throw std::invalid_argument("PLEStore: need num_shards + 1 row starts");
  }
  for (const auto& path : files) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      for (int f : fds_) ::close(f);
      throw std::runtime_error("PLEStore: cannot open " + path);
    }
    if (backend_ == PLEBackend::PreadNoCache) {
      ::fcntl(fd, F_NOCACHE, 1);
    }
    fds_.push_back(fd);
    const uint8_t* map = nullptr;
    size_t size = 0;
    if (backend_ == PLEBackend::Mmap) {
      size = static_cast<size_t>(::lseek(fd, 0, SEEK_END));
      void* p = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
      if (p == MAP_FAILED) {
        for (int f : fds_) ::close(f);
        throw std::runtime_error("PLEStore: cannot map " + path);
      }
      ::madvise(p, size, MADV_RANDOM);
      map = static_cast<const uint8_t*>(p);
    }
    maps_.push_back(map);
    map_sizes_.push_back(size);
  }
  for (const auto& shard : shards_) {
    for (int part = 0; part < 3; ++part) {
      if (shard.file[part] < 0 ||
          shard.file[part] >= static_cast<int>(fds_.size())) {
        throw std::invalid_argument("PLEStore: shard names a missing file");
      }
    }
  }
  if (threads > 1) pool_ = std::make_unique<ThreadPool>(threads - 1);
}

PLEStore::~PLEStore() {
  for (size_t i = 0; i < maps_.size(); ++i) {
    if (maps_[i]) ::munmap(const_cast<uint8_t*>(maps_[i]), map_sizes_[i]);
  }
  for (int fd : fds_) ::close(fd);
}

bool PLEStore::read(int file, uint64_t offset, size_t bytes, uint8_t* dst) {
  if (backend_ == PLEBackend::Mmap) {
    if (offset + bytes > map_sizes_[file]) return false;
    std::memcpy(dst, maps_[file] + offset, bytes);
    return true;
  }
  size_t done = 0;
  while (done < bytes) {
    ssize_t got = ::pread(fds_[file], dst + done, bytes - done,
                          static_cast<off_t>(offset + done));
    if (got <= 0) return false;
    done += static_cast<size_t>(got);
  }
  return true;
}

void PLEStore::gather_range(const int64_t* ids, size_t lo, size_t hi,
                            uint8_t* w, uint8_t* s, uint8_t* b) {
  const size_t wb = w_row_bytes_, sb = sb_row_bytes_;
  const int64_t total = row_starts_.back();
  uint64_t errors = 0;
  for (size_t i = lo; i < hi; ++i) {
    const int64_t id = ids[i];
    uint8_t* dw = w + i * wb;
    uint8_t* ds = s + i * sb;
    uint8_t* db = b + i * sb;
    if (id < 0 || id >= total) {
      std::memset(dw, 0, wb);
      std::memset(ds, 0, sb);
      std::memset(db, 0, sb);
      ++errors;
      continue;
    }
    // Last shard whose start is <= id.
    const size_t shard =
        std::upper_bound(row_starts_.begin(), row_starts_.end(), id) -
        row_starts_.begin() - 1;
    const uint64_t local = static_cast<uint64_t>(id - row_starts_[shard]);
    const PLEShard& sh = shards_[shard];
    bool ok = read(sh.file[0], sh.offset[0] + local * wb, wb, dw) &&
              read(sh.file[1], sh.offset[1] + local * sb, sb, ds) &&
              read(sh.file[2], sh.offset[2] + local * sb, sb, db);
    if (!ok) {
      std::memset(dw, 0, wb);
      std::memset(ds, 0, sb);
      std::memset(db, 0, sb);
      ++errors;
    }
  }
  if (errors) errors_ += errors;
}

void PLEStore::gather(const int64_t* ids, size_t n, uint8_t* w, uint8_t* s,
                      uint8_t* b) {
  auto start = std::chrono::steady_clock::now();
  if (pool_ && n >= static_cast<size_t>(parallel_min_rows_)) {
    // Several chunks per thread so a slow (cold) chunk does not stall the rest.
    const int tasks = std::min<size_t>(n, pool_->size() * 4);
    const size_t step = (n + tasks - 1) / tasks;
    pool_->run(tasks, [&](int t) {
      size_t lo = t * step, hi = std::min(n, lo + step);
      if (lo < hi) gather_range(ids, lo, hi, w, s, b);
    });
    ++parallel_calls_;
  } else {
    gather_range(ids, 0, n, w, s, b);
  }
  auto elapsed = std::chrono::steady_clock::now() - start;
  nanos_ += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  rows_ += n;
  ++calls_;
}

std::map<std::string, uint64_t> PLEStore::stats() const {
  return {{"calls", calls_.load()},
          {"rows", rows_.load()},
          {"errors", errors_.load()},
          {"nanos", nanos_.load()},
          {"parallel_calls", parallel_calls_.load()},
          {"bytes", rows_.load() * (w_row_bytes_ + 2 * sb_row_bytes_)}};
}

void PLEStore::reset_stats() {
  calls_ = 0;
  rows_ = 0;
  errors_ = 0;
  nanos_ = 0;
  parallel_calls_ = 0;
}

// ── PLEGather primitive ──────────────────────────────────────────

void PLEGather::eval_cpu(const std::vector<mx::array>& inputs,
                         std::vector<mx::array>& outputs) {
  const auto& ids = inputs[0];
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  const size_t n = ids.size();
  if (n == 0) return;
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(ids);
  for (auto& out : outputs) encoder.set_output_array(out);
  encoder.dispatch([store = store_, n,
                    ids = mx::array::unsafe_weak_copy(ids),
                    w = mx::array::unsafe_weak_copy(outputs[0]),
                    s = mx::array::unsafe_weak_copy(outputs[1]),
                    b = mx::array::unsafe_weak_copy(outputs[2])]() mutable {
    store->gather(ids.data<int64_t>(), n, w.data<uint8_t>(), s.data<uint8_t>(),
                  b.data<uint8_t>());
  });
}

std::vector<mx::array> ple_gather(const std::shared_ptr<PLEStore>& store,
                                  const mx::array& ids) {
  auto cpu = mx::default_stream(mx::Device::cpu);
  auto flat = mx::contiguous(
      mx::astype(mx::flatten(ids, cpu), mx::int64, cpu), false, cpu);
  const int n = flat.shape(0);
  return mx::array::make_arrays(
      {{n, store->w_row_bytes() / 4},
       {n, store->sb_row_bytes() / 2},
       {n, store->sb_row_bytes() / 2}},
      {mx::uint32, mx::bfloat16, mx::bfloat16},
      std::make_shared<PLEGather>(cpu, store), {flat});
}

}  // namespace cider
