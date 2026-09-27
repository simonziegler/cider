#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/ops.h"
#include "pergroup_primitive.h"
#include "ple_store.h"
#include "sdpa_primitive.h"
#include "w4a8_primitive.h"
#include "w8a8_primitive.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

namespace {

// Python-facing owner of a PLEStore. close() only drops this reference:
// gathers already queued hold their own, so the files stay open until the
// last one has run.
struct PLEStoreHandle {
  std::shared_ptr<cider::PLEStore> store;

  cider::PLEStore& get() const {
    if (!store) throw std::runtime_error("PLEStore is closed");
    return *store;
  }
};

cider::PLEBackend parse_backend(const std::string& name) {
  if (name == "mmap") return cider::PLEBackend::Mmap;
  if (name == "pread") return cider::PLEBackend::Pread;
  if (name == "pread_nocache") return cider::PLEBackend::PreadNoCache;
  throw std::invalid_argument("PLEStore backend must be mmap, pread or pread_nocache");
}

}  // namespace

NB_MODULE(_cider_prim, m) {
  m.doc() = "cider: W8A8 + W4A8 INT8 TensorOps + SDPA primitives for Apple M5+";

  m.def("perchannel_linear", &cider::perchannel_linear, "x"_a, "w"_a, "scale_w"_a,
        "bias"_a, "kernel_dir"_a, nb::kw_only(), "stream"_a = nb::none(),
        "W8A8 quantized linear: y = dequant(quant_a(x) @ w_int8) + bias");

  m.def("w4a8_linear", &cider::w4a8_linear, "x"_a, "packed_w"_a, "scale_w"_a,
        "kernel_dir"_a, nb::kw_only(), "stream"_a = nb::none(),
        "W4A8 quantized linear: y = dequant(quant_a(x) @ unpack4(w))");

  m.def("int8_matmul_int32", &cider::int8_matmul_int32, "a"_a, "b"_a,
        "kernel_dir"_a, nb::kw_only(), "stream"_a = nb::none(),
        "Raw INT8xINT8->INT32 matmul (bit-exact, no dequant)");

  m.def("pergroup_linear", &cider::pergroup_linear, "x"_a, "w"_a, "scale_w"_a,
        "bias"_a, "new_bias"_a, "group_size"_a, "kernel_dir"_a, nb::kw_only(),
        "stream"_a = nb::none(),
        "Per-group INT8 linear with bias: prefill GEMM or decode MV with "
        "per-group scales");

  // ── SDPA ──
  m.def("cider_sdpa_1pass", &cider::cider_sdpa_1pass,
        "queries"_a, "keys"_a, "values"_a,
        "gqa_factor"_a, "scale"_a, "kernel_dir"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        "Cider v9 SDPA 1-pass (short sequences)");

  m.def("cider_sdpa_2pass", &cider::cider_sdpa_2pass,
        "queries"_a, "keys"_a, "values"_a,
        "gqa_factor"_a, "blocks"_a, "scale"_a, "kernel_dir"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        "Cider v9 SDPA 2-pass with contiguous chunks + TILE=4 tiling");

  // ── SSD-backed PLE (n-gram embedding) row store ──
  nb::class_<PLEStoreHandle>(m, "PLEStore")
      .def(
          "__init__",
          [](PLEStoreHandle* self, const std::vector<std::string>& files,
             const std::vector<int64_t>& row_starts,
             const std::vector<std::vector<int64_t>>& tensor_file,
             const std::vector<std::vector<int64_t>>& tensor_offset,
             int w_row_bytes, int sb_row_bytes, const std::string& backend,
             int threads, int parallel_min_rows) {
            if (tensor_file.size() != tensor_offset.size())
              throw std::invalid_argument("tensor_file and tensor_offset differ in length");
            std::vector<cider::PLEShard> shards(tensor_file.size());
            for (size_t i = 0; i < shards.size(); ++i) {
              if (tensor_file[i].size() != 3 || tensor_offset[i].size() != 3)
                throw std::invalid_argument("each shard needs weight, scales and biases");
              for (int p = 0; p < 3; ++p) {
                shards[i].file[p] = static_cast<int>(tensor_file[i][p]);
                shards[i].offset[p] = static_cast<uint64_t>(tensor_offset[i][p]);
              }
            }
            new (self) PLEStoreHandle{std::make_shared<cider::PLEStore>(
                files, row_starts, shards, w_row_bytes, sb_row_bytes,
                parse_backend(backend), threads, parallel_min_rows)};
          },
          "files"_a, "row_starts"_a, "tensor_file"_a, "tensor_offset"_a,
          "w_row_bytes"_a, "sb_row_bytes"_a, "backend"_a = "mmap",
          "threads"_a = 16, "parallel_min_rows"_a = 256)
      .def(
          "gather",
          [](const PLEStoreHandle& self, const mx::array& ids) {
            self.get();
            return cider::ple_gather(self.store, ids);
          },
          "ids"_a,
          "Lazy row gather on the CPU stream: returns [codes uint32, scales "
          "bf16, biases bf16], each with one row per id")
      .def("stats", [](const PLEStoreHandle& self) { return self.get().stats(); })
      .def("reset_stats", [](const PLEStoreHandle& self) { self.get().reset_stats(); })
      .def("close", [](PLEStoreHandle& self) { self.store.reset(); })
      .def_prop_ro("closed", [](const PLEStoreHandle& self) { return !self.store; })
      .def_prop_ro("num_rows", [](const PLEStoreHandle& self) { return self.get().num_rows(); });
}
