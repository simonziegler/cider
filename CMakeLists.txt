cmake_minimum_required(VERSION 3.27)

project(cider LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ── Python + nanobind + MLX ──────────────────────────────────────
# 3.11 is the interpreter oMLX bundles. nanobind applies STABLE_ABI on 3.12+
# only and ignores it below, so one CMakeLists serves both.
find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)

# nanobind_ROOT and MLX_ROOT may be passed in (-D...) to build against another
# interpreter's mlx, e.g. oMLX's bundle, whose Python has no nanobind or
# headers. The nanobind version must match the one mlx was built with (its ABI
# tag), or every op fails with "incompatible function arguments".
if(NOT nanobind_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)

if(NOT MLX_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT)
endif()
find_package(MLX CONFIG REQUIRED)

# ── Shared library: all primitives ───────────────────────────────
add_library(cider_prim_lib)

target_sources(cider_prim_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/csrc/src/w8a8_primitive.mm
  ${CMAKE_CURRENT_LIST_DIR}/csrc/src/w4a8_primitive.mm
  ${CMAKE_CURRENT_LIST_DIR}/csrc/src/pergroup_primitive.mm
  ${CMAKE_CURRENT_LIST_DIR}/csrc/src/sdpa_primitive.mm
)

target_include_directories(cider_prim_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/csrc/include
)

target_link_libraries(cider_prim_lib PUBLIC
  mlx
  "-framework Metal"
  "-framework Foundation"
)

# ── nanobind Python module ───────────────────────────────────────
nanobind_add_module(
  _cider_prim
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/csrc/src/prim_bindings.cpp
)
target_link_libraries(_cider_prim PRIVATE cider_prim_lib)

if(BUILD_SHARED_LIBS)
  target_link_options(_cider_prim PRIVATE -Wl,-rpath,@loader_path)
endif()

# ── Install ──────────────────────────────────────────────────────
install(TARGETS cider_prim_lib LIBRARY DESTINATION cider/lib)
install(TARGETS _cider_prim LIBRARY DESTINATION cider/lib)
