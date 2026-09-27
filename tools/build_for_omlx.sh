#!/usr/bin/env bash
# Build cider's extension against oMLX's bundled Python and mlx, and stage an
# importable copy of the package for that interpreter.
#
# oMLX ships CPython 3.11 without headers or nanobind, so the build borrows:
#   - a matching CPython 3.11 for headers (PY311, e.g. uv's 3.11.x),
#   - nanobind's CMake dir from a Python that has nanobind == the version mlx
#     was built with (NANOBIND_PY, default: this repo's .venv),
#   - MLX's CMake package from the oMLX bundle itself.
#
# Usage: tools/build_for_omlx.sh <out_dir> [<oMLX.app>]
# Then:  PYTHONPATH=<out_dir>:<oMLX Resources>:<oMLX site-packages> python3.11 ...
set -euo pipefail

OUT=${1:?usage: build_for_omlx.sh <out_dir> [oMLX.app]}
APP=${2:-/Applications/oMLX.app}
RES="$APP/Contents/Resources"
SITE="$RES/Python/framework-mlx-base/lib/python3.11/site-packages"
OMLX_PY="$RES/Python/cpython-3.11/bin/python3.11"
SRC=$(cd "$(dirname "$0")/.." && pwd)
PY311=${PY311:-$(ls -d "$HOME"/.local/share/uv/python/cpython-3.11.*-macos-aarch64-none/bin/python3.11 2>/dev/null | tail -1)}
NANOBIND_PY=${NANOBIND_PY:-"$SRC/.venv/bin/python"}

[[ -x "$OMLX_PY" ]] || { echo "no oMLX interpreter at $OMLX_PY" >&2; exit 1; }
[[ -x "$PY311" ]] || { echo "set PY311 to a CPython 3.11 with headers" >&2; exit 1; }

MLX_ROOT=$(PYTHONPATH="$SITE" "$OMLX_PY" -m mlx --cmake-dir)
NB_ROOT=$("$NANOBIND_PY" -m nanobind --cmake_dir)
NB_VER=$("$NANOBIND_PY" -c "import nanobind; print(nanobind.__version__)")
MLX_VER=$(PYTHONPATH="$SITE" "$OMLX_PY" -c "import mlx.core as mx; print(mx.__version__)")
echo "oMLX mlx $MLX_VER, nanobind $NB_VER, headers from $PY311"

BUILD="$OUT/build"
mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY311" -Dnanobind_ROOT="$NB_ROOT" -DMLX_ROOT="$MLX_ROOT"
cmake --build "$BUILD" --config Release -j

rm -rf "$OUT/cider"
rsync -a --exclude __pycache__ --exclude 'lib/*.so' --exclude 'lib/*.dylib' "$SRC/cider/" "$OUT/cider/"
mkdir -p "$OUT/cider/lib"
cp "$BUILD"/_cider_prim*.so "$BUILD"/libcider_prim_lib.dylib "$OUT/cider/lib/"
echo "staged $OUT/cider"
