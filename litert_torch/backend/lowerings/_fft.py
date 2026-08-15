# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides lowering for coreaten to stablehlo for FFT operations."""

from litert_torch.backend.lowerings import registry
from litert_converter.mlir import ir
from litert_converter.mlir.dialects import stablehlo
import torch

lower = registry.lower


@lower(torch.ops.aten._fft_r2c.default)
def _aten_fft_r2c(
    lctx,
    input: ir.Value,
    dim: list[int],
    normalization: int,
    onesided: bool,
):
  if normalization != 0:
    raise NotImplementedError("FFT normalization != 0 is not yet implemented.")

  # A two-sided real-to-complex FFT returns the full spectrum, while
  # stablehlo.fft's RFFT only produces the non-redundant half. Rejecting it
  # here avoids emitting an op whose result type disagrees with the graph.
  if not onesided:
    raise NotImplementedError(
        "Two-sided real-to-complex FFT (onesided=False) is not yet implemented."
    )

  input_type: ir.RankedTensorType = input.type
  rank = len(input_type.shape)
  pos_dim = [d if d >= 0 else rank + d for d in dim]

  # TFLite's RFFT2D transforms the trailing dimensions and halves the last one,
  # so the transformed dims must be the trailing dims in ascending order.
  if pos_dim != list(range(rank - len(pos_dim), rank)):
    raise NotImplementedError(
        "Only FFT over the trailing dimensions is currently supported, got"
        f" dim={dim} for a rank-{rank} input."
    )

  if len(pos_dim) > 2:
    raise NotImplementedError(
        f"Only 1D and 2D FFT are currently supported, got a {len(pos_dim)}D"
        " FFT."
    )

  # TFLite's RFFT2D kernel only accepts power-of-two FFT lengths. 
  # Fail during conversion rather than when the interpreter
  # prepares the RFFT2D node.
  for d in pos_dim:
    length = input_type.shape[d]
    if length <= 0 or (length & (length - 1)) != 0:
      raise NotImplementedError(
          "TFLite's RFFT2D kernel requires power-of-two FFT lengths, but"
          f" dimension {d} of the input has size {length}."
      )

  out_aval = lctx.node.meta.get("tensor_meta") or lctx.node.meta.get("val")
  complex_elem_type = ir.ComplexType.get(input_type.element_type)

  fft_input_shape = list(input_type.shape)
  fft_output_shape = list(out_aval.shape)

  # TFLite only supports 2D FFT (RFFT2D). We implement a 1D FFT of length L by
  # reshaping the input to [..., 1, L], performing a 2D FFT with length [1, L],
  # and reshaping the output back to [..., L // 2 + 1]. A 2D FFT maps onto
  # RFFT2D directly and needs no reshaping.
  is_1d = len(pos_dim) == 1
  fft_input = input
  if is_1d:
    fft_input_shape.insert(-1, 1)
    fft_output_shape.insert(-1, 1)
    fft_input = stablehlo.reshape(
        ir.RankedTensorType.get(fft_input_shape, input_type.element_type),
        input,
    )

  fft_res = stablehlo.fft(
      results=[ir.RankedTensorType.get(fft_output_shape, complex_elem_type)],
      operand=fft_input,
      fft_type=stablehlo.FftTypeAttr.get("RFFT"),
      fft_length=ir.DenseI64ArrayAttr.get(fft_input_shape[-2:]),
  )

  # 4. Reshape output back [..., 1, L // 2 + 1] -> [..., L // 2 + 1]
  if is_1d:
    res_type = ir.RankedTensorType.get(out_aval.shape, complex_elem_type)
    return stablehlo.reshape(res_type, fft_res)

  return fft_res

