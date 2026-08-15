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
import litert_torch
from litert_torch import backend
import numpy as np
import torch
from torch import fft

from absl.testing import absltest as googletest
from absl.testing import parameterized


class RfftModule(torch.nn.Module):

  def forward(self, x):
    return torch.fft.rfft(x)


class Rfft2Module(torch.nn.Module):

  def forward(self, x):
    return torch.fft.rfft2(x)


class Rfft2AbsModule(torch.nn.Module):
  def __init__(self, frame_size, hop, frame_count, out_features=4):
    super().__init__()
    self.frame_size = frame_size
    self.hop = hop
    self.fc = torch.nn.Linear(frame_count * (frame_size // 2 + 1), out_features)

  def forward(self, x):
    frames = x.unfold(-1, self.frame_size, self.hop)
    spec = fft.rfft2(frames).abs()
    return self.fc(spec.flatten(start_dim=1))


class TestFft(parameterized.TestCase):

  def test_rfft(self):
    x = torch.randn(1, 2, 2048)
    ep = torch.export.export(RfftModule(), (x,))
    edge_model = litert_torch.convert(ep.module(), (x,), {})
    expected = torch.fft.rfft(x).numpy()
    actual = edge_model(x.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("rank2", (256, 512)),
      ("rank3", (1, 256, 512)),
      ("rank4", (2, 3, 64, 128)),
  )
  def test_rfft2(self, shape):
    x = torch.randn(*shape)
    ep = torch.export.export(Rfft2Module(), (x,))
    edge_model = litert_torch.convert(ep.module(), (x,), {})
    expected = torch.fft.rfft2(x).numpy()
    actual = edge_model(x.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  def test_rfft2_non_power_of_two_raises(self):
    # TFLite's RFFT2D kernel rejects non-power-of-two FFT lengths, so this must
    # fail at conversion time rather than when the interpreter runs the model.
    x = torch.randn(1, 374, 512)
    ep = torch.export.export(Rfft2Module(), (x,))
    with self.assertRaisesRegex(Exception, "power-of-two"):
      litert_torch.convert(ep.module(), (x,), {})

  def test_rfft2_unfold_abs(self):
    frame_size, hop, frame_count = 512, 384, 256
    length = frame_size + (frame_count - 1) * hop
    model = Rfft2AbsModule(frame_size, hop, frame_count).eval()
    x = torch.randn(1, length)
    ep = torch.export.export(model, (x,))
    edge_model = litert_torch.convert(ep.module(), (x,), {})
    with torch.no_grad():
      expected = model(x).numpy()
    actual = edge_model(x.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  googletest.main()
