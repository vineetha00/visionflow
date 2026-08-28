ONNX export: **ok** — torchscript+static-position-ids exporter, opset 17, 22.3s, input [1, 3, 384, 384] → output [1, 729, 1152], max |ONNX − PyTorch| = 5.04e-04

| Execution provider | Status | p50 (ms) | p95 (ms) | mean (ms) |
|---|---|---|---|---|
| TensorrtExecutionProvider | *not measured — TensorrtExecutionProvider not available in this onnxruntime build* | — | — | — |
| CUDAExecutionProvider | *not measured — CUDAExecutionProvider not available in this onnxruntime build* | — | — | — |
| CPUExecutionProvider | ok | 2585.44 | 2675.20 | 2556.77 |

Vision encoder only (see module docstring for why the decoder is excluded).
