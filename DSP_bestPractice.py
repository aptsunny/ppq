# ------------------------------------------------------------
# PPQ 最佳实践示例工程，在这个工程中，我们将向你展示如何充分调动 PPQ 的各项功能
# ------------------------------------------------------------
import torch
from ppq import *
from ppq.api import *
from mfnr_net import DualBranchUnet_v43

BATCHSIZE   = 2
INPUT_SHAPE = [BATCHSIZE, 5, 4, 128, 128]
DEVICE      = 'cpu'
PLATFORM    = TargetPlatform.PPL_DSP_MFNR_INT8
CALIBRATION = [torch.rand(size=INPUT_SHAPE) for _ in range(2)]
QS          = QuantizationSettingFactory.default_setting()

QS.dispatching_table.append(
    operation='/tail_conv/Conv', platform=TargetPlatform.FP32)
QS.dispatching_table.append(
    operation='/up_rgb/up_rgb.1/Conv', platform=TargetPlatform.FP32)
def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

model = DualBranchUnet_v43()
model = model.to(DEVICE)

quantized = quantize_torch_model(
    model=model, calib_dataloader=CALIBRATION,
    calib_steps=8, input_shape=INPUT_SHAPE,
    collate_fn=collate_fn, platform=PLATFORM, setting=QS,
    onnx_export_file='onnx.model', device=DEVICE, verbose=0)

reports = layerwise_error_analyse(
    graph=quantized, running_device=DEVICE, 
    collate_fn=collate_fn, dataloader=CALIBRATION)

export_ppq_graph(
    graph=quantized, platform=TargetPlatform.ONNXRUNTIME,
    graph_save_to='model.onnx')
