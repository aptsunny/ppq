# ------------------------------------------------------------
# PPQ 最佳实践示例工程，在这个工程中，我们将向你展示如何充分调动 PPQ 的各项功能
# ------------------------------------------------------------
import torch
from ppq import *
from ppq.api import *
from mfnr_exp.mfnr_net import DualBranchUnet_v43, DualBranchUnet_v43_addConv2
from ppq.quantization.quantizer import PPL_DSP_MFNR_Quantizer
from mmengine import fileio

BATCHSIZE   = 2
INPUT_SHAPE = [BATCHSIZE, 5, 4, 128, 128]
DEVICE      = 'cpu'
PLATFORM    = TargetPlatform.PPL_DSP_MFNR_INT8
CALIBRATION = [torch.rand(size=INPUT_SHAPE) for _ in range(2)]
QS          = QuantizationSettingFactory.default_setting()

dispatch_nodes = fileio.load('mfnr_exp/modelv2_dispatch.yaml')
if dispatch_nodes:
    for i in list(dispatch_nodes.keys()):
        QS.dispatching_table.append(
            operation=i, platform=TargetPlatform.FP32)

u16_nodes = fileio.load('mfnr_exp/modelv2_u16.yaml')

class DecrementQuantizer(PPL_DSP_MFNR_Quantizer):
    """
    U16 退化
    """
    def init_quantize_config(
        self, operation: Operation) -> OperationQuantizationConfig:
        self.spec = {"bias_bits_i32": []}
        # 通过配置文件给到u16的层
        self.spec['input_bits_u16'] = list(u16_nodes.keys())

        if 'input_bits_u16' in self.spec and operation.name in self.spec['input_bits_u16']:
            self._num_of_bits = 16
            self._quant_min = 0
            self._quant_max = int(pow(2, self._num_of_bits) - 1)
        else:
            self._num_of_bits = 8
            self._quant_min = 0
            self._quant_max = int(pow(2, self._num_of_bits) - 1)

        base_quant_config = self.create_default_quant_config(
            op=operation,
            num_of_bits=self._num_of_bits,
            exponent_bits=0,
            quant_max=self._quant_max,
            quant_min=self._quant_min,
            observer_algorithm='percentile',
            policy=self.quantize_policy,
            rounding=self.rounding_policy,
        )
        if operation.type in {'Conv', 'ConvTranspose'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv', 'ConvTranspose'}:
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                conv_weight_config.num_of_bits = 8
                conv_weight_config.channel_axis = 1
                conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)
                conv_weight_config.observer_algorithm = 'minmax'
                conv_weight_config.quant_max = 127
                conv_weight_config.quant_min = -128

            if operation.num_of_input == 3:
                bias_config = base_quant_config.input_quantization_config[-1]
                if 'bias_bits_i32' in self.spec:
                    bias_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    bias_config.num_of_bits = 30
                    bias_config.quant_max = int(pow(2, 30 - 1))
                    bias_config.quant_min = - int(pow(2, 30 - 1))
                    bias_config.state = QuantizationStates.PASSIVE_INIT
                    bias_config.channel_axis = 0
                    bias_config.observer_algorithm = 'minmax'
                elif 'bias_bits_fp32' in self.spec:
                    bias_config.state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

register_network_quantizer(
    quantizer=DecrementQuantizer,
    platform=TargetPlatform.PPL_DSP_MFNR_INT8)

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

# model = DualBranchUnet_v43()
model = DualBranchUnet_v43_addConv2(deploy1=True, deploy2=True)
model = model.to(DEVICE)

quantized = quantize_torch_model(
    model=model, calib_dataloader=CALIBRATION,
    calib_steps=8, input_shape=INPUT_SHAPE,
    collate_fn=collate_fn, platform=PLATFORM, setting=QS,
    onnx_export_file='onnx.model', device=DEVICE, verbose=0)

# reports = layerwise_error_analyse(
#     graph=quantized, running_device=DEVICE, 
#     collate_fn=collate_fn, dataloader=CALIBRATION)

# remove_activation 也影响导出的QDQ, 模型激活函数显示
export_ppq_graph(
    remove_activation=False,
    graph=quantized, platform=TargetPlatform.ONNXRUNTIME,
    graph_save_to='model.onnx',
    u16_converted=True)

from ppq.parser import NativeExporter
exporter = NativeExporter()
exporter.export(
    file_path='model.native',
    graph=quantized)

load_graph_re = load_native_graph(import_file='model.native')

if dispatch_nodes:
    for i in list(dispatch_nodes.keys()):
        QS.dispatching_table.append(
            operation=i, platform=TargetPlatform.PPL_DSP_MFNR_INT8)

re_quantized = quantize_native_model(
    model=load_graph_re, calib_dataloader=CALIBRATION,
    calib_steps=8, input_shape=INPUT_SHAPE,
    collate_fn=collate_fn, platform=PLATFORM, setting=QS,
    device=DEVICE, verbose=0)

export_ppq_graph(
    remove_activation=False,
    graph=re_quantized, platform=TargetPlatform.ONNXRUNTIME,
    graph_save_to='model_re.onnx',
    u16_converted=True)