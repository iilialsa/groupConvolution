import torch
import math
from torch import nn
import numpy as np
import onnx
from onnxsim import simplify
import tensorrt as trt
from torch._C import dtype
import common
import sys
np.set_printoptions(threshold=sys.maxsize)
TRT_LOGGER = trt.Logger()
class model(nn.Module):
    def __init__(self, in_channels, out_channels, groups, padding_mode='zeros'):
        super(model, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups, padding_mode=padding_mode)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.constant_(m.weight.data, val=0.5)
                print(m.weight.shape)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.1)
        
    def forward(self, x):
        x = self.conv0(x)
        return x


if __name__ == '__main__':
    device = 'cuda:0'

    n1 = 16
    n2 = 16
    channels_in = 4 * n1
    channels_out = 4 * n2
    model = model(channels_in, channels_out, 2).to(device)
    # image = np.random.random([1, 8, 20, 20]).astype(np.float32)
    image = np.ones([1, channels_in, 4, 4]).astype(np.float32)
    input = torch.from_numpy(image).to(device)
    output = model(input)
    print(output, output.shape)

    onnx_file = 'groupconv.onnx'
    torch.onnx.export(model, input, onnx_file, verbose=False, opset_version=12, do_constant_folding=True,
                        input_names=['input'], output_names=['output'],
                        # dynamic_axes={'input' : {0: 'batch', 2: 'height', 3: 'width'}, 
                        #               'output': {0: 'batch', 2: 'y', 3: 'x'}}
    )
    onnx_model = onnx.load(onnx_file)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    model_simp, check = simplify(onnx_model, 
                                    # dynamic_input_shape=True,
                                    # input_shapes={'input':list(img.shape)}
                                )
    onnx.save(model_simp, onnx_file)


    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
        config.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        with open(onnx_file, 'rb') as model:
            parser.parse(model.read())
        # plan = builder.build_serialized_network(network, config)
        engine = builder.build_engine(network, config)
        # engine = runtime.deserialize_cuda_engine(plan)
        trt_outputs = []
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = image
        with engine.create_execution_context() as context:
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            outputs = trt_outputs[0].reshape([channels_out, 4, 4])
            print(outputs)
