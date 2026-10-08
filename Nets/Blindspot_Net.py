import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParamConv_reg_variable_dilation(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(ParamConv_reg_variable_dilation, self).__init__()

        self.kernel_size_i = kernel_size[0]
        self.kernel_size_j = kernel_size[1]
        # Create a static edge mask
        self.mask = torch.zeros(3, 3)
        self.mask[0,0]=1
        self.mask[1,0]=1
        self.mask[1,-1]=1
        self.mask[0,1]=1
        self.mask[-1,1]=1
        self.mask[0,-1]=1
        self.mask[-1,0]=1
        self.mask[-1,-1]=1

        # Identify non-zero positions in the mask
        self.register_buffer('indices', torch.nonzero(self.mask).t(), persistent=False)
        # Number of parameters required
        num_params = 8  
        # Initialize the learnable parameters for the non-zero positions
        self.params = nn.Parameter(torch.Tensor(out_channels, in_channels, num_params))
        nn.init.kaiming_uniform_(self.params, a=math.sqrt(5))
        self.stride = stride
        self.padding = padding
        self.dilation = [self.kernel_size_i//2, self.kernel_size_j//2]
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels

    def _zero_kernel(self):
        # built directly on the parameter's device/dtype: no CPU tensor, no host->device copy
        return torch.zeros(self.out_channels, self.in_channels, 3, 3, device=self.params.device, dtype=self.params.dtype)

    def forward(self, x, shuffle_true=1):
        # Use the zeroed kernel and add learnable parameters
        kernel = self._zero_kernel()
        dilation_real = [self.dilation[0] - shuffle_true, self.dilation[1] - shuffle_true]
        kernel[:,:,self.indices[0], self.indices[1]] = self.params
        padding = [self.padding[0] - shuffle_true, self.padding[1] - shuffle_true]
        if dilation_real[0] <= 0 or dilation_real[1] <= 0:
            dilation_real = [1,1]
            kernel = self._zero_kernel()
            kernel[:,:,1,1] = torch.sum(self.params, dim=2)
            padding = [1,1]
        x_padding = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])
        return F.conv2d(x_padding, kernel, bias=None, stride=self.stride, padding=0, dilation=dilation_real, groups=self.groups)

class SHINE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, add_dilation=(0,0), frame_num = 5, filter=64, blocks=14, Bias=False):
        super(SHINE, self).__init__()
        self.n_frames = frame_num
        self.n_filters = filter
        self.n_block = blocks
        self.in_channels = in_channels
        self.add_dilation_i = add_dilation[0]
        self.add_dilation_j = add_dilation[1]
        self.pad_before_downsample = nn.ModuleList()
        self.activation = nn.GELU()
        ratio_1 = 3/4
        if frame_num == 1:
            ratio_1 = 1
        else:
            self.convolution_layer0_a = nn.Sequential(
                *[nn.Sequential(
                    nn.Conv2d(in_channels*(self.n_frames-1), int(self.n_filters*(1-ratio_1)), 3, 1, 1, bias=Bias),
                    self.activation)
                ] + [
                    nn.Sequential(
                        nn.Conv2d(int(self.n_filters*(1-ratio_1)), int(self.n_filters*(1-ratio_1)), 3, 1, 1, bias=Bias),
                        self.activation)
                    for _ in range(np.maximum(self.add_dilation_i,self.add_dilation_j))
                ]
            )
        self.convolution_layer0_b = nn.Conv2d(in_channels, int(self.n_filters*ratio_1), 1, 1, 0, bias=Bias)
        self.first_output = ParamConv_reg_variable_dilation('donut', self.n_filters, self.n_filters,((self.add_dilation_i+1)*2 + 1,(self.add_dilation_j+1)*2 + 1), 1,
                                                    (1 + self.add_dilation_i, 1 + self.add_dilation_j), dilation=1, bias=Bias)
        self.convolution_layer10 = torch.nn.ModuleList()
        self.convolution_layer2 = torch.nn.ModuleList()

        for i in range(4):
            if i > 0:
                scaler = 1
            elif i > -1:
                scaler = 1
            else:
                scaler = 1
            self.convolution_layer10.append(
                nn.Sequential(
                    nn.Conv2d(self.n_filters*scaler, self.n_filters*scaler, 3, 1, 1,bias=Bias),
                    self.activation,
                    nn.Conv2d(self.n_filters*scaler, self.n_filters*scaler, 3, 1, 1,bias=Bias),
                ))

            self.convolution_layer2.append(
                nn.Sequential(
                    nn.Conv2d(self.n_filters*scaler, self.n_filters*scaler, 3, 1, 1,bias=Bias),
                    self.activation,
                    nn.Conv2d(self.n_filters*scaler, self.n_filters*scaler, 3, 1, 1,bias=Bias),
                ))
        
        self.dilated_convs = nn.ModuleList()
        self.dilated_convs_res = nn.ModuleList()
        kernel_size_i = (self.add_dilation_i+1+4)*2 + 1 
        padding_i = kernel_size_i//2
        kernel_size_j = (self.add_dilation_j+1+4)*2 + 1
        padding_j = kernel_size_j//2
        for i in range(4):  # Create 5 dilated conv layers
            if i > 0:
                scaler = 1
            elif i > -1:
                scaler = 1
            else:
                scaler = 1
            layer = ParamConv_reg_variable_dilation('donut', self.n_filters*scaler,  self.n_filters*scaler, (kernel_size_i,kernel_size_j), 1, (padding_i,padding_j), dilation=1, bias=Bias)
            self.dilated_convs.append(layer)
            layer = ParamConv_reg_variable_dilation('donut',  self.n_filters*scaler,  self.n_filters*scaler, (kernel_size_i-4,kernel_size_j-4), 1, (padding_i-2,padding_j-2), dilation=1, bias=Bias)
            self.dilated_convs_res.append(layer)
            # Update kernel_size and padding for the next iteration
            kernel_size_i = (kernel_size_i //4 + 2 + 4) * 2 + 1 
            padding_i = kernel_size_i // 2
            kernel_size_j = (kernel_size_j //4 + 2 + 4) * 2 + 1 
            padding_j = kernel_size_j // 2

        feature_size = int(self.n_filters*(576/64))

        self.outconvs = nn.Sequential(
                                    nn.Conv2d(feature_size,feature_size//2,1,1,0,bias=Bias),
                                    self.activation,
                                    nn.Conv2d(feature_size//2,self.n_filters,1,1,0,bias=Bias),
                                    self.activation)                                   
        self.last_out = nn.Conv2d(self.n_filters,out_channels,1,1,0,bias=Bias)

        self.pool2d = nn.AvgPool2d(2)
        self.upsample_list = nn.ModuleList()
        for i in range(4):
            self.upsample_list.append(nn.Sequential(
                nn.Upsample(scale_factor=2**(i), mode='bilinear'),
                nn.Conv2d(self.n_filters, self.n_filters,1, stride=1, padding=0, bias=Bias),
                self.activation))

    def forward(self, input_image, shuffle=0):
        N, C, nH, nW = input_image.shape
        if nH % 64 != 0 or nW % 64 != 0:
            input_image = F.pad(input_image, [0, (-nW) % 64, 0, (-nH) % 64], mode = 'constant')
        if shuffle !=0:
            target = input_image[:,self.n_frames//2,:,:].unsqueeze(1).clone()
        else:
            target = input_image[:,self.n_frames//2,:,:].unsqueeze(1)
        if self.n_frames > 1:
            features = torch.cat((input_image[:,:self.n_frames//2,:,:], input_image[:,self.n_frames//2+1:,:,:]), dim=1)
            target = self.activation(self.convolution_layer0_b(target))
            features = self.convolution_layer0_a(features)
            base = torch.cat((target, features), dim=1)
        else:
            base = self.activation(self.convolution_layer0_b(input_image))

        initial_base = base
        output = self.activation(self.first_output(base, 0))
        base = self.activation(self.convolution_layer10[0](base) + base)
        merged = self.activation(self.dilated_convs_res[0](base, 0))
        output = torch.cat((output,merged),dim=1)
        base = self.activation(self.convolution_layer2[0](base) + base)
        for i in range(4):
            merged = self.activation(self.dilated_convs[i](base, 0))
            merged = self.upsample_list[i](merged)
            output = torch.cat((output,merged), dim=1)

            if i < 3:
                if i == -1:
                    base = self.pool2d(torch.cat((base,initial_base),dim=1))
                else:
                    base = self.pool2d(base)
                initial_base = base
                base = self.activation(self.convolution_layer10[i+1](base) + base)
                merged = self.activation(self.dilated_convs_res[i+1](base, 0))
                merged = self.upsample_list[i+1](merged)
                output = torch.cat((output, merged), dim=1)
                base = self.activation(self.convolution_layer2[i+1](base) + base)
        output = self.outconvs(output)
        output = self.last_out(output)
      
        if nH % 64 != 0 or nW % 64 != 0:
            output = output[:, :, 0:nH, 0:nW]

        return output
    

class ParamEdgeConv_top_open(nn.Module):
    def __init__(self, mask_type, initial_dilation, in_channels, out_channels, kernel_size,
                 stride, padding, dilation=1, groups=1, bias=False):
        super(ParamEdgeConv_top_open, self).__init__()

        self.kernel_size_i = kernel_size[0]
        self.kernel_size_j = kernel_size[1]
        initial_dilation_i = int(initial_dilation[0])
        initial_dilation_j = int(initial_dilation[1])

        mask = torch.zeros(self.kernel_size_i, self.kernel_size_j)
        mask[0:initial_dilation_i+1, 0] = 1
        mask[0:initial_dilation_i+1, -1] = 1
        mask[-initial_dilation_i-1:, 0] = 1
        mask[-initial_dilation_i-1:, -1] = 1
        mask[0, 0:initial_dilation_j+1] = 1
        mask[-1, 0:initial_dilation_j+1] = 1
        mask[0, -initial_dilation_j-1:] = 1
        mask[-1, -initial_dilation_j-1:] = 1

        # Register indices as buffer so they live on GPU with the model
        self.register_buffer('indices', torch.nonzero(mask).t())
        num_params = int(mask.sum())

        self.params = nn.Parameter(torch.Tensor(out_channels, in_channels, num_params))
        nn.init.kaiming_uniform_(self.params, a=math.sqrt(5))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # Store shapes needed for kernel construction
        self.out_channels = out_channels
        self.in_channels_per_group = in_channels // groups

    def forward(self, x):
        kernel = torch.zeros(
            self.out_channels, self.in_channels_per_group,
            self.kernel_size_i, self.kernel_size_j,
            device=self.params.device, dtype=self.params.dtype
        )
        kernel[:, :, self.indices[0], self.indices[1]] = self.params
        return F.conv2d(x, kernel, bias=None, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)
 
class SHINE_defect(torch.nn.Module):
    """SHINE variant redesigned for inputs with structural defects.
 
    Key differences from SHINE:
      - Activation: ReLU instead of GELU (faster, more stable on defect-sparse data)
      - Neighbor frame embedding: 1x1 conv (suppresses spatial mixing of
        potentially defect-contaminated neighbor pixels at the input stage)
      - Center frame embedding: Conv + ReLU (activation fused into layer)
      - Base concat order: [feat_left | target | feat_right] (temporally symmetric)
      - Sparse conv kernel: ParamEdgeConv_top_open instead of ParamConv_reg_variable_dilation
        (asymmetric corner-biased mask, avoids center pixel contamination from defects)
      - 3 stages instead of 4 (with channel doubling after stage 0: n_filters -> 2*n_filters)
      - Upsampling: PixelShuffle instead of bilinear Upsample + 1x1 conv
      - Padding schedule: dilation-anchored rather than kernel-derived
      - Skip concat at first downsample: pool2d(cat(base, initial_base)) at i=0 only
    """
    def __init__(self, in_channels, out_channels, add_dilation=(0, 0),
                 frame_num=5, filter=64, blocks=14, Bias=False):
        super(SHINE_defect, self).__init__()
        self.n_frames = frame_num
        self.n_filters = filter
        self.n_block = blocks
        self.in_channels = in_channels
        self.add_dilation_i = add_dilation[0]
        self.add_dilation_j = add_dilation[1]
 
        self.activation = nn.ReLU(inplace=True)
        ratio_1 = 3 / 4
 
        # ------------
        # Input embedding
        # ------------
        if frame_num == 1:
            ratio_1 = 1
        else:
            # 1x1 conv for neighbor frames: intentionally avoids spatial mixing
            # of potentially defect-contaminated neighbor pixels
            self.convolution_layer0_a = nn.Sequential(
                *[nn.Sequential(
                    nn.Conv2d(in_channels * (self.n_frames - 1),
                              int(self.n_filters * (1 - ratio_1)), 1, 1, 0, bias=Bias),
                    self.activation)
                ] + [
                    nn.Sequential(
                        nn.Conv2d(int(self.n_filters * (1 - ratio_1)),
                                  int(self.n_filters * (1 - ratio_1)), 3, 1, 1, bias=Bias),
                        self.activation)
                    for _ in range(np.maximum(self.add_dilation_i, self.add_dilation_j))
                ]
            )
 
        # Center frame branch: fused activation unlike SHINE's bare Conv2d
        self.convolution_layer0_b = nn.Sequential(
            nn.Conv2d(in_channels, int(self.n_filters * ratio_1), 1, 1, 0, bias=Bias),
            self.activation
        )
 
        # ------------
        # First blindspot output: ParamEdgeConv_top_open (asymmetric corner mask)
        # ------------
        self.first_output = nn.Sequential(
            ParamEdgeConv_top_open(
                'donut', (self.add_dilation_i, self.add_dilation_j),
                self.n_filters, self.n_filters,
                ((self.add_dilation_i + 1) * 2 + 1, (self.add_dilation_j + 1) * 2 + 1),
                1, (1 + self.add_dilation_i, 1 + self.add_dilation_j),
                dilation=1, bias=Bias
            ),
            self.activation
        )
 
        # ------------
        # Residual conv blocks (3 stages; channels double after stage 0)
        # ------------
        self.convolution_layer10 = torch.nn.ModuleList()
        self.convolution_layer2 = torch.nn.ModuleList()
 
        for i in range(3):
            scaler = 1 if i == 0 else 2
            self.convolution_layer10.append(
                nn.Sequential(
                    nn.Conv2d(self.n_filters * scaler, self.n_filters * scaler, 3, 1, 1, bias=Bias),
                    self.activation,
                    nn.Conv2d(self.n_filters * scaler, self.n_filters * scaler, 3, 1, 1, bias=Bias),
                ))
            self.convolution_layer2.append(
                nn.Sequential(
                    nn.Conv2d(self.n_filters * scaler, self.n_filters * scaler, 3, 1, 1, bias=Bias),
                    self.activation,
                    nn.Conv2d(self.n_filters * scaler, self.n_filters * scaler, 3, 1, 1, bias=Bias),
                ))
 
        # ------------
        # Dilated blindspot convs: ParamEdgeConv_top_open at each stage
        # Padding is dilation-anchored (not kernel-derived) to keep
        # receptive field locked to the defect scale as kernel grows.
        # ------------
        self.dilated_convs = nn.ModuleList()
        self.dilated_convs_res = nn.ModuleList()
        kernel_size_i = (self.add_dilation_i + 1 + 4) * 2 + 1
        padding_i = self.add_dilation_i + 1 + 4       # dilation-anchored
        kernel_size_j = (self.add_dilation_j + 1 + 4) * 2 + 1
        padding_j = self.add_dilation_j + 1 + 4
 
        for i in range(3):
            scaler = 1 if i == 0 else 2
            self.dilated_convs.append(nn.Sequential(
                ParamEdgeConv_top_open(
                    'donut', (self.add_dilation_i, self.add_dilation_j),
                    self.n_filters * scaler, self.n_filters * scaler,
                    (kernel_size_i, kernel_size_j), 1, (padding_i, padding_j),
                    dilation=1, bias=Bias
                ),
                self.activation
            ))
            self.dilated_convs_res.append(nn.Sequential(
                ParamEdgeConv_top_open(
                    'donut', (self.add_dilation_i, self.add_dilation_j),
                    self.n_filters * scaler, self.n_filters * scaler,
                    (kernel_size_i - 4, kernel_size_j - 4), 1,
                    (padding_i - 2, padding_j - 2),
                    dilation=1, bias=Bias
                ),
                self.activation
            ))
            kernel_size_i = (kernel_size_i // 4 + 4 + 1) * 2 + 1
            padding_i = kernel_size_i // 2
            kernel_size_j = (kernel_size_j // 4 + 4 + 1) * 2 + 1
            padding_j = kernel_size_j // 2
 
        # ------------
        # Output aggregation
        # feature_size=272 verified by tracing channel accumulation:
        #   first_output(64) + res[0](64) + dc[0](64) + res[1](32) + dc[1](32) + res[2](8) + dc[2](8) = 272
        # ------------
        feature_size = int(self.n_filters * (272 / 64))
        self.outconvs = nn.Sequential(
            nn.Conv2d(feature_size, feature_size // 2, 1, 1, 0, bias=Bias),
            self.activation,
            nn.Conv2d(feature_size // 2, self.n_filters, 1, 1, 0, bias=Bias),
            self.activation
        )
        self.last_out = nn.Conv2d(self.n_filters, out_channels, 1, 1, 0, bias=Bias)
 
        self.pool2d = nn.AvgPool2d(2)
 
        # PixelShuffle upsampling: artifact-free, enabled by channel doubling at stage 1+
        self.upsample_list = nn.ModuleList()
        for i in range(3):
            self.upsample_list.append(nn.PixelShuffle(2 ** i))
 
    def forward(self, input_image, shuffle=False):
        N, C, nH, nW = input_image.shape
        if nH % 64 != 0 or nW % 64 != 0:
            input_image = F.pad(input_image, [0, (-nW) % 64, 0, (-nH) % 64], mode='constant')
 
        # ------------
        # Input embedding
        # ------------
        if self.n_frames > 1:
            target = input_image[:, self.n_frames // 2, :, :].unsqueeze(1)
            features = torch.cat((input_image[:, :self.n_frames // 2, :, :],
                                   input_image[:, self.n_frames // 2 + 1:, :, :]), dim=1)
            target = self.convolution_layer0_b(target)
            features = self.convolution_layer0_a(features)
            # Temporally symmetric concat: [feat_left | target | feat_right]
            base = torch.cat((features[:, :self.n_frames // 2, :, :],
                               target,
                               features[:, self.n_frames // 2:, :, :]), dim=1)
        else:
            base = self.convolution_layer0_b(input_image)
 
        # ------------
        # Stage 0 (n_filters channels)
        # ------------
        initial_base = base
        outputs = [self.first_output(base)]
        base = self.activation(self.convolution_layer10[0](base) + base)
        merged = self.dilated_convs_res[0](base)
        outputs.append(merged)
        base = self.activation(self.convolution_layer2[0](base) + base)
 
        # ------------
        # Stages 1-3: downsample -> wider channels -> PixelShuffle upsample
        # ------------
        for i in range(3):
            merged = self.dilated_convs[i](base)
            merged = self.upsample_list[i](merged)
            outputs.append(merged)
 
            if i < 2:
                # Skip concat only at first downsample (i=0); plain pool thereafter
                if i == 0:
                    base = self.pool2d(torch.cat((base, initial_base), dim=1))
                else:
                    base = self.pool2d(base)
                initial_base = base
                base = self.activation(self.convolution_layer10[i + 1](base) + base)
                merged = self.dilated_convs_res[i + 1](base)
                merged = self.upsample_list[i + 1](merged)
                outputs.append(merged)
                base = self.activation(self.convolution_layer2[i + 1](base) + base)
 
        # ------------
        # Aggregation and output
        # ------------
        output = torch.cat(outputs, dim=1)
        output = self.outconvs(output)
        output = self.last_out(output)
 
        if nH % 64 != 0 or nW % 64 != 0:
            output = output[:, :, 0:nH, 0:nW]
 
        return output
 