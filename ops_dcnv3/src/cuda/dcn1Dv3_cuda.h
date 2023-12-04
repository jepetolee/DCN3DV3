/*!
**************************************************************************************************
* InternImage
* Copyright (c) 2022 OpenGVLab
* Licensed under The MIT License [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

at::Tensor dcn1Dv3_cuda_forward(const at::Tensor &input, const at::Tensor &offset,
                              const at::Tensor &mask, const int kernel_l,
                              const int stride_l, const int pad_l,
                              const int dilation_l, const int group,
                              const int group_channels,
                              const float offset_scale, const int im2col_step);

std::vector<at::Tensor>
dcn1Dv3_cuda_backward(const at::Tensor &input, const at::Tensor &offset,
                    const at::Tensor &mask, const int kernel_l,
                    const int stride_l, const int pad_l, const int dilation_l,
                    const int group, const int group_channels, const float offset_scale,
                    const at::Tensor &grad_output, const int im2col_step);
