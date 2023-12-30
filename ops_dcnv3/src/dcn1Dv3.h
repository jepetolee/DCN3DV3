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

#include "cpu/dcn1Dv3_cpu.h"

#ifdef WITH_CUDA
#include "cuda/dcn2Dv3_cuda.h"
#endif

at::Tensor dcn1Dv3_forward(const at::Tensor &input, const at::Tensor &offset,
                           const at::Tensor &mask, const int kernel_l,
                           const int stride_l, const int pad_l,
                           const int dilation_l, const int group, const int group_channels,
                           const float offset_scale, const int im2col_step) {
    if (input.type().is_cuda()) {
#ifdef WITH_CUDA
        return dcn2Dv3_cuda_forward(input, offset, mask, kernel_h, kernel_w,
                                  stride_h, stride_w, pad_h, pad_w, dilation_h,
                                  dilation_w, group, group_channels,
                                  offset_scale, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
dcn1Dv3_backward(const at::Tensor &input, const at::Tensor &offset,
                 const at::Tensor &mask, const int kernel_l,
                 const int stride_l, const int pad_l,
                 const int dilation_l, const int group, const int group_channels,
                 const float offset_scale, const at::Tensor &grad_output,
                 const int im2col_step) {
    if (input.type().is_cuda()) {
#ifdef WITH_CUDA
        return dcn2Dv3_cuda_backward(input, offset, mask, kernel_h, kernel_w,
                                   stride_h, stride_w, pad_h, pad_w, dilation_h,
                                   dilation_w, group, group_channels,
                                   offset_scale, grad_output, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}
