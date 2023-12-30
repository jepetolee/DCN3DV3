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

#include "dcnv3.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dcn1Dv3_forward", &dcn1Dv3_forward, "dcn1Dv3_forward");
    m.def("dcn1Dv3_backward", &dcn1Dv3_backward, "dcn1Dv3_backward");
    m.def("dcn2Dv3_forward", &dcn2Dv3_forward, "dcn2Dv3_forward");
    m.def("dcn2Dv3_backward", &dcn2Dv3_backward, "dcn2Dv3_backward");
    m.def("dcn3Dv3_forward", &dcn3Dv3_forward, "dcn3Dv3_forward");
    m.def("dcn3Dv3_backward", &dcn3Dv3_backward, "dcn3Dv3_backward");
}
