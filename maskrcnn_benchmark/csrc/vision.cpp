// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "SigmoidFocalLoss.h"
#include "deform_conv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
  m.def("deform_conv_forward_cuda", &deform_conv_forward_cuda,
        "deform forward (CUDA)");
  m.def("deform_conv_backward_input_cuda", &deform_conv_backward_input_cuda,
        "deform_conv_backward_input (CUDA)");
  m.def("deform_conv_backward_parameters_cuda",
        &deform_conv_backward_parameters_cuda,
        "deform_conv_backward_parameters (CUDA)");
  m.def("modulated_deform_conv_cuda_forward",
        &modulated_deform_conv_cuda_forward,
        "modulated deform conv forward (CUDA)");
  m.def("modulated_deform_conv_cuda_backward",
        &modulated_deform_conv_cuda_backward,
        "modulated deform conv backward (CUDA)");
}
