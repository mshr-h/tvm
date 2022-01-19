# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable
"""SSD priorbox operators"""
import tvm

from tvm.te import hybrid
from tvm.tir import sqrt

@hybrid.script
def hybrid_priorbox(
    layer,
    input_data, 
    min_size,
    max_size,
    aspect_ratio,
    variance,
    offset,
    flip,
    clip,
):

    """Hybrid routing for priorbox operator.

    Parameters
    ----------
    layer : tvm.te.Tensor or numpy NDArray
        The layer data tensor. 4-D with shape [batch, c_in, h_in, w_in]

    input_data : tvm.te.Tensor or numpy NDArray
        The input data tensor. 4-D with shape [batch, c_in, h_in, w_in]

    min_size : tvm ConsExpr
        Minimum box size in pixels. can be multiple.

    max_size : tvm ConsExpr
        Maximum box size in pixels. can be ignored or same as the number of min_size.

    aspect_ratio : tvm ConsExpr
        Various of aspect ratios. Duplicate ratios will be ignored. 
        If none is provided, we use default ratio 1.0.

    variance : tvm ConsExpr
        Variance for adjusting the prior bboxes.

    offset : tvm ConsExpr
        Offset to the top left corner of each cell.

    flip : tvm ConsExpr
        If true, will flip each aspect ratio.
        For example, if there is aspect ratio "r",
        it will generate aspect ratio "1.0/r" as well.

    clip : tvm ConsExpr
        Clip the prior's coordidate such that it is within [0, 1].

    Returns
    -------
    out : tvm.te.Tensor or numpy NDArray
        prior_num_ = aspect_ratio.size * min_size.size + max_size.size
        3-D tensor with shape [1, 2, layer_height * layer_width * prior_num_ * 4]
    """

    layer_width  = layer.shape[3]
    layer_height = layer.shape[2]
    img_width  = input_data.shape[3]
    img_height = input_data.shape[2]
    step_w = 1.0 * img_width / layer_width
    step_h = 1.0 * img_height / layer_height

    num_min_size = len(min_size)
    num_max_size = len(max_size)
    num_aspect_ratios = len(aspect_ratio)
    num_priors_ = num_aspect_ratios * num_min_size + num_max_size

    dim = layer_height * layer_width * num_priors_ * 4
    output = output_tensor((1, 2, dim), "float32")
    idx = 0
    for h in range(layer_height):
        for w in range(layer_width):
            center_x = (w + offset) * step_w
            center_y = (h + offset) * step_h
            for s in const_range(num_min_size):
                min_size_ = min_size[s]
                # first prior: aspect_ratio = 1, size = min_size
                box_width = min_size_
                box_height = min_size_
 
                # xmin
                output[0, 0, idx] = (center_x - box_width / 2.0) / img_width
                idx+=1
                # ymin
                output[0, 0, idx] = (center_y - box_height / 2.0) / img_height
                idx+=1
                # xmax
                output[0, 0, idx] = (center_x + box_width / 2.0) / img_width
                idx+=1
                # ymax
                output[0, 0, idx] = (center_y + box_height / 2.0) / img_height
                idx+=1

                if num_max_size > 0:
                    max_size_ = max_size[s]
                    # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = sqrt(min_size_ * max_size_)
                    box_height = box_width
                    # xmin
                    output[0, 0, idx] = (center_x - box_width / 2.0) / img_width
                    idx+=1
                    # ymin
                    output[0, 0, idx] = (center_y - box_height / 2.0) / img_height
                    idx+=1
                    # xmax
                    output[0, 0, idx] = (center_x + box_width / 2.0) / img_width
                    idx+=1
                    # ymax
                    output[0, 0, idx] = (center_y + box_height / 2.0) / img_height
                    idx+=1

                # rest of priors (skip first aspect = 1.0 because it's already calculated)
                for r in const_range(1, num_aspect_ratios):
                    box_width = min_size_ * sqrt(aspect_ratio[r])
                    box_height = min_size_ / sqrt(aspect_ratio[r])
                    # xmin
                    output[0, 0, idx] = (center_x - box_width / 2.0) / img_width
                    idx += 1
                    # ymin
                    output[0, 0, idx] = (center_y - box_height / 2.0) / img_height
                    idx += 1
                    # xmax
                    output[0, 0, idx] = (center_x + box_width / 2.0) / img_width
                    idx += 1
                    # ymax
                    output[0, 0, idx] = (center_y + box_height / 2.0) / img_height
                    idx += 1

    # clip the prior's coordidate such that it is within [0, 1]
    if clip:
        for i in const_range(dim):
            if output[0, 0, i] < 0.0:
                output[0, 0, i] = 0.0
            elif output[0, 0, i] > 1.0:
                output[0, 0, i] = 1.0

    # set the variance.
    if len(variance) == 1:
        for d in const_range(dim):
            output[0, 1, d] = variance[0]
    else:
        count = 0
        for h in range(layer_height):
            for w in range(layer_width):
                for i in range(num_priors_):
                    for j in const_range(4):
                        output[0, 1, count] = variance[j]
                        count += 1
    return output

def priorbox(
    layer,
    input_data,
    min_size,
    max_size,
    aspect_ratio=[1.0],
    variance=[1.0],
    offset=0.5,
    flip=True,
    clip=False
):
    """Generate prior(anchor) boxes from layer and input, and parameters.

    Parameters
    ----------
    layer : relay.Expr
        The layer data tensor. 4-D with shape [batch, c_in, h_in, w_in]

    input_data : relay.Expr
        The input data tensor. 4-D with shape [batch, c_in, h_in, w_in]

    min_size : float or tuple of float
        Minimum box size in pixels. can be multiple.

    max_size : float or tuple of float, optional
        Maximum box size in pixels. can be ignored or same as the number of min_size.

    aspect_ratio : float or tuple of float, optional
        Various of aspect ratios. Duplicate ratios will be ignored.
        If none is provided, we use default ratio 1.0.

    variance : float or tuple of float, optional
        Variance for adjusting the prior bboxes.

    offset : float, optional
        Offset to the top left corner of each cell.

    flip : bool, optional
        If true, it will flip each aspect ratio.
        For example, if there is aspect ratio "r",
        it will generate aspect ratio "1.0/r" as well.

    clip : bool, optional
        Clip the prior's coordidate such that it is within [0, 1].

    Returns
    -------
    out : tvm.te.Tensor
        prior_num_ = (aspect_ratio.size + 1) * min_size.size + max_size.size
        3-D tensor with shape [1, 2, layer_height * layer_width * prior_num_ * 4]
    """
    out = hybrid_priorbox(
        layer,
        input_data,
        tvm.runtime.convert(min_size),
        tvm.runtime.convert(max_size),
        tvm.runtime.convert(aspect_ratio),
        tvm.runtime.convert(variance),
        tvm.tir.const(offset, "float32"),
        tvm.tir.const(flip, "bool"),
        tvm.tir.const(clip, "bool"),
    )
    return out
