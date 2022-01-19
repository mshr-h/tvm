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
"""PriorBox operations."""
from . import _make


def priorbox(
    layer,
    input_data,
    min_size,
    max_size,
    aspect_ratio,
    variance,
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
        If true, will flip each aspect ratio.
        For example, if there is aspect ratio "r",
        it will generate aspect ratio "1.0/r" as well.
        
    clip : bool, optional
        Clip the prior's coordidate such that it is within [0, 1].

    Returns
    -------
    out : relay.Expr
        prior_num_ = (aspect_ratio.size + 1) * min_size.size + max_size.size
        3-D tensor with shape [1, 2, layer_height * layer_width * prior_num_ * 4]
    """
    return _make.priorbox(layer, input_data, min_size, max_size, aspect_ratio, variance, offset, flip, clip)
