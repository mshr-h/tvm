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
# pylint: disable=import-error, invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable, too-many-nested-blocks, too-many-branches, too-many-statements, too-many-function-args
"""Detection Output operator"""
import tvm

from tvm.te import hybrid


@hybrid.script
def hybrid_get_loc_predictions(
    loc,
    num,
    num_preds_per_class,
    num_loc_classes,
    share_location
):
    if share_location:
        all_loc_preds = output_tensor((  1, num_loc_classes, num_preds_per_class, 4), loc.dtype)
    else:
        all_loc_preds = output_tensor((num, num_loc_classes, num_preds_per_class, 4), loc.dtype)

    for i in parallel(num):
        for p in range(num_preds_per_class):
            for c in range(num_loc_classes):
                t = i * (num_preds_per_class * num_loc_classes * 4) + p * num_loc_classes * 4 + c * 4
                all_loc_preds[i, c, p, 0] = loc[0, t + 0]
                all_loc_preds[i, c, p, 1] = loc[0, t + 1]
                all_loc_preds[i, c, p, 2] = loc[0, t + 2]
                all_loc_preds[i, c, p, 3] = loc[0, t + 3]

    return all_loc_preds


@hybrid.script
def hybrid_get_confidence_scores(
    conf,
    num,
    num_preds_per_class,
    num_classes
):
    all_conf_scores = output_tensor((num, num_classes, num_preds_per_class), conf.dtype)

    for i in parallel(num):
        for p in range(num_preds_per_class):
            for c in range(num_classes):
                all_conf_scores[i, c, p] = conf[0, i * (num_preds_per_class * num_classes) + p * num_classes + c]

    return all_conf_scores

@hybrid.script
def hybrid_get_priorbboxes(
    priorbox,
    num_priors
):
    prior_bboxes = output_tensor((num_priors, 4), priorbox.dtype)

    for i in parallel(num_priors):
        for j in range(4):
            prior_bboxes[i, j] = priorbox[0, 0, i * 4 + j]

    return prior_bboxes

@hybrid.script
def hybrid_get_prior_variances(
    priorbox,
    num_priors
):
    prior_variances = output_tensor((num_priors, 4), priorbox.dtype)

    for i in parallel(num_priors):
        for j in range(4):
            prior_variances[i, j] = priorbox[0, 1, i * 4 + j]

    return prior_variances


@hybrid.script
def hybrid_decode_bbox(
    prior_bboxes,
    prior_variances,
    code_type,
    variance_encoded_in_target,
    clip_bbox,
    bboxes,
    bbes_label,
    bb_label
):
    i = bbes_label[0]
    j = bbes_label[1]
    k = bb_label[0]
    decode_bbox = output_tensor((4,), "float32")
    if code_type == 1: # PriorBoxParameter_CodeType_CORNER
        if variance_encoded_in_target:
            # variance is encoded in target, we simply need to add the offset
            # predictions.
            decode_bbox[0] = prior_bboxes[k, 0] + bboxes[i, j, k, 0]
            decode_bbox[1] = prior_bboxes[k, 0] + bboxes[i, j, k, 1]
            decode_bbox[2] = prior_bboxes[k, 0] + bboxes[i, j, k, 2]
            decode_bbox[3] = prior_bboxes[k, 0] + bboxes[i, j, k, 3]
        else:
            # variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox[0] = prior_bboxes[k, 0] + prior_variances[k, 0] * bboxes[i, j, k, 0]
            decode_bbox[1] = prior_bboxes[k, 1] + prior_variances[k, 1] * bboxes[i, j, k, 1]
            decode_bbox[2] = prior_bboxes[k, 2] + prior_variances[k, 2] * bboxes[i, j, k, 2]
            decode_bbox[3] = prior_bboxes[k, 3] + prior_variances[k, 3] * bboxes[i, j, k, 3]
    elif code_type == 2: # PriorBoxParameter_CodeType_CENTER_SIZE
        prior_w = prior_bboxes[k, 2] - prior_bboxes[k, 0]
        prior_h = prior_bboxes[k, 3] - prior_bboxes[k, 1]
        prior_center_x = (prior_bboxes[k, 0] + prior_bboxes[k, 2]) / 2.0
        prior_center_y = (prior_bboxes[k, 1] + prior_bboxes[k, 3]) / 2.0
        decode_bbox_center_x = 0.0
        decode_bbox_center_y = 0.0
        decode_bbox_w = 0.0
        decode_bbox_h = 0.0
        if variance_encoded_in_target:
            decode_bbox_center_x = bboxes[i, j, k, 0] * prior_w + prior_center_x
            decode_bbox_center_y = bboxes[i, j, k, 1] * prior_h + prior_center_y
            decode_bbox_w = exp(bboxes[i, j, k, 2]) * prior_w 
            decode_bbox_h = exp(bboxes[i, j, k, 3]) * prior_h 
        else:
            decode_bbox_center_x = prior_variances[k, 0] * bboxes[i, j, k, 0] * prior_w + prior_center_x
            decode_bbox_center_y = prior_variances[k, 1] * bboxes[i, j, k, 1] * prior_h + prior_center_y
            decode_bbox_w = exp(prior_variances[k, 2] * bboxes[i, j, k, 2]) * prior_w
            decode_bbox_h = exp(prior_variances[k, 3] * bboxes[i, j, k, 3]) * prior_h
        decode_bbox[0] = decode_bbox_center_x - decode_bbox_w / 2.0
        decode_bbox[1] = decode_bbox_center_y - decode_bbox_h / 2.0
        decode_bbox[2] = decode_bbox_center_x + decode_bbox_w / 2.0
        decode_bbox[3] = decode_bbox_center_y + decode_bbox_h / 2.0
    elif code_type == 3: # PriorBoxParameter_CodeType_CORNER_SIZE
        decode_bbox[0] = 0.0
        decode_bbox[1] = 0.0
        decode_bbox[2] = 1.0
        decode_bbox[3] = 1.0
    else:
        decode_bbox[0] = 0.0
        decode_bbox[1] = 0.0
        decode_bbox[2] = 1.0
        decode_bbox[3] = 1.0

    if clip_bbox:
        for i in range(4):
            if decode_bbox[0, i] > 1.0:
                decode_bbox[0, i] = 1.0
            if decode_bbox[0, i] < 0.0:
                decode_bbox[0, i] = 0.0

    return decode_bbox


@hybrid.script
def hybrid_decode_bboxes(
    prior_bboxes,
    prior_variances,
    code_type,
    variance_encoded_in_target,
    clip_bbox,
    bboxes,
    bbes_label,
):
    assert (prior_bboxes.shape[0] == prior_variances.shape[0]), "size invalid between prior_bboxes.shape[0] and prior_variances.shape[0]"
    assert (prior_bboxes.shape[0] == bboxes.shape[2]), "size invalid between prior_bboxes.shape[0] and bboxes.shape[2]"

    num_bboxes = prior_bboxes.shape[0]
    decode_bboxes = output_tensor((num_bboxes, 4), "float32")
    bb_label = allocate((1,), "int32")

    for i in range(num_bboxes):
        bb_label[0] = i
        decode_bbox = hybrid_decode_bbox(
            prior_bboxes,
            prior_variances,
            code_type,
            variance_encoded_in_target,
            clip_bbox,
            bboxes,
            bbes_label,
            bb_label
        )
        for j in range(4):
            decode_bboxes[i, j] = decode_bbox[j]

    return decode_bboxes


@hybrid.script
def hybrid_decode_bboxes_all(
    loc,
    all_loc_preds,
    prior_bboxes,
    prior_variances,
    share_location,
    num_loc_classes,
    bg_label_id,
    code_type,
    variance_encoded_in_target,
    clip_bbox
):
    num = loc.shape[0]
    num_bboxes = prior_bboxes.shape[0]
    bbes_label = allocate((2,), "int32")

    if share_location:
        all_decode_bboxes = output_tensor((num,               1, num_bboxes, 4), "float32")
    else:
        all_decode_bboxes = output_tensor((num, num_loc_classes, num_bboxes, 4), "float32")

    for i in range(num):
        bbes_label[0] = i
        if share_location:
            bbes_label[1] = 0
            decode_bboxes = hybrid_decode_bboxes(
                prior_bboxes,
                prior_variances,
                code_type,
                variance_encoded_in_target,
                clip_bbox,
                all_loc_preds,
                bbes_label,
            )
            for j in range(num_bboxes):
                for k in range(4):
                    all_decode_bboxes[i, 0, j, k] = decode_bboxes[j, k]
        else:
            for c in range(num_loc_classes):
                if int32(c) != bg_label_id:
                    bbes_label[1] = c
                    decode_bboxes = hybrid_decode_bboxes(
                        prior_bboxes,
                        prior_variances,
                        code_type,
                        variance_encoded_in_target,
                        clip_bbox,
                        all_loc_preds,
                        bbes_label,
                    )
                    for j in range(num_bboxes):
                        for k in range(4):
                            all_decode_bboxes[i, c, j, k] = decode_bboxes[j, k]

    return all_decode_bboxes


@hybrid.script
def hybrid_prepare_detection_output(
    loc,
    conf,
    priorbox,
    bg_label_id,
    code_type,
    conf_th,
    keep_top_k,
    nms_th,
    nms_top_k,
    num_classes,
    share_location,
    num_loc_classes,
    variance_encoded_in_target,
    eta,
    num_priors,
):
    """Hybrid routing for detection output (non-maximum suppression).

    Parameters
    ----------

    Returns
    -------
    output : tvm.te.Tensor or tuple of tvm.te.Tensor
        4-D tensor with shape [1, 1, keep_top_k, 7]
        7 means [image_id, label, confidence, xmin, ymin, xmax, ymax]

    """
    assert num_priors * num_loc_classes * 4 == loc.shape[1], "loc.shape invalid"
    assert num_priors * num_classes == conf.shape[1], "conf.shape invalid"

    all_loc_preds = hybrid_get_loc_predictions(
        loc,
        loc.shape[0],
        num_priors,
        num_loc_classes,
        share_location
    )

    all_conf_scores = hybrid_get_confidence_scores(
        conf,
        loc.shape[0],
        num_priors,
        num_classes
    )

    prior_bboxes = hybrid_get_priorbboxes(
        priorbox,
        num_priors
    )

    prior_variances = hybrid_get_prior_variances(
        priorbox,
        num_priors
    )

    return all_loc_preds, all_conf_scores, prior_bboxes, prior_variances


@hybrid.script
def hybrid_get_max_score_index(
    all_conf_scores,
    score_th,
    nms_top_k,
    nms_label
):
    score_index_vec = output_tensor((2, nms_top_k), "float32")
    i = nms_label[0]
    c_idx = nms_label[1]
    num_score = all_conf_scores.shape[2]
    cnt = 0
    score = 0.0
    index = 0

    for k in range(nms_top_k):
        score_index_vec[0, k] = 0.0
        score_index_vec[1, k] = -1.0

    for area_k in range(num_score):
        score = all_conf_scores[i, c_idx, area_k]
        if score > score_th:
            index = cnt
            if cnt > nms_top_k:
                min_score = score
                for l in range(nms_top_k):
                    if score_index_vec[0, l] < min_score:
                        min_score = score_index_vec[0, l]
                        index = l
            if index < nms_top_k:
                score_index_vec[0, index] = score
                score_index_vec[1, index] = float32(area_k)
            cnt = cnt + 1

    top_k = cnt if cnt < nms_top_k else nms_top_k
    # 降順に並べる(バブルソートで対応、数が増えるならば別のアルゴリズムで対応)
    tmp_score = 0.0
    tmp_index = 0.0
    for k in range(top_k):
        for l in range(top_k - k - 1):
            if score_index_vec[0, l] < score_index_vec[0, l + 1]:
                tmp_score = score_index_vec[0, l]
                tmp_index = score_index_vec[1, l]
                score_index_vec[0, l] = score_index_vec[0, l + 1] 
                score_index_vec[1, l] = score_index_vec[1, l + 1] 
                score_index_vec[0, l + 1] = tmp_score
                score_index_vec[1, l + 1] = tmp_index

    return score_index_vec

@hybrid.script
def hybrid_jaccard_overlap(
    bboxes,
    bbox_index
):
    overlap = output_tensor((1,), "float32")
    b1_idx = bbox_index[0]
    b2_idx = bbox_index[1]
    b1_xmin = bboxes[0, 0, b1_idx, 0]
    b1_ymin = bboxes[0, 0, b1_idx, 1]
    b1_xmax = bboxes[0, 0, b1_idx, 2]
    b1_ymax = bboxes[0, 0, b1_idx, 3]
    b2_xmin = bboxes[0, 0, b2_idx, 0]
    b2_ymin = bboxes[0, 0, b2_idx, 1]
    b2_xmax = bboxes[0, 0, b2_idx, 2]
    b2_ymax = bboxes[0, 0, b2_idx, 3]
    if (b2_xmin > b1_xmax) or (b2_xmax < b1_xmin) or (b2_ymin > b1_ymax) or (b2_ymax < b1_ymin):
        overlap[0] = float32(0.0)
    else:
        inter_xmin = b1_xmin if b1_xmin > b2_xmin else b2_xmin
        inter_ymin = b1_ymin if b1_ymin > b2_ymin else b2_ymin
        inter_xmax = b1_xmax if b1_xmax < b2_xmax else b2_xmax
        inter_ymax = b1_ymax if b1_ymax < b2_ymax else b2_ymax
        inter_w = inter_xmax - inter_xmin
        inter_h = inter_ymax - inter_ymin
        inter_size = inter_w * inter_h
        bbox1_size = (b1_xmax - b1_xmin) * (b1_ymax - b1_ymin)
        bbox2_size = (b2_xmax - b2_xmin) * (b2_ymax - b2_ymin)
        overlap[0] = float32(inter_size / (bbox1_size + bbox2_size - inter_size))
    return overlap

@hybrid.script
def hybrid_apply_nms_fast(
    all_decode_bboxes,
    all_conf_scores,
    score_th,
    nms_th,
    eta,
    nms_top_k,
    keep_top_k,
    nms_label
):
    assert (all_decode_bboxes.shape[2] == all_conf_scores.shape[2]), "bboxes and scores have different size."
    class_index = output_tensor((1, nms_top_k), "int32")
    bbox_index = allocate((2,), "int32")

    score_index_vec = hybrid_get_max_score_index(all_conf_scores, score_th, nms_top_k, nms_label)

    adaptive_th = nms_th
    score_size = 0
    for i in range(nms_top_k):
        class_index[0, i] = -1
        if score_index_vec[1, i] > 0.0:
            score_size = score_size + 1

    adaptive_th = nms_th
    index = int32(0)
    cnt = 0
    for i in range(score_size):
        index = int32(score_index_vec[1, i])
        keep = True
        for k in range(cnt):
            if keep:
                kept_idx = class_index[0, k]
                bbox_index[0] = index
                bbox_index[1] = kept_idx
                overlap = hybrid_jaccard_overlap(all_decode_bboxes, bbox_index)
                keep = True if overlap[0] <= adaptive_th else False
        if keep:
            class_index[0, cnt] = index
            cnt = cnt + 1
        if keep and eta < 1.0 and adaptive_th > 0.5:
            adaptive_th = adaptive_th * eta
    return class_index


@hybrid.script
def hybrid_nms_output(
    all_conf_scores,
    all_decode_bboxes,
    all_indices,
    keep_top_k
):
    num = all_conf_scores.shape[0]
    output = output_tensor((num, 1, keep_top_k, 7), "float32")

    for i in range(num):
        for j in range(keep_top_k):
            c = int32(all_indices[i, j, 0])
            idx = int32(all_indices[i, j, 1])
            if c != -1:
                output[i, 0, j, 0] = float32(i)                      # image_id
                output[i, 0, j, 1] = float32(c)                      # class
                output[i, 0, j, 2] = all_conf_scores[i, c, idx]      # confidence
                output[i, 0, j, 3] = all_decode_bboxes[i, 0, idx, 0] # xmin
                output[i, 0, j, 4] = all_decode_bboxes[i, 0, idx, 1] # ymin
                output[i, 0, j, 5] = all_decode_bboxes[i, 0, idx, 2] # xmax
                output[i, 0, j, 6] = all_decode_bboxes[i, 0, idx, 3] # ymax
            else:
                output[i, 0, j, 0] = float32(i)  # image_id
                output[i, 0, j, 1] = -1.0        # class
                output[i, 0, j, 2] = -1.0        # confidence
                output[i, 0, j, 3] = 0.0         # xmin
                output[i, 0, j, 4] = 0.0         # ymin
                output[i, 0, j, 5] = 0.0         # xmax
                output[i, 0, j, 6] = 0.0         # ymax

    return output


@hybrid.script
def hybrid_nms_impl(
    loc,
    all_decode_bboxes,
    all_conf_scores,
    share_location,
    num_classes,
    bg_label_id,
    conf_th,
    nms_th,
    eta,
    nms_top_k,
    keep_top_k,
):
    num = loc.shape[0]
    all_indices = output_tensor((num, keep_top_k, 2), "int32")
    nms_label = allocate((3,), "int32")

    num_kept = 0
    for i in range(num):
        nms_label[0] = i
        num_det = 0
        for j in range(keep_top_k):
            all_indices[i, j, 0] = -1
            all_indices[i, j, 1] = -1
        for c in range(num_classes):
            if c != bg_label_id:
                nms_label[1] = c
                if share_location:
                    nms_label[2] = 0
                else:
                    nms_label[2] = c
                class_index = hybrid_apply_nms_fast(
                    all_decode_bboxes,
                    all_conf_scores,
                    conf_th,
                    nms_th,
                    eta,
                    nms_top_k,
                    keep_top_k,
                    nms_label
                )
                for k in range(keep_top_k):
                    idx = class_index[0, k]
                    if idx != -1:
                        if num_det < keep_top_k:
                            all_indices[i, num_det, 0] = c
                            all_indices[i, num_det, 1] = idx
                            num_det = num_det + 1
                        else:
                            min_score = all_conf_scores[i, c, idx]
                            min_indices = -1
                            for l in range(keep_top_k):
                                if all_conf_scores[i, c, l] < min_score:
                                    min_score = all_conf_scores[i, c, l]
                                    min_indices = l
                            if min_indices != -1:
                                all_indices[i, min_indices, 0] = c
                                all_indices[i, min_indices, 1] = idx
    output = hybrid_nms_output(
        all_conf_scores,
        all_decode_bboxes,
        all_indices,
        keep_top_k
    )
    return all_indices, output




@tvm.target.generic_func
def detection_output(
    loc,
    conf,
    priorbox,
    num_classes,
    share_location,
    background_label_id,
    nms_threshold,
    nms_top_k,
    nms_eta,
    code_type,
    variance_encoded_in_target,
    keep_top_k,
    confidence_threshold
):
    """Detection Output (Non-maximum suppression) operator for object detection.

    Parameters
    ----------

    loc : tvm.te.Tensor
        2-D tensor of location regression predictions.

    conf : tvm.te.Tensor
        2-D tensor of class probabilities.

    priorbox : tvm.te.Tensor
        3-D tensor of prior anchor boxes.

    bg_label_id : int
        Background label id.

    code_type : int
        Type of coding method for bounding boxes.

    conf_th : float
        Threshold to consider detections.

    keep_top_k : int
        Maximum number of bounding boxes per batch to be kept after NMS step.

    nms_th : float
        Threshold to be a positive prediction.

    nms_top_k : int
        Keep maximum top k detections after nms.

    num_classes : int
        Number of classes to be predicted.

    share_location boolean
        Whether to share one BBoxes among different classes.

    Returns
    -------
    out : tvm.te.Tensor or tuple of tvm.te.Tensor
        4-D tensor with shape [1, 1, keep_top_k, 7]
        The last indice means [image_id, label, confidence, xmin, ymin, xmax, ymax]

    """
    num_loc_classes = 1 if share_location else num_classes
    num_priors = int(int(priorbox.shape[2]) / 4)
    all_loc_preds, all_conf_scores, prior_bboxes, prior_variances = hybrid_prepare_detection_output(
        loc,
        conf,
        priorbox,
        tvm.tir.const(background_label_id, dtype="int32"),
        tvm.tir.const(code_type, dtype="int32"),
        tvm.tir.const(confidence_threshold, dtype="float"),
        tvm.tir.const(keep_top_k, dtype="int32"),
        tvm.tir.const(nms_threshold, dtype="float"),
        tvm.tir.const(nms_top_k, dtype="int32"),
        tvm.tir.const(num_classes, dtype="int32"),
        tvm.tir.const(share_location, dtype="bool"),
        tvm.tir.const(num_loc_classes, dtype="int32"),
        tvm.tir.const(variance_encoded_in_target, dtype="bool"),
        tvm.tir.const(nms_eta, dtype="float"),
        tvm.tir.const(num_priors, dtype="int32"),
    )

    clip_bbox = False
    all_decode_bboxes = hybrid_decode_bboxes_all(
        loc,
        all_loc_preds,
        prior_bboxes,
        prior_variances,
        tvm.tir.const(share_location, dtype="bool"),
        tvm.tir.const(num_loc_classes, dtype="int32"),
        tvm.tir.const(background_label_id, dtype="int32"),
        tvm.tir.const(code_type, dtype="int32"),
        tvm.tir.const(variance_encoded_in_target, dtype="bool"),
        tvm.tir.const(clip_bbox, dtype="bool"),
    )

    all_indices, out = hybrid_nms_impl(
        loc,
        all_decode_bboxes,
        all_conf_scores,
        tvm.tir.const(share_location, dtype="bool"),
        tvm.tir.const(num_classes, dtype="int32"),
        tvm.tir.const(background_label_id, dtype="int32"),
        tvm.tir.const(confidence_threshold, dtype="float"),
        tvm.tir.const(nms_threshold, dtype="float"),
        tvm.tir.const(nms_eta, dtype="float"),
        tvm.tir.const(nms_top_k, dtype="int32"),
        tvm.tir.const(keep_top_k, dtype="int32"),
    )

    return out
