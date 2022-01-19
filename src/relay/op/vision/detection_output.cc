/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file detection_output.cc
 * \brief Detection Output operators
 */
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(DetectionOutputAttrs);

bool DetectionOutputRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* priorbox = types[2].as<TensorTypeNode>();
  if (priorbox == nullptr) return false;
  const DetectionOutputAttrs* param = attrs.as<DetectionOutputAttrs>();
  ICHECK(param != nullptr);

  std::vector<IndexExpr> oshape({1, 1, param->keep_top_k, 7});

  // assign output type
  reporter->Assign(types[3], TensorType(oshape, priorbox->dtype));
  return true;
}

Expr MakeDetectionOutput(Expr loc, Expr conf, Expr priorbox,
              int bg_label_id, int code_type, double conf_th, int keep_top_k,
              double nms_th, int nms_top_k, int num_classes, bool share_location) {
  auto attrs = make_object<DetectionOutputAttrs>();
  attrs->bg_label_id = bg_label_id;
  attrs->code_type = code_type;
  attrs->conf_th = conf_th;
  attrs->keep_top_k = keep_top_k;
  attrs->nms_th = nms_th;
  attrs->nms_top_k = nms_top_k;
  attrs->num_classes = num_classes;
  attrs->share_location = share_location;
  static const Op& op = Op::Get("vision.detection_output");
  return Call(op, {loc, conf, priorbox}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.detection_output").set_body_typed(MakeDetectionOutput);

RELAY_REGISTER_OP("vision.detection_output")
    .describe(R"doc(Detection Output.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("loc", "Tensor", "Location tensor.")
    .add_argument("conf", "Tensor", "Confidence tensor.")
    .add_argument("priorbox", "Tensor", "PriorBox tensor")
    .set_support_level(5)
    .add_type_rel("DetectionOutput", DetectionOutputRel);

}  // namespace relay
}  // namespace tvm
