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
 * \file priorbox_op.cc
 * \brief PriorBox related operators
 */
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/op.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(PriorBoxAttrs);

bool PriorBoxRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* layer = types[0].as<TensorTypeNode>();
  const auto* input = types[1].as<TensorTypeNode>();
  const PriorBoxAttrs* param = attrs.as<PriorBoxAttrs>();
  const auto& ishape = input->shape;
  const auto& lshape = layer->shape;
  CHECK_EQ(ishape.size(), 4) << "Input data should be 4D: "
                                "[batch, channel, height, width]";
  CHECK_EQ(lshape.size(), 4) << "Layer data should be 4D: "
                                "[batch, channel, height, width]";

  IndexExpr l_height = lshape[2];
  IndexExpr l_width  = lshape[3];
  int num_priors_ = static_cast<int>(param->aspect_ratios.size()) * static_cast<int>(param->min_size.size()) + static_cast<int>(param->max_size.size());
  // since input sizes are same in each batch, we could share MultiBoxPrior
  std::vector<IndexExpr> oshape({1, 2, l_height * l_width * num_priors_ * 4});

  // assign output type
  reporter->Assign(types[2], TensorType(oshape, layer->dtype));

  return true;
}

Expr MakePriorBoxAttrs(Expr layer, Expr input, Array<IndexExpr> min_size, Array<IndexExpr> max_size,
                       Array<IndexExpr> aspect_ratios, Array<IndexExpr> variance, double offset,
                       bool flip, bool clip) {
  auto attrs = make_object<PriorBoxAttrs>();
  attrs->min_size = std::move(min_size);
  attrs->max_size = std::move(max_size);
  attrs->aspect_ratios = std::move(aspect_ratios);
  attrs->variance = std::move(variance);
  attrs->offset = offset;
  attrs->flip = flip;
  attrs->clip = clip;
  static const Op& op = Op::Get("vision.priorbox");
  return Call(op, {layer, input}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.priorbox").set_body_typed(MakePriorBoxAttrs);

RELAY_REGISTER_OP("vision.priorbox")
    .describe(R"doc("Generate prior(anchor) boxes from layer and input, and parameters.")doc" TVM_ADD_FILELINE)
    .set_attrs_type<PriorBoxAttrs>()
    .set_num_inputs(2)
    .add_argument("layer", "Tensor", "The layer tensor.")
    .add_argument("input", "Tensor", "The input tensor.")
    .set_support_level(5)
    .add_type_rel("PriorBox", PriorBoxRel);

}  // namespace relay
}  // namespace tvm
