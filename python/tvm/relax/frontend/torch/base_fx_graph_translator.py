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

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""PyTorch FX frontend of Relax."""
from typing import Callable, Dict, List, Optional, Tuple, Union
import abc
from functools import reduce

from tvm import relax


class BaseFXGraphImporter(metaclass=abc.ABCMeta):
    """Base class for FX Graph Importer."""

    import torch  # type: ignore
    from torch import fx

    def __init__(self) -> None:
        import torch  # type: ignore
        from torch import fx

        self.env: Dict[fx.Node, relax.Expr] = {}
        self.params: Dict[torch.Tensor, relax.Expr] = {}
        self.block_builder: relax.BlockBuilder = None
        self.convert_map: Dict[[Union[str, torch.Module]], Callable] = self.create_convert_map()

    ########## Utilities ##########
    def _fetch_attr(self, model, target: str):
        import torch  # type: ignore

        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced non existing target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        if isinstance(attr_itr, torch.Tensor):
            # Its possible for the resulting tensor to be a parameter.
            # If so, return the parameter instead.
            if attr_itr in self.params:
                return self.params[attr_itr]
            return self._convert_torch_tensor_to_relax(attr_itr)
        return attr_itr

    @staticmethod
    def _convert_data_type(input_type: Union[str, torch.dtype], env: Optional[Dict] = None):
        """converts the PyTorch scalar type input_type to a TVM dtype."""
        import torch  # type: ignore

        if env is not None and input_type in env:
            input_type = env[input_type]

        input_type = input_type.lower() if isinstance(input_type, str) else input_type
        if input_type in ["float", "float32", "torch.float32", torch.float32]:
            return "float32"
        elif input_type in ["float16", "torch.float16", torch.float16]:
            return "float16"
        elif input_type in ["int64", "torch.int64", torch.int64]:
            return "int64"
        elif input_type in ["int32", "torch.int32", torch.int32]:
            return "int32"
        elif input_type in ["bool", "torch.bool", torch.bool]:
            return "bool"
        else:
            raise NotImplementedError("input_type {} is not handled yet".format(input_type))

    @staticmethod
    def _convert_torch_tensor_to_relax(tensor: torch.Tensor) -> relax.Var:
        tensor = tensor.detach().cpu()
        dtype = BaseFXGraphImporter._convert_data_type(str(tensor.data.dtype))
        return relax.const(tensor.data.numpy(), dtype)

    @staticmethod
    def shape_of(tensor):
        """Get the shape of a tensor."""
        import torch  # type: ignore

        if isinstance(tensor, relax.Expr):
            if not isinstance(tensor.struct_info, relax.TensorStructInfo):
                raise TypeError("The input Expr of shape_of should be a Tensor")
            return tensor.struct_info.shape
        elif isinstance(tensor, torch.Tensor):
            return tensor.shape
        raise ValueError("Unsupported type: {}".format(type(tensor)))

    def retrieve_args(self, node: fx.Node):
        return self._retrieve_args(node.args)

    def _retrieve_args(self, node):
        from torch import fx

        if isinstance(node, fx.Node):
            return self.env[node]
        elif isinstance(node, tuple):
            return tuple(self._retrieve_args(x) for x in node)
        elif isinstance(node, list):
            return [self._retrieve_args(x) for x in node]
        elif isinstance(node, dict):
            return {self._retrieve_args(k): self._retrieve_args(v) for k, v in node.items()}
        else:
            return node

    @staticmethod
    def _promote_binary_op_args(lhs, rhs):
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return lhs, rhs
        elif isinstance(lhs, relax.Expr):
            assert isinstance(lhs.struct_info, relax.TensorStructInfo)
            return lhs, relax.const(rhs, lhs.struct_info.dtype)
        elif isinstance(rhs, relax.Expr):
            assert isinstance(rhs.struct_info, relax.TensorStructInfo)
            return relax.const(lhs, rhs.struct_info.dtype), rhs
        else:
            assert False

    def _call_binary_op(self, op, lhs, rhs):
        lhs, rhs = self._promote_binary_op_args(lhs, rhs)
        return self.block_builder.emit(op(lhs, rhs))

    ########## Unary Ops ##########

    def _unary_op(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            return self.block_builder.emit(op(self.env[node.args[0]]))

        return convert

    def _clamp(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        a_min = args[1] if len(args) > 1 else node.kwargs["min"]
        a_max = args[2] if len(args) > 2 else node.kwargs["max"]
        if not isinstance(a_min, (int, float)):
            raise ValueError(
                f"TVM only supports constant min value for torch.clamp/clip, "
                f"but got {a_min} with type {type(a_min)}"
            )
        if not isinstance(a_max, (int, float)):
            raise ValueError(
                f"TVM only supports constant max value for torch.clamp/clip, "
                f"but got {a_max} with type {type(a_max)}"
            )
        return self.block_builder.emit(relax.op.clip(args[0], a_min, a_max))

    def _gelu(self, node: fx.Node) -> relax.Expr:
        approximate = node.kwargs.get("approximate", "none")
        if approximate == "none":
            return self.block_builder.emit(relax.op.nn.gelu(self.env[node.args[0]]))
        elif approximate == "tanh":
            return self.block_builder.emit(relax.op.nn.gelu_tanh(self.env[node.args[0]]))
        else:
            raise KeyError("Unregonized approximate algorithm for gelu: {}.".format(approximate))

    def _hardsigmoid(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        x0 = relax.op.add(x, relax.const(3, dtype))
        x1 = relax.op.clip(x0, 0, 6)
        return self.block_builder.emit(relax.op.divide(x1, relax.const(6, dtype)))

    def _hardswish(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        x0 = relax.op.add(x, relax.const(3, dtype))
        x1 = relax.op.clip(x0, 0, 6)
        x2 = relax.op.divide(x1, relax.const(6, dtype))
        return self.block_builder.emit(relax.op.multiply(x, x2))

    def _hardtanh(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.args[1] if len(args) > 1 else node.kwargs("min_val", -1.0)
        max_val = node.args[2] if len(args) > 2 else node.kwargs("max_val", 1.0)
        return self.block_builder.emit(relax.op.clip(x, min_val, max_val))

    def _leakyrelu_functional(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("negative_slope", 0.01)
        return self.block_builder.emit(relax.op.nn.leakyrelu(x, alpha))

    def _log_softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.log_softmax(x, dim))

    def _round(self, node: fx.Node) -> relax.Expr:
        if "decimals" in node.kwargs and node.kwargs["decimals"] != 0:
            raise ValueError("specifying decimals for round is not supported yet")
        arg = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.round(arg))

    def _softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.softmax(x, dim))

    def _tril_triu(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            x = self.env[node.args[0]]
            k = node.args[1] if len(node.args) > 1 else node.kwargs.get("diagonal", 0)
            assert isinstance(k, int)
            return self.block_builder.emit(op(x, k))

        return convert

    ########## Binary Ops ##########

    def _binary_op(self, relax_op: Callable, intrinsic_op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            lhs, rhs = self.retrieve_args(node)
            if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
                return self._call_binary_op(relax_op, lhs, rhs)
            elif isinstance(lhs, relax.expr.Constant):
                return self._call_binary_op(
                    relax_op, lhs, relax.const(rhs, dtype=lhs.struct_info.dtype)
                )
            elif isinstance(rhs, relax.expr.Constant):
                return self._call_binary_op(
                    relax_op, relax.const(lhs, dtype=rhs.struct_info.dtype), rhs
                )
            return intrinsic_op(lhs, rhs)

        return convert

    ########## Neural Network ##########

    def _native_batch_norm_legit_no_training(self, node: fx.Node) -> relax.Var:
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        weight = self.env.get(node.args[1], relax.const(np.ones(channel), dtype=dtype))
        bias = self.env.get(node.args[2], relax.const(np.zeros(channel), dtype=dtype))
        running_mean = self.env.get(node.args[3], relax.const(np.zeros(channel), dtype=dtype))
        running_var = self.env.get(node.args[4], relax.const(np.ones(channel), dtype=dtype))
        momentum = node.args[5] if len(node.args) > 5 else node.kwargs.get("momentum", 0.1)
        eps = node.args[6] if len(node.args) > 6 else node.kwargs.get("eps", 1e-05)

        return self.block_builder.emit(
            relax.op.nn.batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                axis=1,
                epsilon=eps,
                momentum=momentum,
            )
        )

    def _adaptive_avg_pool2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1]
        return self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, output_size, layout="NCHW")
        )

    def _addmm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        y = self.env[node.args[1]]
        z = self.env[node.args[2]]
        alpha = node.kwargs.get("alpha", 1)
        beta = node.kwargs.get("beta", 1)

        res = None
        if alpha != 0:
            res = self.block_builder.emit(relax.op.linear_algebra.matmul(y, z, out_dtype="float32"))
            if alpha != 1:
                dtype = res.struct_info.dtype
                res = self.block_builder.emit(relax.op.multiply(res, relax.const(alpha, dtype)))
        if beta != 0:
            dtype = x.struct_info.dtype
            if beta != 1:
                bias = self.block_builder.emit(relax.op.multiply(x, relax.const(beta, dtype)))
            else:
                bias = x
            res = bias if res is None else self.block_builder.emit(relax.op.add(bias, res))
        return res

    def _avg_pool2d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[int] = 0,
        ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        stride = kernel_size if stride is None or stride == [] else stride
        return self.block_builder.emit(
            relax.op.nn.avg_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                layout="NCHW",
            )
        )

    def _avg_pool2d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        return self._avg_pool2d_impl(x, kernel_size, stride, padding, ceil_mode)

    def _baddbmm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        a = self.env[node.args[1]]
        b = self.env[node.args[2]]
        alpha = node.kwargs.get("alpha", 1)
        beta = node.kwargs.get("beta", 1)

        res = None
        if alpha != 0:
            res = self.block_builder.emit(relax.op.matmul(a, b))
            if alpha != 1:
                dtype = res.struct_info.dtype
                res = self.block_builder.emit(relax.op.multiply(res, relax.const(alpha, dtype)))
        if beta != 0:
            dtype = x.struct_info.dtype
            if beta != 1:
                bias = self.block_builder.emit(relax.op.multiply(x, relax.const(beta, dtype)))
            else:
                bias = x
            res = bias if res is None else self.block_builder.emit(relax.op.add(res, bias))
        return res

    def _conv1d_transpose_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ) -> relax.Var:
        conv1d_transpose = self.block_builder.emit(
            relax.op.nn.conv1d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv1d_transpose

        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1))
        return self.block_builder.emit(relax.op.add(conv1d_transpose, bias))

    def _conv1d_transpose_functional(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv1d_transpose_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv2d_transpose_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ) -> relax.Var:
        conv2d_transpose = self.block_builder.emit(
            relax.op.nn.conv2d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv2d_transpose

        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv2d_transpose, bias))

    def _conv2d_transpose_functional(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv2d_transpose_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv1d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ) -> relax.Var:
        conv1d = self.block_builder.emit(
            relax.op.nn.conv1d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv1d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1))
        return self.block_builder.emit(relax.op.add(conv1d, bias))

    def _conv1d_functional(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv2d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ):
        conv2d = self.block_builder.emit(
            relax.op.nn.conv2d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv2d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv2d, bias))

    def _conv2d_functional(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv3d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ):
        conv3d = self.block_builder.emit(
            relax.op.nn.conv3d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCDHW",
                kernel_layout="OIDHW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv3d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv3d, bias))

    def _conv3d_functional(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv3d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _einsum(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        operands = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.einsum(operands, args[0]))

    def _embedding_impl(
        self,
        x,
        weight,
    ) -> relax.Var:
        x = self.block_builder.emit(relax.op.astype(x, "int32"))

        ndim = x.struct_info.ndim
        if ndim == 1:
            return self.block_builder.emit(relax.op.take(weight, x, axis=0))
        else:
            x_shape = x.struct_info.shape.values
            emb_size = weight.struct_info.shape.values[-1]
            x = self.block_builder.emit(relax.op.reshape(x, shape=[-1]))
            embedding = self.block_builder.emit(relax.op.take(weight, x, axis=0))
            return self.block_builder.emit(relax.op.reshape(embedding, [*x_shape, emb_size]))

    def _group_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        num_groups = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05

        dim = len(self.shape_of(x))
        return self.block_builder.emit(
            relax.op.nn.group_norm(
                x,
                gamma,
                beta,
                num_groups=num_groups,
                channel_axis=1,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _layer_norm_impl(self, x, gamma, beta, eps, normalized_shape) -> relax.Var:
        from torch.fx.immutable_collections import immutable_list
        import numpy as np  # type: ignore

        if isinstance(normalized_shape, (immutable_list, tuple)):
            normalized_shape = tuple(normalized_shape)
        else:
            try:
                normalized_shape = self.env[normalized_shape]
            except TypeError:
                normalized_shape = tuple(normalized_shape)

        dim_num = len(normalized_shape)
        axes = list(range(-dim_num, 0))

        if gamma is None:
            shape_tuple = [int(s) for s in normalized_shape]
            gamma = relax.const(np.ones(shape_tuple), x.struct_info.dtype)
        if beta is None:
            shape_tuple = [int(s) for s in normalized_shape]
            beta = relax.const(np.zeros(shape_tuple), x.struct_info.dtype)

        return self.block_builder.emit(
            relax.op.nn.layer_norm(
                x,
                gamma,
                beta,
                axes=axes,
                epsilon=eps,
            )
        )

    def _layer_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        normalized_shape = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05
        return self._layer_norm_impl(x, gamma, beta, eps, normalized_shape)

    def _linear(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        return self.block_builder.emit(relax.op.linear(x, weight, bias, "float32"))

    def _max_pool2d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[int] = 0,
        dilation: Optional[int] = 1,
        ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        stride = kernel_size if stride is None else stride
        return self.block_builder.emit(
            relax.op.nn.max_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                layout="NCHW",
            )
        )

    def _max_pool2d_functional(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        return self._max_pool2d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _scaled_dot_product_attention(self, node: fx.Node) -> relax.Var:
        transpose_S_H = lambda tensor: relax.op.permute_dims(tensor, [0, 2, 1, 3])
        query = transpose_S_H(self.env[node.args[0]])
        key = transpose_S_H(self.env[node.args[1]])
        value = transpose_S_H(self.env[node.args[2]])
        attn_mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask", None)
        dropout_p = node.args[4] if len(node.args) > 4 else node.kwargs.get("dropout_p", 0.0)
        assert dropout_p == 0.0, "Dropout is not supported"
        is_causal = node.args[5] if len(node.args) > 5 else node.kwargs.get("is_causal", False)
        causal_mask = "TopLeft" if is_causal else None

        if attn_mask is not None:
            attn_mask = self.env[attn_mask]
            msg = "Only a float mask is supported for the attn_mask input."
            assert "float" in attn_mask.struct_info.dtype, msg

        return self.block_builder.emit(
            transpose_S_H(
                relax.op.nn.attention(query, key, value, bias=attn_mask, causal_mask=causal_mask)
            )
        )

    def _unbind(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        assert isinstance(dim, int), "Expected 2nd argument of unbind as int"
        selections = self.shape_of(x)[dim].value
        n_section = list(range(1, selections + 1))
        ret, split = [], self.block_builder.emit(relax.op.split(x, n_section, dim))
        for i in range(selections):
            ret.append(self.block_builder.emit(relax.op.squeeze(split[i], axis=dim)))
        return self.block_builder.emit(relax.Tuple(ret))

    def _upsample_bilinear2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", True)
        )
        scale_factor = node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", None)
        method = "linear"
        coord_trans = "align_corners" if align_corners else "half_pixel"

        if size is None:
            shape = self.shape_of(x)
            assert isinstance(shape, relax.ShapeExpr)
            if isinstance(scale_factor, (tuple, list)):
                assert len(scale_factor) == len(shape) - 2
                size = tuple(
                    int(shape[i].value * scale_factor[i - 2]) for i in range(2, len(shape))
                )
            else:
                size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

        return self.block_builder.emit(
            relax.op.image.resize2d(
                x, size, layout="NCHW", method=method, coordinate_transformation_mode=coord_trans
            )
        )

    def _interpolate(self, node: fx.Node) -> relax.Var:
        # torch.nn.functional.interpolate(
        #   input, size=None, scale_factor=None, mode='nearest', align_corners=None,
        #   recompute_scale_factor=None, antialias=False)
        # (TODO) this is a temporary implementation for interpolate that only considers NCHW layout
        # it basically replicates the implementation in tvm.relay.frontend.pytorch
        data = self.env[node.args[0]]
        size = (
            node.args[1]
            if len(node.args) > 1
            else (node.kwargs["size"] if "size" in node.kwargs else None)
        )
        scale_factor = (
            node.args[2]
            if len(node.args) > 2
            else (node.kwargs["scale_factor"] if "scale_factor" in node.kwargs else None)
        )
        method = (
            node.args[3]
            if len(node.args) > 3
            else (node.kwargs["mode"] if "mode" in node.kwargs else "nearest")
        )
        align_corners = (
            node.args[4]
            if len(node.args) > 4
            else (node.kwargs["align_corners"] if "align_corners" in node.kwargs else None)
        )
        recompute_scale_factor = (
            node.args[5]
            if len(node.args) > 5
            else (
                node.kwargs["recompute_scale_factor"]
                if "recompute_scale_factor" in node.kwargs
                else None
            )
        )
        antialias = (
            node.args[6]
            if len(node.args) > 6
            else (node.kwargs["antialias"] if "antialias" in node.kwargs else False)
        )

        assert recompute_scale_factor is None
        assert antialias is False

        if size is None:
            shape = self.shape_of(data)
            assert isinstance(shape, relax.ShapeExpr)
            if isinstance(scale_factor, tuple):
                assert len(scale_factor) == len(shape) - 2
                size = tuple(
                    int(shape[i].value * scale_factor[i - 2]) for i in range(2, len(shape))
                )
            else:
                size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

        if method.startswith("nearest"):
            method = "nearest_neighbor"
        elif method[0:2] == "bi":
            method = method[2:]

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return self.block_builder.emit(
            relax.op.image.resize2d(
                data, size, layout="NCHW", method=method, coordinate_transformation_mode=coord_trans
            )
        )

    ########## Statistical ##########

    def _mean(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.mean(x, dim, keepdims=keepdim))

    def _sum(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        keepdim = node.kwargs["keepdim"] if "keepdim" in node.kwargs else False
        if len(args) == 1:
            return self.block_builder.emit(relax.op.sum(args[0], keepdims=keepdim))
        return self.block_builder.emit(relax.op.sum(args[0], args[1]))

    ########## Search ##########

    def _argmax_argmin(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node):
            x = self.env[node.args[0]]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
            keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
            return self.block_builder.emit(op(x, dim, keepdim))

        return convert

    ########## Tensor Manipulation ##########

    def _cat(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        axis = args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.concat(args[0], axis=axis))

    def _cumsum(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]

        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        else:
            dtype = None
        if "out" in node.kwargs:
            raise ValueError("specifying out for cumsum is not supported yet")

        return self.block_builder.emit(relax.op.cumsum(x, dim, dtype))

    def _expand(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        sizes = args[1:] if len(args) > 2 else args[1]
        broadcast_shape, in_shape = [], self.shape_of(args[0])
        for idx, i in enumerate(sizes):
            if isinstance(i, int) and i == -1:
                broadcast_shape.append(in_shape[idx])
            else:
                broadcast_shape.append(i)
        return self.block_builder.emit(relax.op.broadcast_to(args[0], broadcast_shape))

    def _flatten_impl(self, x, start_dim, end_dim) -> relax.Var:
        shape = self.shape_of(x)
        start_dim = start_dim if start_dim >= 0 else len(shape) + start_dim
        end_dim = end_dim if end_dim >= 0 else len(shape) + end_dim
        flattened = reduce(lambda x, y: x * y, [shape[i] for i in range(start_dim, end_dim + 1)])
        new_shape = (
            [shape[i] for i in range(0, start_dim)]
            + [flattened]
            + [shape[i] for i in range(end_dim + 1, len(shape))]
        )
        return self.block_builder.emit(relax.op.reshape(x, new_shape))

    def _flatten(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        start_dim = node.args[1] if len(node.args) >= 2 else node.kwargs.get("start_dim", 0)
        end_dim = node.args[2] if len(node.args) == 3 else node.kwargs.get("end_dim", -1)
        return self._flatten_impl(x, start_dim, end_dim)

    def _permute(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.permute_dims(x, dims))

    def _repeat(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.tile(x, dims))

    def _reshape(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.reshape(x, dims))

    def _select(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = relax.const(node.args[2], "int64")
        return self.block_builder.emit(relax.op.take(x, index, dim))

    def _slice(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        axes = [node.args[1]]
        begin = [node.args[2]]
        end = [node.args[3]]
        stride = [node.args[4] if len(node.args) > 4 else 1]
        return self.block_builder.emit(relax.op.strided_slice(x, axes, begin, end, stride))

    def _split(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        split_size = node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if isinstance(split_size, (list, tuple)):
            n_section = []
            for s in split_size[:-1]:
                cum_sum = 0 if not n_section else n_section[-1]
                n_section.append(s + cum_sum)
        else:
            n_section = (self.shape_of(x)[dim].value + split_size - 1) // split_size
        return self.block_builder.emit(relax.op.split(x, n_section, dim))

    def _squeeze(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        return self.block_builder.emit(relax.op.squeeze(x, dim))

    def _sym_size_int(self, node: fx.node.Node) -> relax.Expr:
        x = self.env[node.args[0]]
        shape = self.shape_of(x)
        idx = node.args[1]
        return self.block_builder.emit(relax.const(shape[idx].value, "int32"))

    def _tile(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.tile(x, dims))

    def _transpose(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        full_idx = list(range(len(self.shape_of(args[0]))))
        full_idx[args[1]], full_idx[args[2]] = full_idx[args[2]], full_idx[args[1]]
        return self.block_builder.emit(relax.op.permute_dims(args[0], full_idx))

    ########## Tensor Creation ##########

    def _to_copy(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        x = self.env[node.args[0]]
        if len(node.args) == 2:
            if isinstance(node.args[1], torch.dtype):
                dtype = self._convert_data_type(node.args[1], self.env)
                return self.block_builder.emit(relax.op.astype(x, dtype))
        elif "dtype" in node.kwargs:
            dtype = self._convert_data_type(node.kwargs["dtype"], self.env)
            return self.block_builder.emit(relax.op.astype(x, dtype))
        return x

    def _arange(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        start_end_step = [None, None, None]
        if "start" in node.kwargs:
            start_end_step[0] = node.kwargs["start"]
        if "end" in node.kwargs:
            start_end_step[1] = node.kwargs["end"]
        if "step" in node.kwargs:
            start_end_step[2] = node.kwargs["step"]

        if len(node.args) == 1:
            assert start_end_step[1] is None
            start_end_step[1] = node.args[0]
        elif len(node.args) == 2:
            assert start_end_step[0] is None
            assert start_end_step[1] is None
            start_end_step[0] = node.args[0]
            start_end_step[1] = node.args[1]
        elif len(node.args) == 3:
            assert start_end_step[0] is None
            assert start_end_step[1] is None
            assert start_end_step[2] is None
            start_end_step[0] = node.args[0]
            start_end_step[1] = node.args[1]
            start_end_step[2] = node.args[2]

        if start_end_step[0] is None:
            start_end_step[0] = 0
        if start_end_step[2] is None:
            start_end_step[2] = 1

        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        elif any([isinstance(x, float) for x in start_end_step]):
            dtype = self._convert_data_type(torch.get_default_dtype())
        else:
            dtype = "int64"
        start_end_step = [
            self.env[x] if isinstance(x, torch.fx.Node) else x for x in start_end_step
        ]
        return self.block_builder.emit(relax.op.arange(*start_end_step, dtype=dtype))

    def _empty(self, node: fx.Node) -> relax.Var:
        dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        return self.block_builder.emit(relax.op.zeros(node.args[0], dtype))

    def _fill(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        value = args[1] if isinstance(args[1], relax.Expr) else relax.const(args[1], dtype)
        return self.block_builder.emit(relax.op.full(x.struct_info.shape, value, dtype))

    def _new_ones(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        self_var = args[0]
        size = args[1] if isinstance(args[1], (list, tuple)) else args[1:]
        if not isinstance(size, (list, tuple)):
            size = (size,)
        size = relax.ShapeExpr(size)
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(1, self_var.struct_info.dtype),
                self_var.struct_info.dtype,
            )
        )

    ########## Others ##########

    def _getitem(self, node: fx.Node) -> relax.Var:
        from torch import fx

        x = self.env[node.args[0]]
        if isinstance(x, (list, tuple, relax.ShapeExpr, relax.Tuple)):
            return x[node.args[1]]
        elif isinstance(x, relax.Var):
            if isinstance(x.struct_info, relax.TupleStructInfo):
                return self.block_builder.emit(relax.TupleGetItem(x, node.args[1]))

            assert isinstance(x.struct_info, relax.TensorStructInfo)
            take_indices = []
            take_axes = []
            stride_begin = []
            stride_end = []
            stride = []
            stride_axes = []
            expand_dim = []
            i = 0
            shape = self.shape_of(x)
            non_ellipsis_cnt = 0
            for index in node.args[1]:
                if isinstance(index, (int, slice, fx.Node)):
                    non_ellipsis_cnt += 1
            for index in node.args[1]:
                if isinstance(index, int):
                    stride_begin.append(index)
                    stride_end.append(index + 1)
                    stride.append(1)
                    stride_axes.append(i)
                    i = i + 1
                elif isinstance(index, slice):
                    stride_begin.append(0 if index.start is None else index.start)
                    stride_end.append(shape[i] if index.stop is None else index.stop)
                    stride.append(1 if index.step is None else index.step)
                    stride_axes.append(i)
                    i = i + 1
                elif index is None:
                    expand_dim.append(len(stride_axes) + len(expand_dim))
                elif index is Ellipsis:
                    for _ in range(len(shape) - non_ellipsis_cnt):
                        stride_begin.append(0)
                        stride_end.append(shape[i])
                        stride.append(1)
                        stride_axes.append(i)
                        i += 1
                elif isinstance(index, fx.Node):
                    node_index = self.env[index]
                    if not isinstance(node_index, relax.Expr):
                        raise ValueError(
                            "Unsupported index type for relax.op.take: " + str(type(node_index))
                        )
                    take_indices.append(node_index)
                    take_axes.append(i)
                    i = i + 1
                else:
                    raise ValueError("Unsupported index type: " + str(type(index)))
            while i < len(shape):
                stride_begin.append(0)
                stride_end.append(shape[i])
                stride.append(1)
                stride_axes.append(i)
                i += 1
            taken = x
            if len(take_indices) > 1:
                raise ValueError("Multiple tensors as index not yet supported")
            for each_index, each_axis in zip(take_indices, take_axes):
                taken = self.block_builder.emit(relax.op.take(taken, each_index, each_axis))
            sliced = self.block_builder.emit(
                relax.op.strided_slice(taken, stride_axes, stride_begin, stride_end, stride)
            )
            sliced_shape = list(self.shape_of(sliced))
            for i in expand_dim:
                sliced_shape.insert(i, 1)
            return self.block_builder.emit(relax.op.reshape(sliced, sliced_shape))
        elif isinstance(x, relax.Constant):
            dtype = x.struct_info.dtype
            return relax.const(x.data.numpy()[node.args[1]], dtype)
        else:
            assert False

    @abc.abstractmethod
    def create_input_vars(self) -> List[relax.Var]:
        """"""
        ...

    @abc.abstractmethod
    def create_convert_map(self) -> Dict[[Union[str, torch.nn.Module]], Callable]:
        """"""
        ...

    def update_convert_map(self, custom_convert_map: dict):
        """Update self.convert_map with custom convert map

        Parameters
        ----------
        custom_convert_map : Dictionary of str to Relax op
            A custom op conversion map in the same format as self.convert_map
        """

        self.convert_map.update(custom_convert_map)
