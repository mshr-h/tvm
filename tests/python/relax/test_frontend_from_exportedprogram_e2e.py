import torch
from torch.nn import Module
from torch.export import export
import pytest

import tvm
from tvm import relax
import tvm.testing
from tvm.relax.frontend.torch import from_exportedprogram


def verify_model(torch_model, example_args, example_kwargs={}, target: str = "llvm", dev=tvm.cpu()):
    # PyTorch
    exported_program = export(torch_model, args=example_args, kwargs=example_kwargs)
    torch_output: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    mod = from_exportedprogram(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    exe = relax.build(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    tvm_args = [tvm.nd.from_dlpack(x) for x in example_args]
    tvm_output = vm["main"](*tvm_args)

    if isinstance(torch_output, tuple):
        expected = torch.stack(torch_output)
        actual = torch.stack([torch.from_numpy(x.numpy()) for x in tvm_output])
    else:
        expected = torch_output
        actual = torch.from_numpy(tvm_output[0].numpy())

    torch.testing.assert_close(
        actual.shape,
        expected.shape,
        msg=f"expected: {expected.shape}, actual: {actual.shape}",
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
        equal_nan=True,
    )


def test_unary():
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    # sin
    class Sin(Module):
        def forward(self, input):
            return torch.sin(input)

    verify_model(Sin(), example_args)

    # cos
    class Cos(Module):
        def forward(self, input):
            return torch.cos(input)

    verify_model(Cos(), example_args)

    # tan
    class Tan(Module):
        def forward(self, input):
            return torch.tan(input)

    verify_model(Tan(), example_args)

    # asin
    class Asin(Module):
        def forward(self, input):
            return torch.asin(input)

    verify_model(Asin(), example_args)

    # acos
    class Acos(Module):
        def forward(self, input):
            return torch.acos(input)

    verify_model(Acos(), example_args)

    # atan
    class Atan(Module):
        def forward(self, input):
            return torch.atan(input)

    verify_model(Atan(), example_args)

    # sinh
    class Sinh(Module):
        def forward(self, input):
            return torch.sinh(input)

    verify_model(Sinh(), example_args)

    # cosh
    class Cosh(Module):
        def forward(self, input):
            return torch.cosh(input)

    verify_model(Cosh(), example_args)

    # tanh
    class Tanh(Module):
        def forward(self, input):
            return torch.tanh(input)

    verify_model(Tanh(), example_args)

    # asinh
    class Asinh(Module):
        def forward(self, input):
            return torch.asinh(input)

    verify_model(Asinh(), example_args)

    # acosh
    class Acosh(Module):
        def forward(self, input):
            return torch.acosh(input)

    verify_model(Acosh(), example_args)

    # atanh
    class Atanh(Module):
        def forward(self, input):
            return torch.atanh(input)

    verify_model(Atanh(), example_args)

    # exp
    class Exp(Module):
        def forward(self, input):
            return torch.exp(input)

    verify_model(Exp(), example_args)

    # neg
    class Neg(Module):
        def forward(self, input):
            return -input

    verify_model(Neg(), example_args)

    # sqrt
    class Sqrt(Module):
        def forward(self, input):
            return torch.sqrt(input)

    verify_model(Sqrt(), example_args)

    class ReLU0(Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, input):
            return self.relu(input)

    class ReLU1(Module):
        def forward(self, input):
            return torch.nn.functional.relu(input)

    verify_model(ReLU0(), example_args)
    verify_model(ReLU1(), example_args)

    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, input):
            return self.relu6(input)

    verify_model(ReLU6(), example_args)

    # round
    class Round(Module):
        def forward(self, input):
            return torch.round(input)

    verify_model(Round(), example_args)

    # rsqrt
    class Rsqrt(Module):
        def forward(self, input):
            return torch.rsqrt(input)

    verify_model(Rsqrt(), example_args)

    # sigmoid
    class Sigmoid(Module):
        def __init__(self):
            super().__init__()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, input):
            return self.sigmoid(input)

    class Sigmoid2(Module):
        def forward(self, input):
            return torch.sigmoid(input)

    verify_model(Sigmoid(), example_args)
    verify_model(Sigmoid2(), example_args)


def test_clamp():
    class Clamp(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.1, max=0.5)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Clamp(), example_args)


def test_dropout():
    class Dropout1(Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, input):
            return self.dropout(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Dropout1().eval(), example_args)


def test_gelu():
    class Gelu(Module):
        def __init__(self):
            super().__init__()
            self.gelu = torch.nn.GELU()

        def forward(self, input):
            return self.gelu(input)

    class Gelu2(Module):
        def forward(self, input):
            return torch.nn.functional.gelu(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Gelu(), example_args)
    verify_model(Gelu2(), example_args)


def test_hardsigmoid():
    class Hardsigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hs = torch.nn.Hardsigmoid()

        def forward(self, input):
            return self.hs(input)

    class Hardsigmoid2(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardsigmoid(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Hardsigmoid(), example_args)
    verify_model(Hardsigmoid2(), example_args)


def test_hardswish():
    class Hardswish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hs = torch.nn.Hardswish()

        def forward(self, input):
            return self.hs(input)

    class Hardswish2(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardswish(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Hardswish(), example_args)
    verify_model(Hardswish2(), example_args)


def test_leakyrelu():
    import torch
    from torch.nn import Module

    torch.set_grad_enabled(False)

    class LeakyReLU0(Module):
        def __init__(self):
            super().__init__()
            self.leakyrelu = torch.nn.LeakyReLU(0.02)

        def forward(self, input):
            return self.leakyrelu(input)

    class LeakyReLU1(Module):
        def forward(self, input):
            return torch.nn.functional.leaky_relu(input, 0.02)

    example_args = (torch.randn(10, 10, dtype=torch.float32),)

    verify_model(LeakyReLU0(), example_args)
    verify_model(LeakyReLU1(), example_args)


def test_logsoftmax():
    class LogSoftmax(Module):
        def __init__(self):
            super().__init__()
            self.lsm = torch.nn.LogSoftmax(dim=1)

        def forward(self, input):
            return self.lsm(input)

    class LogSoftmax2(Module):
        def forward(self, input):
            return torch.nn.functional.log_softmax(input, dim=1)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(LogSoftmax(), example_args)
    verify_model(LogSoftmax2(), example_args)


def test_silu():
    class SiLU(Module):
        def __init__(self):
            super().__init__()
            self.silu = torch.nn.SiLU()

        def forward(self, input):
            return self.silu(input)

    class SiLU2(Module):
        def forward(self, input):
            return torch.nn.functional.silu(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(SiLU(), example_args)
    verify_model(SiLU2(), example_args)


def test_softmax():
    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.sm = torch.nn.Softmax(dim=1)

        def forward(self, input):
            return self.sm(input)

    class Softmax2(Module):
        def forward(self, input):
            return torch.nn.functional.softmax(input, dim=1)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Softmax(), example_args)
    verify_model(Softmax2(), example_args)


def test_tril():
    class Tril(Module):
        def forward(self, input):
            return torch.tril(input, 1)

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Tril(), example_args)


def test_triu():
    class Triu(Module):
        def forward(self, input):
            return torch.triu(input, 1)

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Triu(), example_args)


def test_binary():
    example_args1 = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(2, 3, dtype=torch.float32),
    )
    example_args2 = (torch.randn(2, 3, dtype=torch.float32),)

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    verify_model(Add1(), example_args1)
    verify_model(Add2(), example_args2)

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    verify_model(Sub1(), example_args1)
    verify_model(Sub2(), example_args2)

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    verify_model(Mul1(), example_args1)
    verify_model(Mul2(), example_args2)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    verify_model(TrueDiv1(), example_args1)
    verify_model(TrueDiv2(), example_args2)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    verify_model(FloorDiv1(), example_args1)
    verify_model(FloorDiv2(), example_args2)

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    verify_model(Power1(), example_args1)
    verify_model(Power2(), example_args2)

    # LT
    class EQ1(Module):
        def forward(self, lhs, rhs):
            return lhs < rhs

    class EQ2(Module):
        def forward(self, lhs):
            return lhs < 1.0

    verify_model(EQ1(), example_args1)
    verify_model(EQ2(), example_args2)

    # EQ
    class EQ1(Module):
        def forward(self, lhs, rhs):
            return lhs == rhs

    class EQ2(Module):
        def forward(self, lhs):
            return lhs == 1.0

    verify_model(EQ1(), example_args1)
    verify_model(EQ2(), example_args2)

    # Max
    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    verify_model(Max(), example_args1)


def test_native_batch_norm_legit_no_training():
    class BatchNormNoTraining(Module):
        def __init__(self):
            super().__init__()
            self.running_mean = torch.Tensor([0.0, 0.0, 0.0])
            self.running_var = torch.Tensor([1.0, 1.0, 1.0])
            self.weight = torch.Tensor([1.0, 1.0, 1.0])
            self.bias = torch.Tensor([0.0, 0.0, 0.0])
            self.momentum = 0.1
            self.eps = 1e-05

        def forward(self, x):
            return torch.nn.functional.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)
    verify_model(BatchNormNoTraining().eval(), example_args)


def test_batchnorm2d():
    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, input):
            return self.bn(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(BatchNorm2d().eval(), example_args)


def test_adaptive_avgpool2d():
    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, input):
            return self.pool(input)

    class AdaptiveAvgPool2d1(Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool2d(input, [10, 10])

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(AdaptiveAvgPool2d0(), example_args)
    verify_model(AdaptiveAvgPool2d1(), example_args)


def test_addmm():
    class Addmm1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3)

    class Addmm2(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3, beta=0.8, alpha=0.5)

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )

    verify_model(Addmm1(), example_args)
    verify_model(Addmm2(), example_args)


def test_avgpool2d():
    class AvgPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    class AvgPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True)

        def forward(self, input):
            return self.pool(input)

    class AvgPool2d3(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool2d(
                input, kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True
            )

    class AvgPool2d4(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool2d(input, kernel_size=[2, 1], divisor_override=2)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(AvgPool2d(), example_args)

    # when ceil_mode=True, Tensor-likes are not close!
    # verify_model(AvgPool2d2(), example_args)

    # when ceil_mode=True, Tensor-likes are not close!
    # verify_model(AvgPool2d3(), example_args)
    verify_model(AvgPool2d4(), example_args)


def test_baddbmm():
    class BAddBMM1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    class BAddBMM2(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    example_args = (
        torch.randn(4, 128, 512, dtype=torch.float32),
        torch.randn(4, 128, 256, dtype=torch.float32),
        torch.randn(4, 256, 512, dtype=torch.float32),
    )

    verify_model(BAddBMM1(), example_args)
    verify_model(BAddBMM2(), example_args)


def test_bmm():
    class BMM(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.bmm(x, y)

    example_args = (
        torch.randn(4, 128, 256, dtype=torch.float32),
        torch.randn(4, 256, 512, dtype=torch.float32),
    )

    verify_model(BMM(), example_args)


@pytest.mark.skip(
    "TVMError: CodeGenVM cannot handle this intrinsic now: Op(relax.nn.conv1d_transpose)"
)
def test_conv1d_transpose():
    class ConvTranspose1d1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(6, 6, 3, bias=True)

        def forward(self, input):
            return self.conv(input)

    class ConvTranspose1d1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 6, 3])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv_transpose1d(input, self.weight, self.bias)

    class ConvTranspose1d2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(6, 6, 3, bias=False)

        def forward(self, input):
            return self.conv(input)

    example_args = (torch.randn(1, 6, 4, dtype=torch.float32),)

    verify_model(ConvTranspose1d1(), example_args)
    verify_model(ConvTranspose1d1Func(), example_args)
    verify_model(ConvTranspose1d2(), example_args)


@pytest.mark.skip(
    "TVMError: CodeGenVM cannot handle this intrinsic now: Op(relax.nn.conv2d_transpose)"
)
def test_conv2d_transpose():
    class ConvTranspose2d1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(3, 3, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class ConvTranspose2d1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[3, 3, 7, 7])
            self.bias = torch.randn(size=[3])

        def forward(self, input):
            return torch.nn.functional.conv_transpose2d(input, self.weight, self.bias)

    class ConvTranspose2d2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(3, 3, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(ConvTranspose2d1(), example_args)
    verify_model(ConvTranspose2d1Func(), example_args)
    verify_model(ConvTranspose2d2(), example_args)


def test_conv1d():
    class Conv1D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv1D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv1d(input, self.weight, self.bias)

    class Conv1D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)

    verify_model(Conv1D1(), example_args)
    verify_model(Conv1D1Func(), example_args)
    verify_model(Conv1D2(), example_args)


def test_conv2d():
    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv2D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv2d(input, self.weight, self.bias)

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Conv2D1(), example_args)
    verify_model(Conv2D1Func(), example_args)
    verify_model(Conv2D2(), example_args)


def test_conv3d():
    class Conv3D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv3D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7, 7, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv3d(input, self.weight, self.bias)

    class Conv3D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    example_args = (torch.randn(1, 3, 10, 10, 10, dtype=torch.float32),)

    verify_model(Conv3D1(), example_args)
    verify_model(Conv3D1Func(), example_args)
    verify_model(Conv3D2(), example_args)


def test_einsum():
    class Einsum1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.einsum("ii", x)

    class Einsum2(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.einsum("i,j->ij", x, y)

    example_args = (torch.randn(4, 4, dtype=torch.float32),)
    verify_model(Einsum1(), example_args)

    example_args = (torch.randn(5, dtype=torch.float32), torch.randn(4, dtype=torch.float32))
    verify_model(Einsum2(), example_args)


@pytest.mark.skip("IndexError: index out of range in self")
def test_embedding():
    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, input):
            return self.embedding(input)

    example_args = (torch.randint(low=-int(1e5), high=int(1e5), size=(4,), dtype=torch.int64),)

    verify_model(Embedding(), example_args)


def test_groupnorm():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.GroupNorm(3, 3)

        def forward(self, input):
            return self.gn(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(GroupNorm(), example_args)


def test_layernorm():
    class LayerNorm1(Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm((10, 10))

        def forward(self, input):
            return self.ln(input)

    class LayerNorm2(Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(shape))
            self.bias = torch.nn.Parameter(torch.zeros(shape))

        def forward(self, input):
            return torch.nn.functional.layer_norm(
                input, self.weight.shape, self.weight, self.bias, 1e-5
            )

    class LayerNorm3(Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.weight = None
            self.bias = None

        def forward(self, input):
            return torch.nn.functional.layer_norm(input, self.shape, self.weight, self.bias, 1e-5)

    class LayerNorm4(Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.weight = torch.nn.Parameter(torch.ones(shape))
            self.bias = torch.nn.Parameter(torch.zeros(shape))

        def forward(self, input):
            return torch.nn.functional.layer_norm(input, self.shape, self.weight, self.bias, 1e-5)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(LayerNorm1(), example_args)
    verify_model(LayerNorm2((10, 10)), example_args)
    verify_model(LayerNorm3((10, 10)), example_args)
    verify_model(LayerNorm4([10, 10]), example_args)


def test_linear():
    # nn.Linear
    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, input):
            return self.linear(input)

    class Dense1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[7, 10])
            self.bias = torch.randn(size=[7])

        def forward(self, input):
            return torch.nn.functional.linear(input, self.weight, self.bias)

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, input):
            return self.linear(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Dense1(), example_args)
    verify_model(Dense1Func(), example_args)
    verify_model(Dense2(), example_args)

    # matmul
    class MatMul1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )

    verify_model(MatMul1(), example_args)


def test_maxpool2d():
    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    class MaxPool2d_functional(Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nn.functional.max_pool2d(input, kernel_size=[1, 1])

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3])

        def forward(self, input):
            return self.pool(input)

    class MaxPool2d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, input):
            return self.pool(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(MaxPool2d(), example_args)
    verify_model(MaxPool2d_functional(), example_args)
    verify_model(MaxPool2d2(), example_args)
    verify_model(MaxPool2d3(), example_args)


def test_sdpa():
    class SDPA1(Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    class SDPA2(Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)

    verify_model(
        SDPA1(),
        (
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
        ),
    )

    verify_model(
        SDPA2(),
        (
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 128, dtype=torch.float32),
        ),
    )


def test_unbind():
    class Unbind1(Module):
        def forward(self, data):
            return torch.unbind(data)

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    example_args = (torch.randn(3, 3, 10, 10, dtype=torch.float32),)
    verify_model(Unbind1(), example_args)
    verify_model(Unbind2(), example_args)


def test_interpolate():
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    class Interpolate(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (5, 5))

    verify_model(Interpolate(), example_args)

    class Interpolate2(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(
                input,
                size=None,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )

    verify_model(Interpolate2(), example_args)

    class Interpolate3(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(
                input,
                size=None,
                scale_factor=(2.0, 1.0),
                mode="bilinear",
                align_corners=False,
            )

    verify_model(Interpolate3(), example_args)


def test_mean():
    class Mean(Module):
        def forward(self, input):
            return input.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, input: torch.Tensor):
            return input.mean(-1, keepdim=True)

    example_args = (torch.randn(256, 256, dtype=torch.float32),)
    verify_model(Mean(), example_args)
    verify_model(MeanKeepDim(), example_args)


def test_sum():
    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Sum(), example_args)


def test_argmax_argmin():
    class Argmax1(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmax(input, dim=-1)

    class Argmax2(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmax(input, dim=-1, keepdim=True)

    class Argmin1(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmin(input)

    class Argmin2(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmin(input, keepdim=True)

    example_args = (torch.randn(256, 256, dtype=torch.float32),)

    verify_model(Argmax1(), example_args)
    verify_model(Argmax2(), example_args)
    verify_model(Argmin1(), example_args)
    verify_model(Argmin2(), example_args)


def test_cat():
    class Cat0(Module):
        def forward(self, x, y):
            return torch.cat((x, y))

    class Cat1(Module):
        def forward(self, x, y):
            return torch.cat((x, y), dim=1)

    class Cat2(Module):
        def forward(self, x, y):
            return torch.cat((x, y), 1)

    class Cat3(Module):
        def forward(self, x, y):
            return torch.concat((x, y), dim=0)

    example_args = (torch.randn(2, 3, dtype=torch.float32), torch.randn(2, 3, dtype=torch.float32))
    verify_model(Cat0(), example_args)
    verify_model(Cat1(), example_args)
    verify_model(Cat2(), example_args)
    verify_model(Cat3(), example_args)


def test_cumsum():
    class Cumsum(Module):
        def forward(self, input):
            return torch.cumsum(input, dim=1, dtype=torch.int32)

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Cumsum(), example_args)


def test_expand():
    class Expand1(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    class Expand2(Module):
        def forward(self, x):
            return x.expand(4, -1, -1, 4)

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Expand1(), example_args)
    verify_model(Expand2(), example_args)


def test_flatten():
    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, input):
            return self.f(input)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Flatten(), example_args)


def test_permute():
    class Permute1(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    class Permute2(Module):
        def forward(self, x):
            return torch.permute(x, (0, 3, 2, 1))

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Permute1(), example_args)
    verify_model(Permute2(), example_args)


def test_repeat():
    class Tile1(Module):
        def forward(self, x: torch.Tensor):
            return x.repeat(2)

    class Tile2(Module):
        def forward(self, x: torch.Tensor):
            return x.repeat(4, 2)

    example_args = (torch.randn(3, dtype=torch.float32),)
    verify_model(Tile1(), example_args)

    example_args = (torch.randn(1, 3, dtype=torch.float32),)
    verify_model(Tile2(), example_args)

    example_args = (torch.randn(1, 3, dtype=torch.float32),)
    verify_model(Tile2(), example_args)


def test_reshape():
    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Reshape(), example_args)


def test_slice():
    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Slice1(), example_args)

    example_args = (torch.randn(8, 16, dtype=torch.float32),)
    verify_model(Slice2(), example_args)


def test_chunk():
    class Chunk(Module):
        def forward(self, input):
            return torch.chunk(input, 3, dim=1)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Chunk(), example_args)


def test_squeeze():
    class Squeeze1(Module):
        def forward(self, input):
            return input.squeeze(1)

    class Squeeze2(Module):
        def forward(self, input):
            return input.squeeze()

    example_args = (torch.randn(3, 1, 4, 1, dtype=torch.float32),)

    verify_model(Squeeze1(), example_args)
    verify_model(Squeeze2(), example_args)


def test_tile():
    class Tile1(Module):
        def forward(self, x):
            return x.tile((2,))

    class Tile2(Module):
        def forward(self, x):
            return x.tile(4, 2)

    class Tile3(Module):
        def forward(self, x):
            return torch.tile(x, (4, 2))

    example_args = (torch.randn(1, 3, dtype=torch.float32),)
    verify_model(Tile1(), example_args)
    verify_model(Tile2(), example_args)
    verify_model(Tile3(), example_args)


def test_transpose():
    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Transpose(), example_args)


def test_unsqueeze():
    class Unsqueeze1(Module):
        def forward(self, input):
            return input.unsqueeze(1)

    class Unsqueeze2(Module):
        def forward(self, input):
            return input.unsqueeze(-1)

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Unsqueeze1(), example_args)
    verify_model(Unsqueeze2(), example_args)


def test_view():
    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(View(), example_args)


def test_arange():
    class Arange(Module):
        def forward(self, input):
            return torch.arange(0, 20, dtype=torch.int32)

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Arange(), example_args)


def test_fill():
    class Fill(Module):
        def forward(self, input: torch.Tensor):
            return torch.fill(input, 1.5)

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Fill(), example_args)


def test_new_ones():
    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    example_args = (torch.randn(1, 2, 3, dtype=torch.float32),)
    verify_model(NewOnes(), example_args)


def verify_torchvision_model(model_name):
    from torchvision.models import get_model, get_model_weights
    from torchvision.io import read_image

    image_tensor = read_image("tests/python/relax/dog1.jpg")
    model = get_model(model_name, weights="DEFAULT").eval()
    weights = get_model_weights(model_name).DEFAULT
    transforms = weights.transforms()

    batch = transforms(image_tensor).unsqueeze(0)
    example_args = (batch,)
    verify_model(model, example_args)


def test_e2e_alexnet():
    verify_torchvision_model("alexnet")


def test_e2e_convnext_tiny():
    verify_torchvision_model("convnext_tiny")


def test_e2e_densenet121():
    verify_torchvision_model("densenet121")


def test_e2e_efficientnet_b0():
    verify_torchvision_model("efficientnet_b0")


def test_e2e_efficientnet_v2_s():
    verify_torchvision_model("efficientnet_v2_s")


@pytest.mark.skip("AssertionError: Tensor-likes are not close!")
def test_e2e_inception_v3():
    verify_torchvision_model("inception_v3")


@pytest.mark.skip("AssertionError: Unsupported function type _unsafe_view.default")
def test_e2e_maxvit_t():
    verify_torchvision_model("maxvit_t")


def test_e2e_mnasnet0_5():
    verify_torchvision_model("mnasnet0_5")


def test_e2e_mobilenet_v2():
    verify_torchvision_model("mobilenet_v2")


def test_e2e_mobilenet_v3_small():
    verify_torchvision_model("mobilenet_v3_small")


def test_e2e_regnet_x_400mf():
    verify_torchvision_model("regnet_x_400mf")


def test_e2e_resnet18():
    verify_torchvision_model("resnet18")


def test_e2e_resnext50_32x4d():
    verify_torchvision_model("resnext50_32x4d")


def test_e2e_shufflenet_v2_x0_5():
    verify_torchvision_model("shufflenet_v2_x0_5")


def test_e2e_squeezenet1_0():
    verify_torchvision_model("squeezenet1_0")


@pytest.mark.skip("AssertionError: Unsupported function type index.Tensor")
def test_e2e_swin_t():
    verify_torchvision_model("swin_t")


@pytest.mark.skip("AssertionError: Unsupported function type index.Tensor")
def test_e2e_swin_v2_t():
    verify_torchvision_model("swin_v2_t")


def test_e2e_vgg11():
    verify_torchvision_model("vgg11")


def test_e2e_vgg11_bn():
    verify_torchvision_model("vgg11_bn")


def test_e2e_vit_b_32():
    verify_torchvision_model("vit_b_32")


def test_e2e_wide_resnet50_2():
    verify_torchvision_model("wide_resnet50_2")


if __name__ == "__main__":
    tvm.testing.main()
