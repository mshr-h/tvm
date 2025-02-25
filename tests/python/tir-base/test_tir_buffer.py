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

import tvm
import tvm.testing
from tvm import te
from tvm.tir import Buffer
from tvm.script import tir as T

import numpy as np
import pytest


def test_buffer():
    m = te.size_var("m")
    n = te.size_var("n")
    l = te.size_var("l")
    Ab = tvm.tir.decl_buffer((m, n), "float32")
    Bb = tvm.tir.decl_buffer((n, l), "float32")

    assert isinstance(Ab, tvm.tir.Buffer)
    assert Ab.dtype == "float32"
    assert tuple(Ab.shape) == (m, n)


def test_buffer_access_ptr():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32", strides=[n + 1, 1])
    aptr = Ab.access_ptr("rw")
    tvm.ir.assert_structural_equal(aptr.args[3], Ab.strides[0] * m)
    assert aptr.args[0].dtype == Ab.dtype
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    aptr = Ab.access_ptr("w")
    assert aptr.args[4].value == Buffer.WRITE


def test_buffer_access_ptr_offset():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32")
    aptr = Ab.access_ptr("rw", offset=100)
    tvm.testing.assert_prim_expr_equal(aptr.args[2], 100)
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    v = te.size_var("int32")
    aptr = Ab.access_ptr("rw", offset=100 + 100 + v)
    tvm.testing.assert_prim_expr_equal(aptr.args[2], 200 + v)
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    aptr = Ab.access_ptr("rw", offset=tvm.tir.call_extern("int32", "test_call", 100 + 100 + v))
    tvm.testing.assert_prim_expr_equal(
        aptr.args[2], tvm.tir.call_extern("int32", "test_call", 200 + v)
    )
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE


def test_buffer_access_ptr_extent():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32")
    aptr = Ab.access_ptr("rw")
    tvm.ir.assert_structural_equal(aptr.args[3], m * n)
    aptr = Ab.access_ptr("rw", offset=100)
    tvm.ir.assert_structural_equal(aptr.args[3], m * n - 100)
    Ab = tvm.tir.decl_buffer((m, n), "float32", strides=[n + 1, 1])
    aptr = Ab.access_ptr("rw", offset=100)
    tvm.ir.assert_structural_equal(aptr.args[3], Ab.strides[0] * m - 100)

    # Test extent from input params
    aptr = Ab.access_ptr("rw", extent=200)
    tvm.ir.assert_structural_equal(aptr.args[3], T.int32(200))
    aptr = Ab.access_ptr("rw", offset=100, extent=100)
    tvm.ir.assert_structural_equal(aptr.args[3], T.int32(100))


def test_buffer_vload():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32", elem_offset=100)
    load = Ab.vload([2, 3])
    tvm.ir.assert_structural_equal(load.indices, [T.int32(2), T.int32(3)])


def test_buffer_offset_of():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32", elem_offset=100)
    offset = Ab.offset_of([2, 3])
    tvm.ir.assert_structural_equal(offset, [n * 2 + 103])


def test_buffer_index_merge_mult_mod():
    m = te.size_var("m")
    n = te.size_var("n")
    s = te.size_var("s")
    k0 = te.size_var("k0")
    k1 = te.size_var("k1")
    A = tvm.tir.decl_buffer((m, n), "float32")
    A_stride = tvm.tir.decl_buffer((m, n), "float32", strides=(s, 1))

    def assert_simplified_equal(index_simplified, index_direct):
        (
            tvm.ir.assert_structural_equal(index_simplified, index_direct),
            "index_simplified=%s, index_direct=%s" % (index_simplified, index_direct),
        )

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    # Test Case1
    index_simplified = A_stride.offset_of(
        (idxd(idxm(k0, k1), s), idxm(idxm(k0, k1), s) + idxd(k0, k1) * k1)
    )
    index_direct = A_stride.offset_of((0, k0))
    assert_simplified_equal(index_simplified, index_direct)

    # Test Case2
    index_simplified = A.offset_of(
        (idxd(idxm(k0, idxd(k1, s)), n), idxm(idxm(k0, idxd(k1, s)), n) + idxm(k0, k1))
    )
    index_direct = A.offset_of((0, idxm(k0, idxd(k1, s)) + idxm(k0, k1)))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case3
    index_simplified = A.offset_of(
        (
            idxd((idxd(k0, idxd(k1, s)) * idxd(k1, s)), n) + idxd(idxm(k0, idxd(k1, s)), n),
            idxm((idxd(k0, idxd(k1, s)) * idxd(k1, s)), n) + idxm(idxm(k0, idxd(k1, s)), n),
        )
    )
    index_direct = A.offset_of((0, k0))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case4 (not able to simplify)
    index_simplified = A.offset_of(
        (idxd(idxm(k0, idxd(k1, s)), n), idxm(idxm(k0, idxd(k1, n)), n) + idxm(k0, k1))
    )
    index_direct = A.offset_of(
        (0, idxd(idxm(k0, idxd(k1, s)), n) * n + (idxm(idxm(k0, idxd(k1, n)), n) + idxm(k0, k1)))
    )
    assert_simplified_equal(index_simplified, index_direct)

    # Test Case5
    B = tvm.tir.decl_buffer((1, 14, 14, 1024))
    i = te.size_var("i")
    j = te.size_var("j")
    k = te.size_var("k")

    index_simplified1 = B.offset_of(
        (
            idxd(idxd(idxd((i * 50176 + j * 28672 + k), 1024), 14), 14),
            idxm(idxd(idxd((i * 50176 + j * 28672 + k), 1024), 14), 14),
            idxm(idxd((i * 50176 + j * 28672 + k), 1024), 14),
            idxm((i * 50176 + j * 28672 + k), 1024),
        )
    )
    index_simplified2 = B.offset_of(
        (
            idxd(idxd(i * 49 + j * 28 + idxd(k, 1024), 14), 14),
            idxm(idxd(i * 49 + j * 28 + idxd(k, 1024), 14), 14),
            idxm(i * 7 + idxd(k, 1024), 14),
            idxm(k, 1024),
        )
    )
    index_direct = B.offset_of((0, 0, 0, (i * 50176 + j * 28672 + k)))
    assert_simplified_equal(index_simplified1, index_direct)
    assert_simplified_equal(index_simplified2, index_direct)


def test_buffer_flatten():
    """A buffer should flatten to a 1-d shape"""
    buf = tvm.tir.decl_buffer([16, 32])
    flat = buf.get_flattened_buffer()
    assert buf.data.same_as(flat.data)
    tvm.ir.assert_structural_equal(flat.shape, [T.int32(16 * 32)])


def test_buffer_flatten_preserves_identity():
    """Flattening a 1-d buffer should return the original"""
    buf = tvm.tir.decl_buffer([16])
    flat = buf.get_flattened_buffer()
    assert buf.same_as(flat)


def test_buffer_flatten_uses_axis_separators():
    """Flattening to N-d physical buffers uses the axis separators"""
    buf = tvm.tir.decl_buffer([4, 16, 32], axis_separators=[2])
    flat = buf.get_flattened_buffer()
    tvm.ir.assert_structural_equal(flat.axis_separators, [T.int32(1)])
    tvm.ir.assert_structural_equal(flat.shape, [T.int32(4 * 16), T.int32(32)])


def test_invalid_axis_separators_raises_exception():
    with pytest.raises(ValueError):
        tvm.tir.decl_buffer([1], axis_separators=[1, 2])


if __name__ == "__main__":
    tvm.testing.main()
