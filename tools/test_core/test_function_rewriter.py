# Copyright (c) OpenMMLab. All rights reserved.
import torch

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close

from mmdeploy.core import FUNCTION_REWRITER, RewriterContext
from mmdeploy.core.rewriters.function_rewriter import FunctionRewriter
from mmdeploy.core.rewriters.rewriter_utils import collect_env
from mmdeploy.utils.constants import IR, Backend


def test_function_rewriter():
    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([2, 4, 6, 8, 10])

    @FUNCTION_REWRITER.register_rewriter(func_name="torch.mul", backend="tensorrt")
    @FUNCTION_REWRITER.register_rewriter(func_name="torch.add", backend="tensorrt")
    def sub_func(x, y):
        ctx = FUNCTION_REWRITER.get_context("torch.add")
        assert hasattr(ctx, "cfg")
        assert hasattr(ctx, "origin_func")
        return x - y

    cfg = dict()
    with RewriterContext(cfg, backend="tensorrt"):
        result = torch.add(x, y)
        # replace add with sub
        torch_assert_close(result, x - y)
        result = torch.mul(x, y)
        # replace add with sub
        torch_assert_close(result, x - y)

    result = torch.add(x, y)
    # recovery origin function
    torch_assert_close(result, x + y)

    with RewriterContext(cfg):
        result = torch.add(x, y)
        # replace should not happen with wrong backend
        torch_assert_close(result, x + y)

    # test different config
    @FUNCTION_REWRITER.register_rewriter(func_name="torch.Tensor.add", backend="default")
    def mul_func_class(x, y):
        return x * y

    with RewriterContext(cfg, backend="tensorrt"):
        result = x.add(y)
        # replace add with multi
        torch_assert_close(result, x * y)

    result = x.add(y)
    # recovery origin function
    torch_assert_close(result, x + y)

    with RewriterContext(cfg):
        result = x.add(y)
        # replace add with multi
        torch_assert_close(result, x * y)

    # test origin_func
    @FUNCTION_REWRITER.register_rewriter(func_name="torch.add", backend="default")
    def origin_add_func(x, y, **kwargs):
        ctx = FUNCTION_REWRITER.get_context("torch.add")
        return ctx.origin_func(x, y, **kwargs) + 1

    with RewriterContext(cfg):
        result = torch.add(x, y)
        # replace with origin + 1
        torch_assert_close(result, x + y + 1)

    # remove torch.add
    del FUNCTION_REWRITER._origin_functions[-1]
    torch_assert_close(torch.add(x, y), x + y)

    FUNCTION_REWRITER._registry.remove_record(sub_func)
    FUNCTION_REWRITER._registry.remove_record(mul_func_class)
    FUNCTION_REWRITER._registry.remove_record(origin_add_func)


def test_rewrite_empty_function():
    function_rewriter = FunctionRewriter()

    @function_rewriter.register_rewriter(func_name="torch.abcdefghijklmn")
    def func(x, y):
        return x + y

    function_rewriter.enter()
    assert len(function_rewriter._origin_functions) == 0
    function_rewriter.exit()


class TestHomonymicRewriter:
    def test_rewrite_homonymic_methods(self):
        import package

        path1 = "package.C.method"
        path2 = "package.module.C.method"

        c = package.C()

        function_rewriter = FunctionRewriter()

        assert c.method() == 1

        @function_rewriter.register_rewriter(func_name=path1)
        def func_2(self):
            return 2

        @function_rewriter.register_rewriter(func_name=path2, backend=Backend.NCNN.value)
        def func_3(self):
            return 3

        function_rewriter.enter(env=collect_env(Backend.NCNN, ir=IR.DEFAULT))
        assert c.method() == 3
        function_rewriter.exit()

        assert c.method() == 1

        function_rewriter2 = FunctionRewriter()

        @function_rewriter2.register_rewriter(func_name=path1, backend=Backend.NCNN.value)
        def func_4(self):
            return 4

        @function_rewriter2.register_rewriter(func_name=path2)
        def func_5(self):
            return 5

        function_rewriter2.enter(env=collect_env(Backend.NCNN, ir=IR.DEFAULT))
        assert c.method() == 4
        function_rewriter2.exit()

        assert c.method() == 1


def test_rewrite_derived_methods():
    import package

    path1 = "package.C.method"
    path2 = "package.C2.method"

    base_obj = package.C()
    derived_obj = package.C2()

    assert base_obj.method() == 1
    assert derived_obj.method() == 1

    function_rewriter = FunctionRewriter()

    @function_rewriter.register_rewriter(func_name=path1)
    def func_2(self):
        return 2

    @function_rewriter.register_rewriter(func_name=path2, backend=Backend.NCNN.value)
    def func_3(self):
        return 3

    function_rewriter.enter(env=collect_env(Backend.DEFAULT, ir=IR.DEFAULT))
    assert base_obj.method() == 2
    assert derived_obj.method() == 2
    function_rewriter.exit()

    function_rewriter.enter(env=collect_env(Backend.NCNN, ir=IR.DEFAULT))
    assert base_obj.method() == 2
    assert derived_obj.method() == 3
    function_rewriter.exit()

    assert base_obj.method() == 1
    assert derived_obj.method() == 1

    # Check if the recovery is correct
    function_rewriter.enter(env=collect_env(Backend.DEFAULT, ir=IR.DEFAULT))
    assert base_obj.method() == 2
    assert derived_obj.method() == 2
    function_rewriter.exit()

    assert base_obj.method() == 1
    assert derived_obj.method() == 1
