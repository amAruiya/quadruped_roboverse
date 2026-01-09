"""Setup 回调函数。

在任务初始化时调用，handler 已就绪。
"""

from __future__ import annotations

import torch
from loguru import logger as log


def example_setup(task, **kwargs) -> None:
    """示例 setup 回调。

    Args:
        task: BaseLocomotionTask 实例
        **kwargs: 额外参数
    """
    log.info("Setup callback executed")