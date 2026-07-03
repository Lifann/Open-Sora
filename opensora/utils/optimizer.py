import torch
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler


def _build_hybrid_adam(params, **kw):
    """尝试构造 ColossalAI 的 HybridAdam。

    HybridAdam 内部会加载一个 CUDA 扩展（CPUAdamLoader），在无 CUDA 的
    环境（如寒武纪 MLU 590）下会因找不到 CUDA_HOME 而抛 AssertionError。
    此时回退到纯 PyTorch 的 AdamW（Open-Sora 配置里 adamw_mode=True，
    即 HybridAdam 退化为 AdamW 语义，行为一致）。

    注意：params 必须由调用方传 list（而非 generator），否则 HybridAdam
    构造失败时可能已耗尽该迭代器，导致回退的 AdamW 拿到空参数列表而报错。
    """
    try:
        return HybridAdam(params, **kw)
    except Exception as e:  # CUDA_HOME missing / 扩展加载失败
        import logging

        logging.getLogger("opensora").warning(
            "[optimizer] HybridAdam 不可用（%s），回退到 torch.optim.AdamW", e
        )
        # HybridAdam 专属字段对 AdamW 无效，逐字段剥离后安全传递
        kw.pop("adamw_mode", None)
        return AdamW(params, **kw)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    Create an optimizer.

    Args:
        model (torch.nn.Module): The model to be optimized.
        optimizer_config (dict): The configuration of the optimizer.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    optimizer_name = optimizer_config.pop("cls", "HybridAdam")
    # 物化为 list：params 可能要传给两次构造（HybridAdam 失败回退 AdamW），
    # generator 只能消费一次，必须固化成 list。
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if optimizer_name == "HybridAdam":
        optimizer = _build_hybrid_adam(params, **optimizer_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps_per_epoch: int,
    epochs: int = 1000,
    warmup_steps: int | None = None,
    use_cosine_scheduler: bool = False,
    initial_lr: float = 1e-6,
) -> _LRScheduler | None:
    """
    Create a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        num_steps_per_epoch (int): The number of steps per epoch.
        epochs (int): The number of epochs.
        warmup_steps (int |  None): The number of warmup steps.
        use_cosine_scheduler (bool): Whether to use cosine scheduler.

    Returns:
        _LRScheduler |  None: The learning rate scheduler
    """
    if warmup_steps is None and not use_cosine_scheduler:
        lr_scheduler = None
    elif use_cosine_scheduler:
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=num_steps_per_epoch * epochs,
            warmup_steps=warmup_steps,
        )
    else:
        lr_scheduler = LinearWarmupLR(optimizer, initial_lr=1e-6, warmup_steps=warmup_steps)
        # lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)

    return lr_scheduler


class LinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_step (int, optional): The index of last step, defaults to -1. When last_step=-1,
            the schedule is started from the beginning or When last_step=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, initial_lr=0, warmup_steps: int = 0, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr + (self.last_epoch + 1) / (self.warmup_steps + 1) * (lr - self.initial_lr)
                for lr in self.base_lrs
            ]
        else:
            return self.base_lrs
