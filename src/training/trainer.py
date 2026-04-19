import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any
from torch.utils.data import DataLoader

from src.models.smallnet import SmallNet
from src.buffers.replay_buffer import RingBuffer, ReservoirBuffer
from src.strategies.uncertainty import refresh_scores
from src.utils.seed import set_seed
from src.utils.device import device


@torch.no_grad()
def eval_all_tasks(model: nn.Module, test_loaders: List[DataLoader]) -> List[float]:
    model.eval()
    accs = []

    for loader in test_loaders:
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        accs.append(correct / max(1, total))

    return accs


def run_cl_experiment(
    method: str,
    buffer_size: int,
    seed: int,
    train_tasks,
    test_tasks,
    input_shape,
    num_classes,
    epochs_per_task: int = 1,
    batch_size: int = 128,
    replay_bs: int = 64,
    lr: float = 1e-3,
    p_drop: float = 0.1,
    refresh_every: int = 200,
    refresh_k: int = 256,
    mc: int = 5,
    beta: float = 0.9,
    replay_weight: float = 0.5,
) -> Dict[str, Any]:
    """
    Supported methods:
      - "no_replay"
      - "ring_uniform"
      - "reservoir_uniform"
      - "reservoir_loss"
      - "reservoir_entropy"
      - "reservoir_ig"
    """

    set_seed(seed)

    model = SmallNet(input_shape, num_classes, p_drop=p_drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loaders = [
        DataLoader(task, batch_size=batch_size, shuffle=True)
        for task in train_tasks
    ]
    test_loaders = [
        DataLoader(task, batch_size=512, shuffle=False)
        for task in test_tasks
    ]

    ring = RingBuffer(buffer_size) if method == "ring_uniform" and buffer_size > 0 else None
    resv = ReservoirBuffer(buffer_size) if method != "ring_uniform" and buffer_size > 0 else None

    step = 0
    acc_matrix = []

    for task_id, loader in enumerate(train_loaders):
        for _ in range(epochs_per_task):
            for x, y in loader:
                step += 1
                x_d, y_d = x.to(device), y.to(device)

                model.train()
                logits_cur = model(x_d)
                loss_cur = F.cross_entropy(logits_cur, y_d)

                loss_rep = torch.tensor(0.0, device=device)

                if buffer_size > 0 and method != "no_replay":
                    bx, by = None, None

                    if method == "ring_uniform" and ring is not None and len(ring) > 0:
                        bx, by = ring.sample_uniform(replay_bs)

                    elif method == "reservoir_uniform" and resv is not None and len(resv) > 0:
                        bx, by = resv.sample_uniform(replay_bs)

                    elif method in ["reservoir_loss", "reservoir_entropy", "reservoir_ig"] and resv is not None and len(resv) > 0:
                        bx, by = resv.sample_weighted(replay_bs)

                    elif method not in [
                        "no_replay",
                        "ring_uniform",
                        "reservoir_uniform",
                        "reservoir_loss",
                        "reservoir_entropy",
                        "reservoir_ig",
                    ]:
                        raise ValueError(f"Unknown method: {method}")

                    if bx is not None and by is not None:
                        bx, by = bx.to(device), by.to(device)
                        logits_rep = model(bx)
                        loss_rep = F.cross_entropy(logits_rep, by)

                loss = loss_cur + replay_weight * loss_rep

                opt.zero_grad()
                loss.backward()
                opt.step()

                if buffer_size > 0:
                    for xi, yi in zip(x, y):
                        if ring is not None:
                            ring.add(xi, int(yi))
                        if resv is not None:
                            resv.add(xi, int(yi))

                if (
                    buffer_size > 0
                    and resv is not None
                    and method in ["reservoir_loss", "reservoir_entropy", "reservoir_ig"]
                    and step % refresh_every == 0
                ):
                    score_mode = {
                        "reservoir_loss": "loss",
                        "reservoir_entropy": "entropy",
                        "reservoir_ig": "ig",
                    }[method]

                    refresh_scores(
                        model=model,
                        buf=resv,
                        score_mode=score_mode,
                        refresh_k=refresh_k,
                        mc=mc,
                        beta=beta,
                    )

        accs = eval_all_tasks(model, test_loaders)
        acc_matrix.append(accs)

    final_avg_acc = sum(acc_matrix[-1]) / len(acc_matrix[-1])

    forgetting = 0.0
    for k in range(len(test_tasks)):
        best_acc = max(acc_matrix[t][k] for t in range(len(acc_matrix)))
        final_acc = acc_matrix[-1][k]
        forgetting += (best_acc - final_acc)

    forgetting /= len(test_tasks)

    return {
        "final_avg_acc": final_avg_acc,
        "avg_forgetting": forgetting,
        "acc_matrix": acc_matrix,
    }
