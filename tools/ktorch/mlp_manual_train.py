#!/usr/bin/env python3
import math

import ktorch
from ktorch import ops


def make_tensor(shape, values):
    t = ktorch.Tensor(shape, dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
    t.copy_from_list_float(values)
    return t


def mse_loss_and_grad(pred_vals, target_vals):
    if len(pred_vals) != len(target_vals):
        raise ValueError("pred/target length mismatch")
    n = len(pred_vals)
    loss = 0.0
    grad = [0.0] * n
    for i in range(n):
        diff = pred_vals[i] - target_vals[i]
        loss += diff * diff
        grad[i] = (2.0 / n) * diff
    return loss / n, grad


def sgd_update(param, grad, lr):
    p = param.to_list_float()
    g = grad.to_list_float()
    updated = [pv - lr * gv for pv, gv in zip(p, g)]
    param.copy_from_list_float(updated)


def main():
    # Tiny synthetic task: 2 samples, input=3, hidden=4, output=2
    x = make_tensor([2, 3], [
        1.0, -1.0, 0.5,
        0.0, 2.0, -1.5,
    ])
    target = [
        0.5, -1.0,
        1.0, 0.0,
    ]

    w1 = make_tensor([4, 3], [
        0.10, -0.20, 0.05,
        -0.30, 0.25, 0.15,
        0.40, -0.10, -0.05,
        0.20, 0.30, -0.25,
    ])
    b1 = make_tensor([4], [0.0, 0.0, 0.0, 0.0])

    w2 = make_tensor([2, 4], [
        0.20, -0.10, 0.15, 0.05,
        -0.05, 0.25, -0.20, 0.10,
    ])
    b2 = make_tensor([2], [0.0, 0.0])

    lr = 0.1
    steps = 20
    losses = []

    for step in range(steps):
        z1, ctx1 = ops.linear_forward(x, w1, b1)
        a1, relu_ctx = ops.relu_forward(z1)
        z2, ctx2 = ops.linear_forward(a1, w2, b2)

        pred = z2.to_list_float()
        loss, dz2_vals = mse_loss_and_grad(pred, target)
        losses.append(loss)

        dz2 = make_tensor([2, 2], dz2_vals)

        g2 = ops.linear_backward(dz2, ctx2, True, True, True)
        dz1 = ops.relu_backward(g2.dX, relu_ctx)
        g1 = ops.linear_backward(dz1, ctx1, False, True, True)

        sgd_update(w2, g2.dW, lr)
        sgd_update(b2, g2.db, lr)
        sgd_update(w1, g1.dW, lr)
        sgd_update(b1, g1.db, lr)

        if step in (0, steps - 1):
            print(f"step={step} loss={loss:.8f}")

    if not math.isfinite(losses[-1]):
        raise RuntimeError("Training diverged: final loss is not finite.")

    if losses[-1] > losses[0]:
        raise RuntimeError(f"Loss did not decrease: start={losses[0]:.8f}, end={losses[-1]:.8f}")

    # Tiny inference smoke after training
    y1, _ = ops.linear_forward(x, w1, b1)
    y2, _ = ops.relu_forward(y1)
    y3, _ = ops.linear_forward(y2, w2, b2)
    print("inference_output=", [round(v, 6) for v in y3.to_list_float()])


if __name__ == "__main__":
    main()
