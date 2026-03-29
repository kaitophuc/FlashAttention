import importlib

_C = importlib.import_module("ktorch._C")


LinearContext = _C.LinearContext
LinearGrads = _C.LinearGrads
LayerNormContext = _C.LayerNormContext
LayerNormGrads = _C.LayerNormGrads
ReluContext = _C.ReluContext
SoftmaxCrossEntropyContext = _C.SoftmaxCrossEntropyContext


def linear_forward(x, w, b=None):
    return _C.linear_forward(x, w, b)


def linear_backward(dy, ctx, needs_dx=True, needs_dw=True, needs_db=True):
    return _C.linear_backward(dy, ctx, needs_dx, needs_dw, needs_db)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    return _C.layernorm_forward(x, gamma, beta, eps)


def layernorm_backward(dy, ctx, needs_dx=True, needs_dgamma=True, needs_dbeta=True):
    return _C.layernorm_backward(dy, ctx, needs_dx, needs_dgamma, needs_dbeta)


def relu_forward(x):
    return _C.relu_forward(x)


def relu_backward(dy, ctx):
    return _C.relu_backward(dy, ctx)


def softmax_forward(x):
    return _C.softmax_forward(x)


def softmax_backward(dy, y):
    return _C.softmax_backward(dy, y)


def softmax_cross_entropy_forward(logits, labels):
    return _C.softmax_cross_entropy_forward(logits, labels)


def softmax_cross_entropy_backward(ctx):
    return _C.softmax_cross_entropy_backward(ctx)
