import matplotlib.pyplot as plt

import mxnet as mx

CTX = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# print(f"=== MXNet is using {CTX} ===")

MODEL_CHKPNT: str = ".model_checkpoints"
