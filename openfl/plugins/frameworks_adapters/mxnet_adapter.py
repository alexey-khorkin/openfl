# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MXNet Framework Adapter plugin."""


# import numpy as np
import mxnet as mx
from mxnet import nd

from .framework_adapter_interface import FrameworkAdapterPluginInterface

class FrameworkAdapterPlugin(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    def __init__(self) -> None:
        """Initialize framework adapter."""
        pass

    # @staticmethod
    # def serialization_setup():
    #     """Prepare model for serialization (optional)."""
    #     from mxnet import gluon, nd
    #     from mxnet.gluon import Block, HybridBlock

    #     def unpack(prefix_path):
    #         path_to_json = prefix_path + '-symbol.json'
    #         path_to_params = prefix_path + '-0000.params'
    #         deserialized_net = gluon.nn.SymbolBlock.imports(path_to_json, ['data'], path_to_params)
    #         return deserialized_net

    #     def make_mx_hybrid_blocks_picklable():

    #         def __reduce__(self):
    #             print(self)
    #             path_to_file = "net_serialized"
    #             self(nd.ones((1, 1, 96, 96))) # forward pass
    #             self.export(path_to_file)
    #             return (unpack, (path_to_file,))

    #         cls = Block
    #         cls.__reduce__ = __reduce__
        
    #     # apply changes
    #     make_mx_hybrid_blocks_picklable()


    @staticmethod
    def get_tensor_dict(model, optimizer=None) -> dict:
        """
        Extract tensor dict from a model and an optimizer.

        Returns:
        dict {weight name: numpy ndarray}
        """
        state, model_params = {}, model.collect_params()
        for param_name, param_tensor in model_params.items():
            if isinstance(param_tensor.data(), mx.ndarray.ndarray.NDArray):
                state[param_name] = param_tensor.data().asnumpy()

        if optimizer is not None:
            opt_state = _get_optimizer_state(optimizer)
            state = {**state, **opt_state}

        return state

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device=mx.cpu()):
        """
        Set tensor dict from a model and an optimizer.

        Given a dict {weight name: numpy ndarray} sets weights to
        the model and optimizer objects inplace.
        """
        device = mx.cpu()

        model_params = model.collect_params()
        for param_name in model_params:
            model_params[param_name].set_data(nd.array(tensor_dict.pop(param_name), ctx=device))
        

def _get_optimizer_state(optimizer):
    """Return the optimizer state.

    Args:
        optimizer
    """
    opt_state = {}
    return opt_state
