# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dill serializer plugin."""

import dill
import cloudpickle

from .serializer_interface import Serializer


class CommonSerializer(Serializer):
    """Serializer API plugin."""

    def __init__(self) -> None:
        """Initialize serializer."""
        super().__init__()

    @staticmethod
    def serialize(object_, filename):
        """Serialize an object and save to disk."""
        try:
            with open(filename, 'wb') as f:
                cloudpickle.dump(object_, f)
        except Exception as exc:
            with open(filename, 'wb') as f:
                dill.dump(object_, f, recurse=True)


    @staticmethod
    def restore_object(filename):
        """Load and deserialize an object."""
        try:
            with open(filename, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as exc:
            with open(filename, 'rb') as f:
                return dill.load(f)
