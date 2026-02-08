"""
Tuning Utilities.

Provides factory functions for creating representations and other utilities.

Per DEC-0008: Representation interface requires __init__(config), forward(state), output_shape().
"""

from typing import Dict, Any

from representations.base import Representation
from representations.onehot import OneHotRepresentation
from representations.embedding import EmbeddingRepresentation
from representations.cnn import CNNRepresentation


def create_representation(repr_type: str, params: Dict[str, Any]) -> Representation:
    """Create representation instance from type and params.

    Args:
        repr_type: One of "onehot", "embedding", "cnn_2x2", "cnn_4x1", "cnn_multi"
        params: Hyperparameters including representation-specific ones

    Returns:
        Representation instance

    Raises:
        ValueError: If repr_type is unknown
    """
    if repr_type == "onehot":
        return OneHotRepresentation({})

    elif repr_type == "embedding":
        embed_dim = params.get("embed_dim", 32)
        return EmbeddingRepresentation({"embed_dim": embed_dim})

    elif repr_type == "cnn_2x2":
        channels = params.get("cnn_channels", 64)
        return CNNRepresentation({
            "kernels": [
                {"size": [2, 2], "out_channels": channels, "stride": [1, 1]}
            ],
            "combine": "concat",
            "activation": "relu"
        })

    elif repr_type == "cnn_4x1":
        channels = params.get("cnn_channels", 64)
        return CNNRepresentation({
            "kernels": [
                {"size": [4, 1], "out_channels": channels, "stride": [1, 1]},
                {"size": [1, 4], "out_channels": channels, "stride": [1, 1]},
            ],
            "combine": "concat",
            "activation": "relu"
        })

    elif repr_type == "cnn_multi":
        channels = params.get("cnn_channels", 64)
        return CNNRepresentation({
            "kernels": [
                {"size": [2, 2], "out_channels": channels, "stride": [1, 1]},
                {"size": [4, 1], "out_channels": channels, "stride": [1, 1]},
                {"size": [1, 4], "out_channels": channels, "stride": [1, 1]},
            ],
            "combine": "concat",
            "activation": "relu"
        })

    else:
        raise ValueError(f"Unknown representation type: {repr_type}")


def get_representation_output_dim(repr_type: str, params: Dict[str, Any]) -> int:
    """Calculate output dimension for a representation.

    Args:
        repr_type: Representation type
        params: Hyperparameters

    Returns:
        Output dimension (flattened)
    """
    repr_module = create_representation(repr_type, params)
    return repr_module.output_shape()[0]
