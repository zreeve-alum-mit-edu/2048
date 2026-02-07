"""
Tests for the base Representation interface.

Per DEC-0008: Representation module interface requires:
- __init__(config)
- forward(state) -> Tensor
- output_shape() -> tuple
"""

import pytest
import torch

from representations.base import Representation


class TestRepresentationInterface:
    """Test that Representation is a proper abstract base class."""

    def test_cannot_instantiate_abstract_base(self):
        """Representation base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Representation({})

    def test_subclass_must_implement_forward(self):
        """Subclass missing forward() should fail to instantiate."""
        class IncompleteRep(Representation):
            def output_shape(self):
                return (10,)

        with pytest.raises(TypeError):
            IncompleteRep({})

    def test_subclass_must_implement_output_shape(self):
        """Subclass missing output_shape() should fail to instantiate."""
        class IncompleteRep(Representation):
            def forward(self, state):
                return state

        with pytest.raises(TypeError):
            IncompleteRep({})

    def test_complete_subclass_can_instantiate(self):
        """Complete subclass with all methods can be instantiated."""
        class CompleteRep(Representation):
            def forward(self, state):
                return state.view(state.size(0), -1)

            def output_shape(self):
                return (272,)

        rep = CompleteRep({"test": True})
        assert rep.config == {"test": True}

    def test_representation_is_nn_module(self):
        """Representation should be an nn.Module for gradient support."""
        class CompleteRep(Representation):
            def forward(self, state):
                return state.view(state.size(0), -1)

            def output_shape(self):
                return (272,)

        rep = CompleteRep({})
        assert isinstance(rep, torch.nn.Module)
