# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import random
from typing import Any, ClassVar

import numpy as np
import pytorch3d.transforms as pt
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from gr00t.data.schema import DatasetMetadata, RotationType, StateActionMetadata
from gr00t.data.transform.base import InvertibleModalityTransform, ModalityTransform


class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.inverse_funcs)


class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary"]

    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = statistics
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # Normalize the tensor
        if self.mode == "q99":
            # Range of q99 is [-1, 1]
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (
                max[..., mask] - min[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where min == max
            # normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            # Range of scale is [0, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min), torch.abs(max))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean
        elif self.mode == "min_max":
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            return (x + 1) / 2 * (max - min) + min
        elif self.mode == "binary":
            return (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(
        default_factory=dict, description="The input dtypes for each state key."
    )
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each state key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, np.ndarray
            ), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class StateActionTransform(InvertibleModalityTransform):
    """
    Class for state or action transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        normalization_modes (dict[str, str]): The normalization modes for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be normalized.
        target_rotations (dict[str, str]): The target representations for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be rotated.
        pose_keys (list[str]): Keys that contain pose data (position + rotation).
            For these keys, first 3 dims are position, last 4 dims are quaternion rotation.
        position_normalization_modes (dict[str, str]): Normalization modes for position part of pose keys.
        rotation_normalization_modes (dict[str, str]): Normalization modes for rotation part of pose keys.
    """

    # Configurable attributes
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each state key."
    )
    target_rotations: dict[str, str] = Field(
        default_factory=dict, description="The target representations for each state key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each state key."
    )
    pose_keys: list[str] = Field(
        default_factory=list, description="Keys that contain pose data (position + rotation)"
    )
    position_normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="Normalization modes for position part of pose keys"
    )
    rotation_normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="Normalization modes for rotation part of pose keys"
    )
    position_statistics: dict[str, dict] = Field(
        default_factory=dict, description="Statistics for position part of pose keys"
    )
    rotation_statistics: dict[str, dict] = Field(
        default_factory=dict, description="Statistics for rotation part of pose keys"
    )

    # Model variables
    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _position_normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _rotation_normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    # Model constants
    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {
            "min": [-1, -1, -1, -1, -1, -1],
            "max": [1, 1, 1, 1, 1, 1],
        },
        "euler_angles": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
        "quaternion": {
            "min": [-1, -1, -1, -1],
            "max": [1, 1, 1, 1],
        },
        "axis_angle": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
    }

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to", "normalization_modes", "target_rotations", 
                "pose_keys", "position_normalization_modes", "rotation_normalization_modes"
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(
                    config, StateActionMetadata
                ), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert (
                        "min" in normalization_statistics and "max" in normalization_statistics
                    ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["min"]) == len(
                        normalization_statistics["max"]
                    ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                elif normalization_mode == "mean_std":
                    assert (
                        "mean" in normalization_statistics and "std" in normalization_statistics
                    ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["mean"]) == len(
                        normalization_statistics["std"]
                    ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                elif normalization_mode == "q99":
                    assert (
                        "q01" in normalization_statistics and "q99" in normalization_statistics
                    ), f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["q01"]) == len(
                        normalization_statistics["q99"]
                    ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                elif normalization_mode == "binary":
                    assert (
                        len(normalization_statistics) == 1
                    ), f"Binary normalization should only have one value, but got {normalization_statistics}"
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        # Check that all state keys specified in apply_to have their modality_metadata
        for key in self.apply_to:
            split_key = key.split(".")
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(
                    modality_metadata, modality
                ), f"{state_key} config not found"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        # Check that all state keys specified in normalization_modes have their statistics in state_statistics
        for key in self.normalization_modes:
            split_key = key.split(".")
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(
                dataset_statistics, modality
            ), f"{state_key} statistics not found"
            assert (
                len(getattr(modality_metadata, modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[
                state_key
            ].model_dump()

        # Handle pose-specific statistics (split position and rotation)
        for key in self.pose_keys:
            split_key = key.split(".")
            modality, state_key = split_key
            
            if hasattr(dataset_statistics, modality) and state_key in getattr(dataset_statistics, modality):
                full_stats = getattr(dataset_statistics, modality)[state_key].model_dump()
                
                # Split statistics for position (first 3) and rotation (last 4)
                if key in self.position_normalization_modes:
                    self.position_statistics[key] = {}
                    for stat_name, values in full_stats.items():
                        if isinstance(values, list) and len(values) >= 3:
                            self.position_statistics[key][stat_name] = values[:3]
                
                if key in self.rotation_normalization_modes:
                    self.rotation_statistics[key] = {}
                    for stat_name, values in full_stats.items():
                        if isinstance(values, list) and len(values) >= 7:
                            self.rotation_statistics[key][stat_name] = values[3:7]

        # Initialize the rotation transformers
        for key in self.target_rotations:
            if key in self.pose_keys:
                # For pose keys, rotation is always quaternion input
                from_rep = RotationType.QUATERNION
                to_rep = RotationType(self.target_rotations[key])
                
                if from_rep != to_rep:
                    self._rotation_transformers[key] = RotationTransform(
                        from_rep=from_rep.value, to_rep=to_rep.value
                    )
            else:
                # Regular rotation handling
                from_rep = self.modality_metadata[key].rotation_type
                assert from_rep is not None, f"Source rotation type not found for {key}"

                # Get the target representation of the state, will raise an error if the target representation is not valid
                to_rep = RotationType(self.target_rotations[key])

                # If the original representation is not the same as the target representation, initialize the rotation transformer
                if from_rep != to_rep:
                    self._rotation_transformers[key] = RotationTransform(
                        from_rep=from_rep.value, to_rep=to_rep.value
                    )

        # Initialize the normalizers
        for key in self.normalization_modes:
            modality, state_key = key.split(".")
            # If the state has a nontrivial rotation, we need to handle it more carefully
            # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
            # since we can infer the bounds by the representation
            # For relative rotations, we cannot normalize them as we don't know the bounds
            if key in self._rotation_transformers:
                # Case 1: Absolute rotation
                if self.modality_metadata[key].absolute:
                    # Check that the normalization mode is valid
                    assert (
                        self.normalization_modes[key] == "min_max"
                    ), "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
                    rotation_type = RotationType(self.target_rotations[key]).value
                    # If the target representation is euler angles, we need to parse the convention
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    # Get the statistics for the target representation
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                # Case 2: Relative rotation
                else:
                    raise ValueError(
                        f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
                    )
            # If the state is not continuous, we should not use normalization modes other than binary
            elif (
                not self.modality_metadata[key].continuous
                and self.normalization_modes[key] != "binary"
            ):
                raise ValueError(
                    f"{key} is not continuous, so it should be normalized using `binary` mode"
                )
            # Initialize the normalizer
            else:
                statistics = self.normalization_statistics[key]
            self._normalizers[key] = Normalizer(
                mode=self.normalization_modes[key], statistics=statistics
            )

        # Initialize position normalizers for pose keys
        for key in self.position_normalization_modes:
            if key in self.pose_keys and key in self.position_statistics:
                self._position_normalizers[key] = Normalizer(
                    mode=self.position_normalization_modes[key], 
                    statistics=self.position_statistics[key]
                )

        # Initialize rotation normalizers for pose keys
        for key in self.rotation_normalization_modes:
            if key in self.pose_keys:
                if key in self._rotation_transformers:
                    # Use default statistics for target rotation type
                    rotation_type = RotationType(self.target_rotations[key]).value
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                else:
                    # Use computed statistics for quaternions
                    statistics = self.rotation_statistics.get(key, self._DEFAULT_MIN_MAX_STATISTICS["quaternion"])
                
                self._rotation_normalizers[key] = Normalizer(
                    mode=self.rotation_normalization_modes[key], 
                    statistics=statistics
                )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # We allow some keys to be missing in the data, and only process the keys that are present
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
            state = data[key]
            
            if key in self.pose_keys:
                # Handle pose data (position + rotation)
                # Split into position (first 3) and rotation (last 4)
                position = state[..., :3]
                rotation = state[..., 3:7]
                
                # Transform position
                if key in self._position_normalizers:
                    position = self._position_normalizers[key].forward(position)
                
                # Transform rotation
                if key in self._rotation_transformers:
                    rotation = self._rotation_transformers[key].forward(rotation)
                if key in self._rotation_normalizers:
                    rotation = self._rotation_normalizers[key].forward(rotation)
                
                # Concatenate back
                state = torch.cat([position, rotation], dim=-1)
                
            else:
                # Handle regular (non-pose) data
                # Rotate the state
                if key in self._rotation_transformers:
                    state = self._rotation_transformers[key].forward(state)
                # Normalize the state
                if key in self._normalizers:
                    state = self._normalizers[key].forward(state)
            
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            if key in self.pose_keys:
                # Handle pose data (position + rotation)
                # Get the rotation dimension after potential transformation
                if key in self._rotation_transformers:
                    # Get output dimension of rotation transformer
                    rot_dim = {
                        "euler_angles": 3,
                        "quaternion": 4,
                        "rotation_6d": 6,
                        "axis_angle": 3
                    }.get(self.target_rotations.get(key, "quaternion"), 4)
                else:
                    rot_dim = 4  # quaternion
                
                # Split into position and rotation
                position = state[..., :3]
                rotation = state[..., 3:3+rot_dim]
                
                # Reverse rotation transformations
                if key in self._rotation_normalizers:
                    rotation = self._rotation_normalizers[key].inverse(rotation)
                if key in self._rotation_transformers:
                    rotation = self._rotation_transformers[key].inverse(rotation)
                
                # Reverse position transformations
                if key in self._position_normalizers:
                    position = self._position_normalizers[key].inverse(position)
                
                # Concatenate back to original 7D format
                state = torch.cat([position, rotation], dim=-1)
                
            else:
                # Handle regular (non-pose) data
                # Unnormalize the state
                if key in self._normalizers:
                    state = self._normalizers[key].inverse(state)
                # Change the state back to its original representation
                if key in self._rotation_transformers:
                    state = self._rotation_transformers[key].inverse(state)
            assert isinstance(
                state, torch.Tensor
            ), f"State should be tensor after unapplying transformations, but got {type(state)}"
            # Only convert back to the original dtype if it's known, i.e. `apply` was called before
            # If not, we don't know the original dtype, so we don't convert
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data


class StateActionPerturbation(ModalityTransform):
    """
    Class for state or action perturbation.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        std (float): Standard deviation of the noise to be added to the state or action.
    """

    # Configurable attributes
    std: float = Field(
        ..., description="Standard deviation of the noise to be added to the state or action."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't perturb the data in eval mode
            return data
        if self.std < 0:
            # If the std is negative, we don't add any noise
            return data
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            transformed_data_min = torch.min(state)
            transformed_data_max = torch.max(state)
            noise = torch.randn_like(state) * self.std
            state += noise
            # Clip to the original range
            state = torch.clamp(state, transformed_data_min, transformed_data_max)
            data[key] = state
        return data


class StateActionDropout(ModalityTransform):
    """
    Class for state or action dropout.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        dropout_prob (float): Probability of dropping out a state or action.
    """

    # Configurable attributes
    dropout_prob: float = Field(..., description="Probability of dropping out a state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't drop out the data in eval mode
            return data
        if self.dropout_prob < 0:
            # If the dropout probability is negative, we don't drop out any states
            return data
        if self.dropout_prob > 1e-9 and random.random() < self.dropout_prob:
            for key in self.apply_to:
                state = data[key]
                assert isinstance(state, torch.Tensor)
                state = torch.zeros_like(state)
                data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    """
    Class for state or action sin-cos transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            sin_state = torch.sin(state)
            cos_state = torch.cos(state)
            data[key] = torch.cat([sin_state, cos_state], dim=-1)
        return data


class AbsoluteToRelativeAction(InvertibleModalityTransform):
    """
    Convert absolute actions to relative w.r.t. a reference state, and invert back.

    Supports two modes:
    - "joint": element-wise difference action - reference_state (e.g., joint positions)
    - "pose": action is [x,y,z,qx,qy,qz,qw], state is [x,y,z,qx,qy,qz,qw]

    For pose:
    - translation: action_xyz - ref_xyz (optionally rotated to EE frame if in_ee_frame=True)
    - rotation: q_rel = q_ref^{-1} * q_action, then represented as axis-angle (rx,ry,rz)

    The reference state is taken as the last timestep in the provided state sequence.
    The reference for each action key is cached during apply() and used in unapply().
    """

    # Pairs of (action_key, state_key)
    action_state_pairs: list[tuple[str, str]] = Field(..., description="Pairs of ('action.key', 'state.key') to process")
    mode: str = Field(default="joint", description="'joint' or 'pose'")
    in_ee_frame: bool = Field(default=False, description="When mode='pose', express translation in EE frame")
    rotation_out: str = Field(default="quaternion", description="Output rotation format")
    enabled: bool = Field(default=True, description="If False, no-op in apply/unapply")

    _cached_refs: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)

    def _compute_relative_joint(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # action/state: [T, D]
        ref = state[[-1], :]  # [1, D]
        self._current_ref = ref
        return action - ref

    @staticmethod
    def _quat_inverse(q: torch.Tensor) -> torch.Tensor:
        # q: [..., 4] in xyzw
        q_conj = torch.clone(q)
        q_conj[..., :3] = -q_conj[..., :3]
        norm = torch.sum(q * q, dim=-1, keepdim=True)
        return q_conj / torch.clamp(norm, min=1e-8)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        # Hamilton product, xyzw
        x1, y1, z1, w1 = q1.unbind(dim=-1)
        x2, y2, z2, w2 = q2.unbind(dim=-1)
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return torch.stack([x, y, z, w], dim=-1)

    @staticmethod
    def _quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
        # Convert unit quaternion (xyzw) to axis-angle (rx, ry, rz)
        q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)
        # angle = 2*acos(w), axis = v/|v|
        w = torch.clamp(q[..., 3], -1.0, 1.0)
        angle = 2.0 * torch.acos(w)
        s = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-12))
        v = q[..., :3] / s.unsqueeze(-1)
        # For small angles, fall back to linear
        small = s.squeeze(-1) < 1e-6
        v[small] = q[..., :3][small]
        angle = angle.unsqueeze(-1)
        return v * angle

    @staticmethod
    def _axis_angle_to_quat(r: torch.Tensor) -> torch.Tensor:
        # r: [..., 3]
        angle = torch.linalg.norm(r, dim=-1, keepdim=True)
        axis = r / torch.clamp(angle, min=1e-8)
        half = 0.5 * angle
        sin_half = torch.sin(half)
        x, y, z = (axis * sin_half).unbind(dim=-1)
        w = torch.cos(half).squeeze(-1)
        return torch.stack([x, y, z, w], dim=-1)

    @staticmethod
    def _rotate_vec_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # v: [..., 3], q: [..., 4] xyzw
        # Convert v to pure quaternion
        zeros = torch.zeros_like(v[..., :1])
        v_quat = torch.cat([v, zeros], dim=-1)
        q_inv = AbsoluteToRelativeAction._quat_inverse(q)
        rotated = AbsoluteToRelativeAction._quat_multiply(
            AbsoluteToRelativeAction._quat_multiply(q, v_quat), q_inv
        )
        return rotated[..., :3]

    def _compute_relative_pose(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # action/state: [T, 7] -> [T, 6] (xyz + axis-angle)
        assert action.shape[-1] == 7 and state.shape[-1] == 7, f"Expected pose dims 7, got {action.shape[-1]}, {state.shape[-1]}"
        ref = state[[-1], :]  # [1, 7]
        self._cached_ref_pose = ref
        t_rel = action[..., :3] - ref[..., :3]
        if self.in_ee_frame:
            ref_q = ref[..., 3:7].expand_as(action[..., 3:7])
            t_rel = self._rotate_vec_by_quat(t_rel, self._quat_inverse(ref_q))
        q_ref = ref[..., 3:7].expand_as(action[..., 3:7])
        q_rel = self._quat_multiply(self._quat_inverse(q_ref), action[..., 3:7])
        
        if self.rotation_out == "rotation_6d":
            # reuse the existing RotationTransform in this file
            r6d = RotationTransform(from_rep="quaternion", to_rep="rotation_6d").forward(q_rel)
            rel = torch.cat([t_rel, r6d], dim=-1)  # [T, 9]
        elif self.rotation_out == "axis_angle":
            rel = torch.cat([t_rel, self._quat_to_axis_angle(q_rel)], dim=-1)  # [T, 6]
        else:  # quaternion
            rel = torch.cat([t_rel, q_rel], dim=-1)  # [T, 7]

        return rel

    def _invert_relative_joint(self, rel: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return rel + ref

    def _invert_relative_pose(self, rel: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # rel: [T, 6/9/7], ref: [1, 7] -> abs [T, 7]
        t_rel = rel[..., :3]
        
        # Convert rotation back to quaternion based on input format
        if self.rotation_out == "rotation_6d":
            # rel: [T, 9] -> r6d: [T, 6]
            r6d = rel[..., 3:9]
            dq = RotationTransform(from_rep="rotation_6d", to_rep="quaternion").forward(r6d)
        elif self.rotation_out == "axis_angle":
            # rel: [T, 6] -> axis_angle: [T, 3]
            r_rel = rel[..., 3:6]
            dq = self._axis_angle_to_quat(r_rel)
        else:  # quaternion
            # rel: [T, 7] -> quat: [T, 4]
            dq = rel[..., 3:7]
        
        # Handle EE frame translation
        if self.in_ee_frame:
            ref_q = ref[..., 3:7]
            if ref_q.shape[0] == 1 and rel.shape[0] > 1:
                ref_q = ref_q.expand(rel.shape[0], -1)
            t_rel = self._rotate_vec_by_quat(t_rel, ref_q)
        
        # Compose absolute translation and rotation
        ref_pos = ref[..., :3]
        if ref_pos.shape[0] == 1 and rel.shape[0] > 1:
            ref_pos = ref_pos.expand(rel.shape[0], -1)
        t_abs = ref_pos + t_rel
        
        ref_q = ref[..., 3:7]
        if ref_q.shape[0] == 1 and rel.shape[0] > 1:
            ref_q = ref_q.expand(rel.shape[0], -1)
        q_abs = self._quat_multiply(ref_q, dq)
        
        return torch.cat([t_abs, q_abs], dim=-1)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return data
        # Reset cache per call
        self._cached_refs.clear()
        for action_key, state_key in self.action_state_pairs:
            state_present = state_key in data and isinstance(data[state_key], torch.Tensor)
            action_present = action_key in data and isinstance(data[action_key], torch.Tensor)

            # Always cache the reference state if available (supports inference path where actions are absent)
            if state_present:
                # Store the reference state with proper shape handling
                ref_state = data[state_key][-1:, :]  # [1, D] - ensure proper shape
                self._cached_refs[action_key] = ref_state.clone()

            if not (state_present and action_present):
                # Nothing to convert for this pair
                continue

            action = data[action_key]
            state = data[state_key]
            if self.mode == "joint":
                rel = self._compute_relative_joint(action, state)
            elif self.mode == "pose":
                rel = self._compute_relative_pose(action, state)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            data[action_key] = rel
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return data
        # Use cached references to reconstruct absolute actions
        for action_key, state_key in self.action_state_pairs:
            if action_key not in data or action_key not in self._cached_refs:
                continue
            rel = data[action_key]
            ref = self._cached_refs[action_key]
            assert isinstance(rel, torch.Tensor) and isinstance(ref, torch.Tensor)
            if self.mode == "joint":
                abs_action = self._invert_relative_joint(rel, ref)
            elif self.mode == "pose":
                abs_action = self._invert_relative_pose(rel, ref)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            data[action_key] = abs_action
        # Clear cache after use
        self._cached_refs.clear()
        return data
