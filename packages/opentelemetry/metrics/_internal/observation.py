# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

from opentelemetry.util.types import Attributes


class Observation:
    """A measurement observed in an asynchronous instrument

    Return/yield instances of this class from asynchronous instrument callbacks.

    Args:
        value: The float or int measured value
        attributes: The measurement's attributes
    """

    def __init__(
        self, value: Union[int, float], attributes: Attributes = None
    ) -> None:
        self._value = value
        self._attributes = attributes

    @property
    def value(self) -> Union[float, int]:
        return self._value

    @property
    def attributes(self) -> Attributes:
        return self._attributes

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Observation)
            and self.value == other.value
            and self.attributes == other.attributes
        )

    def __repr__(self) -> str:
        return f"Observation(value={self.value}, attributes={self.attributes})"
