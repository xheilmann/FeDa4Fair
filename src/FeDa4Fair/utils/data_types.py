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
# ==============================================================================

"""Data types and dataclasses used for structured data sharing."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ProcessedSiloData:
    """Structured container for processed silo data used in cross-silo and cross-device evaluation."""

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    sf_data: dict[str, np.ndarray]

    def to_tuple(self) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Return the data as a tuple (for backward compatibility during refactoring)."""
        return (self.name, self.x_train, self.y_train, self.x_test, self.y_test, self.sf_data)
