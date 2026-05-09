import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        z_max = max(z)
        z_shifted = z - z_max
        denom = sum(np.exp(z_shifted))
        softmax = np.exp(z_shifted)/denom
        return np.round(softmax, 4)
