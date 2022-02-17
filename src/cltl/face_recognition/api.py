import abc
import dataclasses
from typing import List, Optional, Iterable

import numpy as np
from cltl.backend.api.camera import Bounds
from emissor.representation.entity import Gender


@dataclasses.dataclass
class Face:
    """
    Detected Face with the position in the image, a vector representation of the face
    and additional meta information.
    """
    bounds: Bounds
    embedding: np.ndarray
    gender: Optional[Gender]
    age: Optional[int]


class FaceDetector(abc.ABC):
    """
    Detect faces in an image.
    """

    def detect(self, image: np.ndarray) -> Iterable[Face]:
        """
        Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray
            The binary image.

        Returns
        -------
        Iterable[Face]
            The faces detected in the image.

        """
        raise NotImplementedError()
