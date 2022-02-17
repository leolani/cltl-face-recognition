import logging
from typing import Iterable, Tuple

import jsonpickle
import numpy as np
import requests
from cltl.backend.api.camera import Bounds
from emissor.representation.entity import Gender

from cltl.face_recognition.api import Face, FaceDetector
from cltl.face_recognition.docker import DockerInfra


class FaceDetectorProxy(FaceDetector):
    def __init__(self, port_docker_face_analysis: int, run_on_gpu: int):
        self.face_infra = DockerInfra('face-analysis-cuda' if run_on_gpu else 'face-analysis',
                                      port_docker_face_analysis, 30000, run_on_gpu, 30)

    def __enter__(self):
        self.face_infra.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.face_infra.__exit__(exc_type, exc_val, exc_tb)

    def detect(self, image: np.ndarray) -> Tuple[Iterable[Face], Iterable[Bounds]]:
        logging.info("Processing image %s")

        data = jsonpickle.encode({'image': image})
        response = requests.post(
            f"{'http://127.0.0.1'}:{self.face_infra.host_port}/", json=data)
        if response.ok:
            logging.info("%s received", response)
        else:
            raise ValueError(f"{response} received with reason {response.reason}")

        response = jsonpickle.decode(response.text)

        return zip(*map(self._to_face, response['fa_results']))

    def _to_face(self, face_result) -> Tuple[Face, Bounds]:
        bbox = [int(num) for num in face_result['bbox'].tolist()]
        representation = face_result['normed_embedding']
        if 'gender' in face_result['gender'] and face_result['gender'] == 1:
            gender = Gender.MALE
        elif 'gender' in face_result['gender'] and face_result['gender'] is not None:
            gender = Gender.FEMALE
        else:
            gender = None
        age = face_result['age']

        return Face(representation, gender, age), Bounds(bbox[0], bbox[2], bbox[1], bbox[3])

