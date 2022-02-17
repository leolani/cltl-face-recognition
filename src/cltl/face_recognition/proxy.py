import logging
from typing import Iterable

import jsonpickle
import numpy as np
import requests
from cltl.backend.api.camera import Bounds
from emissor.representation.entity import Gender

from cltl.face_recognition.api import Face
from cltl.face_recognition.docker import DockerInfra


class FaceDetectorProxy:
    def __init__(self, base_path: str, port_docker_face_analysis: int, run_on_gpu: int):
        self._base_path = base_path
        self.face_infra = DockerInfra('face-analysis-cuda' if run_on_gpu else 'face-analysis',
                                      port_docker_face_analysis, 30000, run_on_gpu, 30)

    def __enter__(self):
        self.face_infra.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.face_infra.__exit__(exc_type, exc_val, exc_tb)

    def detect(self, image: np.ndarray) -> Iterable[Face]:
        logging.info("Processing image %s")

        data = jsonpickle.encode({'image': image})
        response = requests.post(
            f"{'http://127.0.0.1'}:{self.face_infra.host_port}/", json=data)
        if response.ok:
            logging.info("%s received", response)
        else:
            raise ValueError(f"{response} received with reason {response.reason}")

        response = jsonpickle.decode(response.text)

        return map(self.to_face, response['fa_results'])

    def to_face(self, face_result):
        bbox = [int(num) for num in face_result['bbox'].tolist()]
        representation = face_result['normed_embedding']
        if 'gender' in face_result['gender'] and face_result['gender'] == 1:
            gender = Gender.MALE
        elif 'gender' in face_result['gender'] and face_result['gender'] is not None:
            gender = Gender.FEMALE
        else:
            gender = None
        age = face_result['age']

        return Face(Bounds(bbox[0], bbox[2], bbox[1], bbox[3]), representation, gender, age)

