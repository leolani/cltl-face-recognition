from dataclasses import dataclass
from typing import Iterable

from cltl.backend.api.camera import Bounds
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from emissor.representation.container import MultiIndex
from emissor.representation.scenario import Mention, ImageSignal, Annotation

from cltl.face_recognition.api import Face


@dataclass
class FaceRecognitionEvent(AnnotationEvent[Annotation[Face]]):
    @classmethod
    def create(cls, image_signal: ImageSignal, faces: Iterable[Face], bounds: Iterable[Bounds]):
        def to_mention(face, bounds):
            segment = image_signal.get_segment(MultiIndex(bounds.x0, bounds.y0, bounds.x1, bounds.y1))
            annotation = Annotation(Face.__name__, face, __name__, timestamp_now())

            return Mention([segment], [annotation])

        return cls(cls.__name__, [to_mention(face, bound) for face, bound in zip(faces, bounds)])