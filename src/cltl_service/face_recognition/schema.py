import uuid
from cltl.backend.api.camera import Bounds
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, ImageSignal, Annotation
from typing import Iterable

from cltl.face_recognition.api import Face


@dataclass
class FaceRecognitionEvent(AnnotationEvent[Annotation[Face]]):
    @classmethod
    def create(cls, image_signal: ImageSignal, faces: Iterable[Face], bounds: Iterable[Bounds]):
        def to_mention(face, bounds):
            segment = image_signal.ruler.get_area_bounding_box(bounds.x0, bounds.y0, bounds.x1, bounds.y1)
            annotation = Annotation(Face.__name__, face, __name__, timestamp_now())

            return Mention(str(uuid.uuid4()), [segment], [annotation])

        return cls(cls.__name__, [to_mention(face, bound) for face, bound in zip(faces, bounds)])