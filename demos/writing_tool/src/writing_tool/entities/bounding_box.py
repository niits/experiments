from dataclasses import dataclass


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]
