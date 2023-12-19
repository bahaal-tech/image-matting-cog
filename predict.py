# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from pymatting import cutout
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        trimap: Path = Input(description="Trimap image"),
    ) -> Path:
        cutout(image, trimap, "output.png")

        return Path("output.png")