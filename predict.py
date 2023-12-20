# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cv2
import numpy as np
from typing import Optional
from pymatting import cutout
from pydantic import BaseModel
from cog import BasePredictor, Input, Path, File

class Output(BaseModel):
    success: bool
    error: Optional[str]
    segmentedImage: Optional[Path]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

    def predict(
        self,
        image: Path = Input(description="Input image"),
        mask: Path = Input(description="Mask image", default=None),
        trimap: Path = Input(description="Trimap image", default=None),
    ) -> Output:
        print("images is ", image)
        print("mask is ", mask)
        print("trimap is ", trimap)

        # if there's no mask/trimap, return an error
        if mask is None and trimap is None:
            return Output(segmentedImage=None, success=False, error="Must provide either mask or trimap")

        # if mask is present, create trimap using mask and then cut out the image
        if mask is not None:
            
            maskMat = cv2.imread(str(mask))

            kernel = np.ones((10, 10), np.uint8)

            maskImage = cv2.cvtColor(maskMat, cv2.COLOR_BGR2GRAY)
            erodedImage = cv2.erode(maskImage, kernel, iterations=1)
            dilatedImage = cv2.dilate(maskImage, kernel, iterations=1)

            newImage = np.zeros((erodedImage.shape[0], erodedImage.shape[1]), np.uint8)


            for i in range(0, erodedImage.shape[0]):
                for j in range(0, erodedImage.shape[1]):
                    erosionPixel = erodedImage[i][j]
                    dilationPixel = dilatedImage[i][j]

                    if erosionPixel > 0:
                        newImage[i][j] = 255
                    elif dilationPixel > 0:
                        newImage[i][j] = 127
            
            cv2.imwrite("/tmp/trimap.png", newImage)

            cutout(image, "/tmp/trimap.png", "output.png")

            output = cv2.imread("output.png")
            cv2.imwrite("output.png", output)
        
        else:
            # if execution reaches here, trimap must be present, thus
            # cut the image using the trimap

            trimap = cv2.imread(str(trimap))

            cutout(image, trimap, "output.png")

        return Output(segmentedImage=Path("output.png"), success=True)