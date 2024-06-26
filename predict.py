# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cv2
import wandb
import numpy as np
from typing import Optional
from pymatting import cutout
from pydantic import BaseModel, Field
from cog import BasePredictor, Input, Path
from constants import WANDB_API_KEY, DIRECTORY_TO_SAVE_IMAGE_OVERLAY
from prediction_using_vit_and_skin_cut import SkinSegmentVitMatte
import sys
import os

from utils import initialize_wandb, log_results_to_wandb, overlay_final_mask_in_black_background

sys.path.append(os.path.abspath('./ViTMatte'))
wandb.login(key=WANDB_API_KEY)


class Output(BaseModel):
    success: bool
    error: Optional[str]
    segmented_image_pyMatting: Optional[Path]
    trimap: Optional[Path]
    segmented_image_vit_matte: Optional[Path] = Field(default="")
    segmented_image_modified_matte: Optional[Path] = Field(default="")
    edge_less_cutout_no_mask: Optional[Path] = Field(default="")
    embedding_check: Optional[bool] = Field(default=False)
    embedding_check_failure_reason: Optional[str] = Field(default="")
    vit_and_modifier_algo_success: Optional[str] = Field(default="")
    embedding_distance: Optional[str] = Field(default="")


class Predictor(BasePredictor):
    def __init__(self):
        self.output_path = None
        self.trimap_path = None
        self.mask_path = None
        self.image_path = None

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.image_path = "/tmp/image.jpg"
        self.mask_path = "/tmp/mask.png"
        self.trimap_path = "/tmp/trimap.png"
        self.output_path = "/tmp/output.png"

    def predict(
            self,
            product_id : str = Input(description="Please enter Product ID here", default=None),
            image: Path = Input(description="Input image"),
            mask: Path = Input(description="Mask image", default=None),
            trimap: Path = Input(description="Trimap image", default=None),
    ) -> Output:
        # if there's no mask/trimap, return an error
        initialize_wandb(product_id)
        if mask is None and trimap is None:
            return Output(segmentedImage=None, success=False, error="Must provide either mask or trimap")

        # if mask is present, create trimap using mask and then cut out the image
        if mask is not None:
            # read save image from inputs to disk
            image_mat = cv2.imread(str(image))
            cv2.imwrite(self.image_path, image_mat)

            # read mask from inputs
            mask_mat = cv2.imread(str(mask))

            erosion_kernel = np.ones((5, 5), np.uint8)
            dilation_kernel = np.ones((3, 3), np.uint8)

            # convert mask to greyscale, erode and dilate to create trimap
            mask_image = cv2.cvtColor(mask_mat, cv2.COLOR_BGR2GRAY)
            eroded_image = cv2.erode(mask_image, erosion_kernel, iterations=1)
            dilated_image = cv2.dilate(mask_image, dilation_kernel, iterations=1)

            # base for the trimap
            new_image = np.zeros((eroded_image.shape[0], eroded_image.shape[1]), np.uint8)

            for i in range(0, eroded_image.shape[0]):
                for j in range(0, eroded_image.shape[1]):
                    erosion_pixel = eroded_image[i][j]
                    dilation_pixel = dilated_image[i][j]

                    if erosion_pixel > 0:
                        new_image[i][j] = 255
                    elif dilation_pixel > 0:
                        new_image[i][j] = 127

            cv2.imwrite(self.trimap_path, new_image)

            # write trimap to disk for debugging
            cv2.imwrite("trimap.png", new_image)
            # log_results_to_wandb("Image", self.image_path)
            # log_results_to_wandb("Segmentation Mask", self.mask_path)
            # log_results_to_wandb("Trimap", self.trimap_path)
            vit_matte_and_skin_cut_matte = SkinSegmentVitMatte().generate_modified_matted_results(image,
                                                                                                  self.trimap_path)
            print("vit matte and skin cut matte is ", vit_matte_and_skin_cut_matte)
            cutout(self.image_path, self.trimap_path, self.output_path)
            output = cv2.imread(self.output_path)
            cv2.imwrite("output.png", output)
            if vit_matte_and_skin_cut_matte["success"]:
                output_from_vit_model = vit_matte_and_skin_cut_matte["vit_matte_path"]
                output_from_modifier_model = vit_matte_and_skin_cut_matte["modified_matte_path"]
                embedding_check_success = vit_matte_and_skin_cut_matte["embedding_check_label"]
                error_log = vit_matte_and_skin_cut_matte["error_reason"]
                error_from_vit = ""
                distance = vit_matte_and_skin_cut_matte["distance"]
                edge_less_no_mask_path = vit_matte_and_skin_cut_matte["edge_less_no_mask"]

                # log_results_to_wandb("VIT Matte Alpha Output", output_from_vit_model)
                # log_results_to_wandb("VIT Matte cutout Output",
                #                      vit_matte_and_skin_cut_matte["vit_matte_cutout_image"])
                # log_results_to_wandb("Skin Cut Model Output",
                #                      vit_matte_and_skin_cut_matte["skin_cut_output"])
                # log_results_to_wandb("Final Modified Mask", output_from_modifier_model)
                image_overlay_dir = os.path.join(DIRECTORY_TO_SAVE_IMAGE_OVERLAY, "overlay.png")
                image_overlay = overlay_final_mask_in_black_background(self.image_path,
                                                                       vit_matte_and_skin_cut_matte["non_converted_final_mask"],
                                                                       image_overlay_dir)
                if image_overlay["success"]:
                    log_results_to_wandb("Image Overlay", image_overlay)
            else:
                output_from_vit_model = vit_matte_and_skin_cut_matte["error"]
                output_from_modifier_model = vit_matte_and_skin_cut_matte["error"]
                embedding_check_success = False
                error_log = ""
                distance = ""
                error_from_vit = vit_matte_and_skin_cut_matte["error"]
                edge_less_no_mask_path = ""
            # cutout the image using pymatting
            return Output(segmented_image_pyMatting=Path(self.output_path), trimap=Path(self.trimap_path),
                          segmented_image_vit_matte=Path(output_from_vit_model),
                          segmented_image_modified_matte=Path(output_from_modifier_model),
                          edge_less_cutout_no_mask=Path(edge_less_no_mask_path),
                          embedding_check=embedding_check_success,
                          embedding_check_failure_reason=error_log,
                          vit_and_modifier_algo_success=error_from_vit,
                          success=True,
                          embedding_distance=distance)

        else:
            # if execution reaches here, trimap must be present, thus,
            # cut the image using the trimap

            trimap = cv2.imread(str(trimap))
            cv2.imwrite(self.trimap_path, trimap)

            # cutout the image using pymatting
            cutout(self.image_path, self.trimap_path, self.output_path)

            return Output(segmented_image_pyMatting=Path(self.output_path), trimap=Path(self.trimap_path),
                          segmented_image_vit_matte=Path(self.output_path),
                          segmented_image_modified_matte=Path(self.output_path),
                          edge_less_cutout_no_mask=Path(self.output_path),
                          embedding_check=False,
                          embedding_check_failure_reason="",
                          vit_and_modifier_algo_success=False,
                          success=True,
                          embedding_distance="")
