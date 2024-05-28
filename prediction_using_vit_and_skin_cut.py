import torch
import os
import torch.nn as nn
from constants import VIT_MATTE_MODEL_NAME, THRESHOLD, DIRECTORY_TO_SAVE_VIT_MATTE, \
    DIRECTORY_TO_SAVE_MODIFIED_MATTE, EMBEDDING_THRESHOLD, MODEL_DIR, DIRECTORY_TO_SAVE_EDGE_LESS_MATTE
from utils import model_initializer, alpha_matte_inference_from_vision_transformer, \
    selective_search_and_remove_skin_tone, calculate_embeddings_diff_between_two_images, \
    extra_edge_removal_from_matte_output, convert_greyscale_image_to_transparent
import torchvision.models as models


class SkinSegmentVitMatte:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vit_matte_model = model_initializer(model=VIT_MATTE_MODEL_NAME, checkpoint=MODEL_DIR,
                                                 device=self.device)
        resnet = models.resnet50(pretrained=True)
        self.embedding_model = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding_model.eval()

    def generate_modified_matted_results(self, input_image, trimap_image):
        cutout_image_from_vit_matting = alpha_matte_inference_from_vision_transformer(self.vit_matte_model, input_image,
                                                                                      trimap_image,
                                                                                      DIRECTORY_TO_SAVE_VIT_MATTE)
        if not cutout_image_from_vit_matting["success"]:
            return {"success": False, "error": f"Vit matting failed due to: {cutout_image_from_vit_matting}"}

        edge_less_matte = extra_edge_removal_from_matte_output(cutout_image_from_vit_matting["cutout_output"],
                                                               DIRECTORY_TO_SAVE_EDGE_LESS_MATTE)
        if not edge_less_matte["success"]:
            edge_less_matte_mask_path = cutout_image_from_vit_matting["vit_matte_output"]
            print(f"{edge_less_matte['error']}")
        else:
            dir_for_edge_less_alpha_output = os.path.join(DIRECTORY_TO_SAVE_EDGE_LESS_MATTE, 'edge_less_alpha.png')
            # convert_greyscale_image_to_transparent(edge_less_matte["mask_edge_less_path"],
            #                                        dir_for_edge_less_alpha_output)
            edge_less_matte_mask_path = edge_less_matte["mask_edge_less_path"]
        print(edge_less_matte_mask_path)
        modified_matte = selective_search_and_remove_skin_tone(input_image,
                                                               edge_less_matte_mask_path,
                                                               THRESHOLD, DIRECTORY_TO_SAVE_MODIFIED_MATTE)

        print("cutout image from vit matting is ", cutout_image_from_vit_matting)

        if not modified_matte["success"]:
            return {"success": False, "error": f"matting modifications failed due to:{modified_matte['error']}",
                    "vit_matte_path": edge_less_matte_mask_path}

        distance_between_modified_and_vit_matte = calculate_embeddings_diff_between_two_images(
            modified_matte["output"], edge_less_matte_mask_path, self.embedding_model)
        if not distance_between_modified_and_vit_matte["success"]:
            return {"success": True, "vit_matte_path": edge_less_matte_mask_path,
                    "modified_matte_path": modified_matte["output"], "embedding_check_label": False, "error_reason":
                    distance_between_modified_and_vit_matte["error"], "distance": ""}
        if distance_between_modified_and_vit_matte["cosine_distance"]["similarity"] > EMBEDDING_THRESHOLD:
            return {"success": True, "vit_matte_path": edge_less_matte_mask_path,
                    "modified_matte_path": modified_matte["output"], "embedding_check_label": True, "error_reason": "",
                    "distance": distance_between_modified_and_vit_matte["cosine_distance"]["similarity"]}
        else:
            return {"success": True, "vit_matte_path": edge_less_matte_mask_path,
                    "modified_matte_path": modified_matte["output"], "embedding_check_label": True, "error_reason": "",
                    "distance": distance_between_modified_and_vit_matte["cosine_distance"]["similarity"]
                    }
