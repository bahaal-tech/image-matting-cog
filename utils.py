import os
import cv2
import torch
import numpy as np
from PIL import Image
from os.path import join as opj
from torchvision.transforms import functional as F
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from constants import HAR_CASCADES_PATH, FACE_SCALE_FACTOR, MIN_NEIGHBOURS, MIN_SIZE, LOWER_SKIN_BOUNDARIES, \
    UPPER_SKIN_BOUNDARIES, MATRIX_TRANSFORM, NORMALIZE_MEAN, NORMALIZE_STD, SIMILARITY_DIMENSION


def generate_inference_from_one_image(model, input_image, dir_to_save=None):
    """
        Infer the alpha matte of one image.
        Input:
            model: the trained model
            image: the input image
            trimap: the input trimap
    """
    output = model(input_image)['phas'].flatten(0, 2)
    output = F.to_pil_image(output)
    output.save(opj(dir_to_save))
    return None


def model_initializer(model, checkpoint, device):
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    assert model in ['vitmatte-s', 'vitmatte-b']
    if model == 'vitmatte-s':
        config = 'ViTMatte/configs/common/model.py'
        cfg = LazyConfig.load(config)
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    elif model == 'vitmatte-b':
        config = 'ViTMatte/configs/common/model.py'
        cfg = LazyConfig.load(config)
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans = 768
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    return model


def generate_model_input(input_image, image_trimap):
    """
    Get the data of one image.
    Input:
        image_dir: the directory of the image
        trimap_dir: the directory of the trimap
    """
    image = Image.open(input_image).convert('RGB')
    image_tensor = F.to_tensor(image).unsqueeze(0)
    trimap = Image.open(image_trimap).convert('L')
    trimap_tensor = F.to_tensor(trimap).unsqueeze(0)

    return {
        'image': image_tensor,
        'trimap': trimap_tensor
    }


def calculate_foreground(input_image, alpha_matte, output_path):
    """
    Calculate the foreground of the image.
    Input:
        image_dir: the directory of the image
        alpha_dir: the directory of the alpha matte
        save_path: the path to save the resulting foreground image
    Output:
        None
    """
    image = Image.open(input_image).convert('RGB')
    alpha = Image.open(alpha_matte)

    alpha = F.to_tensor(alpha).unsqueeze(0)
    image = F.to_tensor(image).unsqueeze(0)

    foreground = image * alpha
    foreground = foreground.squeeze(0).permute(1, 2, 0).numpy()

    foreground = (foreground * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(foreground, cv2.COLOR_RGBA2BGRA))


def alpha_matte_inference_from_vision_transformer(model, input_image, trimap_image, directory_to_save):
    try:
        input_to_vit_model = generate_model_input(input_image, trimap_image)
        if not os.path.exists(directory_to_save):
            os.mkdir(directory_to_save)
        save_dir_for_matte = os.path.join(directory_to_save, 'matte.png')
        generate_inference_from_one_image(model, input_to_vit_model, save_dir_for_matte)
        convert_greyscale_image_to_transparent(save_dir_for_matte, save_dir_for_matte)
        dir_for_cutout = os.path.join(directory_to_save, 'cutout.png')
        calculate_foreground(input_image, save_dir_for_matte, dir_for_cutout)
        return {"success": True, "vit_matte_output": save_dir_for_matte}
    except Exception as e:
        return {"success": False, "error": f"Vit Matte model failed due : {e}"}


def detect_face_and_hsv_from_images(image):
    input_image = cv2.imread(str(image))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAR_CASCADES_PATH)
    gray_scale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    try:
        faces = face_cascade.detectMultiScale(gray_scale_image, scaleFactor=FACE_SCALE_FACTOR,
                                              minNeighbors=MIN_NEIGHBOURS, minSize=MIN_SIZE)
        if len(faces) == 0:
            return {"success": False, "error": f"Number of face detected: {len(faces)}"}
        for (x, y, w, h) in faces:
            region_of_interest = input_image[y:y + h, x:x + w]
            roi_ycrcb = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2YCrCb)
            lower_skin = np.array(LOWER_SKIN_BOUNDARIES, dtype=np.uint8)
            upper_skin = np.array(UPPER_SKIN_BOUNDARIES, dtype=np.uint8)
            mask_skin = cv2.inRange(roi_ycrcb, lower_skin, upper_skin)
            skin_color = cv2.mean(region_of_interest, mask=mask_skin)[:3]
            skin_color = np.uint8([[skin_color]])
            skin_color_hsv = cv2.cvtColor(skin_color, cv2.COLOR_BGR2HSV)
            h, s, v = skin_color_hsv[0][0]
            return {"success": True, "h_value": h, "s_value": s, "v_value": v}
    except Exception as e:
        return {"success": False, "error": f"Face detection or hsv extraction failed due to :{e}"}


def selective_search_and_remove_skin_tone(input_image, matte_image, threshold_for_hsv, directory_to_save):
    hsv_values = detect_face_and_hsv_from_images(input_image)
    if not hsv_values["success"]:
        return {"success": False, "error": f"Either no face detected or might be internal error, "
                                           f"kindly refer:{hsv_values['error']}"}
    h_value = hsv_values["h_value"]
    s_value = hsv_values["s_value"]
    v_value = hsv_values["v_value"]

    image_matte = cv2.imread(str(matte_image))
    hsv_of_matte = cv2.cvtColor(image_matte, cv2.COLOR_BGR2HSV)
    target_hsv = np.array([h_value, s_value, v_value])

    lower_hsv = np.array([target_hsv[0] - threshold_for_hsv, target_hsv[1] - threshold_for_hsv,
                          target_hsv[2] - threshold_for_hsv])
    upper_hsv = np.array([target_hsv[0] + threshold_for_hsv, target_hsv[1] + threshold_for_hsv,
                          target_hsv[2] + threshold_for_hsv])

    mask = cv2.inRange(hsv_of_matte, lower_hsv, upper_hsv)
    inverse_mask = cv2.bitwise_not(mask)  # Invert the mask
    result = np.copy(image_matte)
    result[inverse_mask == 0] = [255, 255, 255]  # Keep the non-masked area

    if not os.path.exists(directory_to_save):
        os.mkdir(directory_to_save)
    output_dir_for_modified_matte = os.path.join(directory_to_save, 'modified_cutout.png')
    cv2.imwrite(output_dir_for_modified_matte, result)
    return {"success": True, "output": output_dir_for_modified_matte}


def get_image_embeddings(model, image_path):
    import torchvision.transforms as transforms
    preprocess_input_image = transforms.Compose([
        transforms.Resize(MATRIX_TRANSFORM),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),])
    try:
        image = Image.open(image_path)
        img = preprocess_input_image(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img)
        return {"success": True, "embeddings": embedding.squeeze().numpy()}
    except Exception as e:
        return {"success": False, "error": f"Image embedding creation failed due to {e}"}


def calculate_cosine_distance(modified_matting_image, vit_matted_image):
    try:
        embedding_new_matte = torch.from_numpy(modified_matting_image)
        embedding_vit_matte = torch.from_numpy(vit_matted_image)
        similarity = torch.nn.functional.cosine_similarity(embedding_new_matte, embedding_vit_matte,
                                                           dim=SIMILARITY_DIMENSION)
        return {"success": True, "similarity": similarity.item()}
    except Exception as e:
        return {"success": False, "error": f"Calculation of cosine similarity failed due to{e}"}


def calculate_embeddings_diff_between_two_images(modified_matting_image, vit_matted_image, model):
    modified_image_embedding = get_image_embeddings(model, modified_matting_image)
    vit_matte_image_embedding = get_image_embeddings(model, vit_matted_image)
    if modified_image_embedding["success"] and vit_matte_image_embedding["success"]:
        cosine_distance = calculate_cosine_distance(modified_image_embedding["embeddings"],
                                                    vit_matte_image_embedding["embeddings"])
        if not cosine_distance["success"]:
            return {"success": False, "error": f"Similarity check failed due to :{cosine_distance['error']}"}
        return {"success": True, "cosine_distance": cosine_distance}
    else:
        return {"success": False, "error": f"Similarity check failed due to {modified_image_embedding} "
                                           f"{vit_matte_image_embedding}"}

def convert_greyscale_image_to_transparent(input_image_path, output_path):
    """
    Converts a greyscale image to a transparent one. Makes
    absolute black pixels transparent, absolute white opaque
    and grey ones to intermediate alpha values.
    Input:
        input_image_path: Path on disk for input greyscale image
        output_path: The path where transparent image needs to be written
    """
    input_image = cv2.imread(input_image_path)

    alpha_image = np.zeros(input_image.shape)

    alpha_channel = cv2.cvtColor(input_image, cv2.COLOR_BGRA2GRAY)

    color_channel = np.where(alpha_channel > 0, 255, 0)

    alpha_image[:, :, 0] = color_channel
    alpha_image[:, :, 1] = color_channel
    alpha_image[:, :, 2] = color_channel
    alpha_image[:, :, 3] = alpha_channel

    cv2.imwrite(output_path, alpha_image)