import os
import cv2
import numpy as np
import supervision as sv
from typing import List
from PIL import Image
import torch
from inference.models import YOLOWorld
from .utils.efficient_sam import inference_with_boxes

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

class YoloWorldModelLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
             "yolo_world_model": (["yolo_world/l", "yolo_world/m", "yolo_world/s","yolo_world/v2-l"], ),        
            },
        }
 
    RETURN_TYPES = ("YOLOWORLDMODEL",)
    RETURN_NAMES = ("yolo_world_model",)
 
    FUNCTION = "load_yolo_world_model"
  
    CATEGORY = "YoloWorldEfficientSAM"

    def load_yolo_world_model(self, yolo_world_model):
        YOLO_WORLD_MODEL = YOLOWorld(model_id=yolo_world_model)

        return [YOLO_WORLD_MODEL]

class EfficientSAMModelLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["CUDA", "CPU"], ),       
            },
        }
 
    RETURN_TYPES = ("EFFICIENTSAMMODEL",)
    RETURN_NAMES = ("efficient_sam_model",)
 
    FUNCTION = "load_efficient_sam_model"
  
    CATEGORY = "YoloWorldEfficientSAM"

    def load_efficient_sam_model(self, device):
            if device == "CUDA":
                model_path = os.path.join(CURRENT_DIRECTORY, "models", "efficient_sam_s_gpu.jit")
            else:
                model_path = os.path.join(CURRENT_DIRECTORY, "models", "efficient_sam_s_cpu.jit")
                
            EFFICIENT_SAM_MODEL = torch.jit.load(model_path)

            return [EFFICIENT_SAM_MODEL]
class YoloWorldEfficientSAM:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), 
                "yolo_world_model": ("YOLOWORLDMODEL",),
                "efficient_sam_model": ("EFFICIENTSAMMODEL",),  
                "categories": ("STRING", {"default": "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat", "multiline": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.03, "min": 0, "max": 1, "step":0.001}),
                "iou_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step":0.001}),
                "box_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_scale": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step":0.01}),
                "with_confidence": ("BOOLEAN", {"default": True}),
                "with_class_agnostic_nms": ("BOOLEAN", {"default": False}),
                "with_segmentation": ("BOOLEAN", {"default": True}),
                "mask_combined": ("BOOLEAN", {"default": True}),
                "mask_extracted": ("BOOLEAN", {"default": True}),
                "mask_extracted_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            },
        }
 
    RETURN_TYPES = ("IMAGE", "MASK", )
    # RETURN_NAMES = ("yoloworld_efficientsam_image",)
 
    FUNCTION = "yoloworld_efficientsam_image"
  
    CATEGORY = "YoloWorldEfficientSAM"
    def yoloworld_efficientsam_image(self, image, yolo_world_model, efficient_sam_model, categories, confidence_threshold, iou_threshold, box_thickness, text_thickness, text_scale, with_segmentation, mask_combined, with_confidence, with_class_agnostic_nms, mask_extracted, mask_extracted_index):
        categories = process_categories(categories)
        processed_images = []
        processed_masks = []
        for img in image:
            img = np.clip(255. * img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)  
            YOLO_WORLD_MODEL = yolo_world_model
            YOLO_WORLD_MODEL.set_classes(categories)
            # results = YOLO_WORLD_MODEL.infer(img,text=categories, confidence=confidence_threshold,iou_threshold=iou_threshold,class_agnostic_nms=with_class_agnostic_nms)
            results = YOLO_WORLD_MODEL.infer(img, confidence=confidence_threshold)
            detections = sv.Detections.from_inference(results)
            detections = detections.with_nms(
                class_agnostic=with_class_agnostic_nms,
                threshold=iou_threshold
            )
            combined_mask = None
            if with_segmentation:
                detections.mask = inference_with_boxes(
                    image=img,
                    xyxy=detections.xyxy,
                    model=efficient_sam_model,
                    device=DEVICE
                )
                if mask_combined:
                    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    det_mask = detections.mask
                    for mask in det_mask:
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                    masks_tensor = torch.tensor(combined_mask, dtype=torch.float32)
                    processed_masks.append(masks_tensor) 
                else:
                    det_mask = detections.mask
                    
                    if mask_extracted:
                        mask_index = mask_extracted_index
                        selected_mask = det_mask[mask_index]
                        masks_tensor = torch.tensor(selected_mask, dtype=torch.float32)
                    else:
                        masks_tensor = torch.tensor(det_mask, dtype=torch.float32)
                        
                    processed_masks.append(masks_tensor)

            output_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            output_image = annotate_image(
                input_image=output_image,
                detections=detections,
                categories=categories,
                with_confidence=with_confidence,
                thickness=box_thickness,
                text_thickness=text_thickness,
                text_scale=text_scale,
            )
            
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            output_image = torch.from_numpy(output_image.astype(np.float32) / 255.0).unsqueeze(0)
            
            processed_images.append(output_image)

        new_ims = torch.cat(processed_images, dim=0)
        
        if processed_masks:
            new_masks = torch.stack(processed_masks, dim=0)
            # if new_masks.numel() == 0:
            #     new_masks = torch.empty(0)
        else:
            new_ims = torch.empty(0)
        return new_ims,new_masks
    
def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]

def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False,
    thickness: int = 2,
    text_thickness: int = 2,
    text_scale: float = 1.0,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=thickness)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=text_thickness, text_scale=text_scale)
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image

NODE_CLASS_MAPPINGS = {
    "YCYY_YoloWorldModelLoader": YoloWorldModelLoader,
    "YCYY_EfficientSAMModelLoader": EfficientSAMModelLoader,
    "YCYY_YoloWorldEfficientSAM": YoloWorldEfficientSAM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCYY_YoloWorldModelLoader": "Load Yolo World Model",
    "YCYY_EfficientSAMModelLoader": "Load EfficientSAM Model",
    "YCYY_YoloWorldEfficientSAM": "Yolo World EfficientSAM"
}