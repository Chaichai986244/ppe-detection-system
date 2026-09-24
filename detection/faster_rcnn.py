"""
Faster R-CNN Inference Wrapper (torchvision).
Compatible interface with DetectionEngine's _process_frame: infer(image) -> boxes_data list.
"""
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms
import logging

logger = logging.getLogger(__name__)


class FasterRCNNInference:
    """Loads a Faster R-CNN checkpoint and runs inference."""

    def __init__(self, model_path, device='cpu', class_names=None, conf_threshold=0.30):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.class_names = class_names or ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
        self.model = None

    def load_model(self):
        """Build model matching checkpoint architecture and load weights."""
        logger.info(f"Loading Faster R-CNN from {self.model_path}")

        # Load checkpoint to determine architecture
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Determine num_classes from cls_score weight shape: [num_classes+1, in_features]
        cls_weight = state_dict.get('roi_heads.box_predictor.cls_score.weight')
        if cls_weight is not None:
            num_classes = cls_weight.shape[0] - 1  # exclude background
            logger.info(f"Detected num_classes={num_classes} from checkpoint")
        else:
            num_classes = len(self.class_names)
            logger.warning(f"Cannot detect num_classes, using {num_classes}")

        # Build model with ResNet-50 FPN backbone
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        backbone = resnet_fpn_backbone('resnet50', weights=None)

        # Faster R-CNN with matching architecture
        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,), (512,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5,
            ),
        )

        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model
        logger.info(f"Faster R-CNN loaded: {num_classes} classes, device={self.device}")

    def infer(self, image_bgr):
        """Run inference on a BGR image. Returns boxes_data list."""
        if self.model is None:
            self.load_model()

        # Preprocess: BGR -> RGB, HWC -> CHW, normalize
        image_rgb = image_bgr[:, :, ::-1]
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        # Mean/std normalization (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        # Filter by confidence
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # NMS
        if len(boxes) > 0:
            keep = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.45)
            boxes = boxes[keep.numpy()]
            scores = scores[keep.numpy()]
            labels = labels[keep.numpy()]

        # Build boxes_data list
        boxes_data = []
        for box, score, label in zip(boxes, scores, labels):
            cls_idx = int(label) - 1  # Faster R-CNN labels are 1-indexed (0=background)
            cls_name = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else f'class_{cls_idx}'
            boxes_data.append({
                'class_idx': cls_idx,
                'class_name': cls_name,
                'confidence': float(score),
                'xyxy': [int(v) for v in box],
            })

        return boxes_data

    def get_class_names(self):
        return self.class_names
