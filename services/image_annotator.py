"""
Image Annotator Service
Draws bounding boxes on document images to highlight low confidence fields.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


# Confidence thresholds
POOR_CONFIDENCE_THRESHOLD = 0.70   # Below this = red box
MODERATE_CONFIDENCE_THRESHOLD = 0.85  # Below this = orange box


def get_box_color(confidence: float) -> Tuple[int, int, int, int]:
    """
    Determine box color based on confidence score.
    
    Returns:
        RGBA tuple for the box color
    """
    if confidence < POOR_CONFIDENCE_THRESHOLD:
        return (255, 50, 50, 200)  # Red for poor confidence
    elif confidence < MODERATE_CONFIDENCE_THRESHOLD:
        return (255, 165, 0, 200)  # Orange for moderate confidence
    else:
        return (50, 205, 50, 150)  # Green for high confidence (optional)


def annotate_document_image(
    image_path: str,
    low_confidence_fields: List[Dict],
    output_path: Optional[str] = None,
    show_labels: bool = True
) -> str:
    """
    Draw bounding boxes on document image for low confidence fields.
    
    Args:
        image_path: Path to the original document image
        low_confidence_fields: List of dicts with keys:
            - field_name: Name of the field
            - confidence: Confidence score (0-1)
            - bounding_boxes: List of {page, l, t, r, b} normalized coords
        output_path: Optional output path. If None, creates _annotated suffix
        show_labels: Whether to show field labels on boxes
    
    Returns:
        Path to the annotated image
    """
    try:
        # Open the original image
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        
        draw = ImageDraw.Draw(img)
        
        # Try to load a font for labels
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()
            small_font = font
        
        # Draw boxes for each low confidence field
        annotations_drawn = 0
        for field in low_confidence_fields:
            field_name = field.get("field_name", "Unknown")
            confidence = field.get("confidence", 0.0)
            bboxes = field.get("bounding_boxes", [])
            
            # Skip if confidence is high (Green) - we only want Red or Orange
            if confidence >= MODERATE_CONFIDENCE_THRESHOLD:
                continue

            # Get color based on confidence
            if confidence < POOR_CONFIDENCE_THRESHOLD:
                box_color = (220, 53, 69)  # Red for poor confidence
                fill_color = (220, 53, 69)
            else:
                box_color = (255, 165, 0)  # Orange for moderate confidence
                fill_color = (255, 165, 0)
            
            for bbox in bboxes:
                # Skip if no valid coordinates
                if not all(k in bbox for k in ["l", "t", "r", "b"]):
                    continue
                
                # Convert normalized coordinates to pixels
                x1 = int(bbox["l"] * img_width)
                y1 = int(bbox["t"] * img_height)
                x2 = int(bbox["r"] * img_width)
                y2 = int(bbox["b"] * img_height)
                
                # Draw thick rectangle border
                border_width = 4
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=border_width)
                
                # Add label if requested
                if show_labels:
                    label = f"{field_name.replace('_', ' ').title()}: {confidence:.0%}"
                    
                    # Calculate label position
                    label_bbox = draw.textbbox((0, 0), label, font=small_font)
                    label_width = label_bbox[2] - label_bbox[0]
                    label_height = label_bbox[3] - label_bbox[1]
                    
                    label_x = x1
                    label_y = max(0, y1 - label_height - 6)
                    
                    # Draw label background
                    draw.rectangle(
                        [label_x, label_y, label_x + label_width + 8, label_y + label_height + 4],
                        fill=box_color
                    )
                    # Draw label text
                    draw.text(
                        (label_x + 4, label_y + 2),
                        label,
                        font=small_font,
                        fill=(255, 255, 255)
                    )
                
                annotations_drawn += 1
        
        # Determine output path
        if output_path is None:
            base_path = Path(image_path)
            output_path = str(base_path.parent / f"{base_path.stem}_annotated{base_path.suffix}")
        
        # Save the annotated image
        img.save(output_path)
        logger.info(f"Annotated image saved: {output_path} ({annotations_drawn} boxes drawn)")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error annotating image: {e}")
        import traceback
        traceback.print_exc()
        return image_path  # Return original if annotation fails


def get_low_confidence_fields(
    extracted_fields: Dict,
    threshold: float = MODERATE_CONFIDENCE_THRESHOLD
) -> List[Dict]:
    """
    Filter extracted fields to only those below the confidence threshold.
    
    Args:
        extracted_fields: Dict of field_name -> ExtractedField or dict
        threshold: Confidence threshold for filtering
    
    Returns:
        List of dicts with field_name, confidence, bounding_boxes
    """
    low_confidence = []
    
    for field_name, field_data in extracted_fields.items():
        # Handle both ExtractedField objects and plain dicts
        if hasattr(field_data, 'confidence'):
            confidence = field_data.confidence
            bboxes = getattr(field_data, 'bounding_boxes', [])
        else:
            confidence = field_data.get("confidence", 1.0)
            bboxes = field_data.get("bounding_boxes", [])
        
        if confidence < threshold:
            low_confidence.append({
                "field_name": field_name,
                "confidence": confidence,
                "bounding_boxes": bboxes
            })
    
    return low_confidence
