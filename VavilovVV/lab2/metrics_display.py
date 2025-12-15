import cv2

def _draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def _put_text_with_shadow(img, text, org, font, font_scale, color, thickness, shadow_offset=(1, 1), shadow_color=(0, 0, 0)):
    x, y = org
    dx, dy = shadow_offset
    cv2.putText(img, text, (x + dx, y + dy), font, font_scale, shadow_color, thickness + 1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

def display_metrics(
    image,
    tpr: float,
    fdr: float,
    frame_tpr: float | None = None,
    frame_fdr: float | None = None,
    position: str = "bottom-left",
    margin: int = 20,
    bg_alpha: float = 0.6,
):
    result = image.copy()
    h, w = image.shape[:2]

    lines = ["Metrics:"]
    lines.append(f"TPR: {tpr:.3f}")
    lines.append(f"FDR: {fdr:.3f}")
    if frame_tpr is not None and frame_fdr is not None:
        lines.append(f"Frame TPR: {frame_tpr:.3f}")
        lines.append(f"Frame FDR: {frame_fdr:.3f}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thickness = 1
    line_height = 28
    text_width = max(cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines)
    box_width = text_width + 30
    box_height = len(lines) * line_height + 15

    if position == "top-right":
        x1, y1 = w - box_width - margin, margin
    elif position == "top-left":
        x1, y1 = margin, margin
    elif position == "bottom-right":
        x1, y1 = w - box_width - margin, h - box_height - margin
    elif position == "bottom-left":
        x1, y1 = margin, h - box_height - margin
    else:
        raise ValueError("position must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")
    x2, y2 = x1 + box_width, y1 + box_height

    overlay = result.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 30), -1)
    cv2.addWeighted(overlay, bg_alpha, result, 1 - bg_alpha, 0, result)

    _draw_rounded_rectangle(result, (x1, y1), (x2, y2), (80, 160, 255), 1, radius=8)

    for i, line in enumerate(lines):
        if i == 0:
            _put_text_with_shadow(
                result, line,
                (x1 + 15, y1 + 25 + i * line_height),
                font, font_scale + 0.1, (220, 240, 255), 2
            )
        else:
            _put_text_with_shadow(
                result, line,
                (x1 + 15, y1 + 25 + i * line_height),
                font, font_scale, (240, 255, 240), font_thickness
            )
    return result