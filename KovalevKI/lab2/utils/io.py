from pathlib import Path
from collections import defaultdict

def load_annotations_custom(labels_txt_path):
    ann_by_id = defaultdict(list)
    with open(labels_txt_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            frame_id = parts[0].zfill(6)
            label = parts[1].lower()
            try:
                x1, y1, x2, y2 = map(float, parts[2:6])
            except ValueError:
                continue
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            ann_by_id[frame_id].append((label, x, y, w, h))
    return dict(ann_by_id)