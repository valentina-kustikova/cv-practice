"""
GUI module based on PyQt
"""
import sys
import tempfile
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QSpinBox, QComboBox, QGroupBox, QMessageBox,
                             QCheckBox)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QDesktopServices

from .filters import ImageFilter


class ImageViewer(QLabel):
    """Widget for displaying images"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Выберите изображение для обработки")
        self.setStyleSheet("border: 1px solid gray")
        self._original_image: Optional[np.ndarray] = None
        self._display_size: Optional[Tuple[int, int]] = None
        self._click_callback: Optional[Callable[[int, int], None]] = None

    def set_image(self, image: np.ndarray):
        """Sets image for display"""
        self._original_image = image.copy()
        self._display_image(image)

    def _display_image(self, image: np.ndarray):
        """Displays image in widget"""
        if image is None:
            return

        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        height, width = rgb_image.shape[:2]
        bytes_per_line = 3 * width if len(rgb_image.shape) == 3 else width

        if len(rgb_image.shape) == 2:
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        self._display_size = (scaled_pixmap.width(), scaled_pixmap.height())

    def get_original_image(self) -> Optional[np.ndarray]:
        """Returns original image"""
        return self._original_image

    def update_image(self, image: np.ndarray):
        """Updates displayed image"""
        self._display_image(image)

    def set_click_callback(self, callback: Optional[Callable[[int, int], None]]):
        """Sets callback for mouse click events on the image"""
        self._click_callback = callback

    def mousePressEvent(self, event):
        if (
            self._click_callback is not None
            and self._original_image is not None
            and self._display_size is not None
            and event.button() == Qt.LeftButton
        ):
            display_width, display_height = self._display_size
            if display_width == 0 or display_height == 0:
                return

            offset_x = (self.width() - display_width) / 2
            offset_y = (self.height() - display_height) / 2

            x = event.pos().x()
            y = event.pos().y()

            if offset_x <= x <= offset_x + display_width and offset_y <= y <= offset_y + display_height:
                relative_x = (x - offset_x) / display_width
                relative_y = (y - offset_y) / display_height

                relative_x = np.clip(relative_x, 0.0, 1.0)
                relative_y = np.clip(relative_y, 0.0, 1.0)

                image_height, image_width = self._original_image.shape[:2]
                source_x = int(round(relative_x * (image_width - 1)))
                source_y = int(round(relative_y * (image_height - 1)))

                self._click_callback(source_x, source_y)
                return

        super().mousePressEvent(event)

class HistogramWidget(QWidget):
    """Simple widget that draws histogram bars using Qt painting"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMinimumWidth(255*2)
        self._hist_data: Optional[np.ndarray] = None
        self._is_color = False
        self._channel_visibility = [True, True, True]

    def update_histogram(self, image: Optional[np.ndarray]):
        if image is None:
            self._hist_data = None
            self._is_color = False
            self._channel_visibility = [True, True, True]
        else:
            if image.ndim == 2:
                hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 255))
                self._hist_data = hist.astype(np.float32)
                self._is_color = False
                self._channel_visibility = [True, True, True]
            else:
                channels = cv2.split(image)
                self._hist_data = np.array([
                    np.histogram(channel.ravel(), bins=256, range=(0, 255))[0].astype(np.float32)
                    for channel in channels
                ])
                self._is_color = True
                if len(self._channel_visibility) != len(self._hist_data):
                    self._channel_visibility = [True] * len(self._hist_data)

            if self._is_color:
                max_value = self._hist_data.max()
            else:
                max_value = self._hist_data.max() if self._hist_data is not None else 1.0

            if max_value > 0:
                self._hist_data = self._hist_data / max_value

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111"))

        if self._hist_data is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Нет данных")
            painter.end()
            return

        padding_left = 20
        padding_right = 20
        padding_bottom = 24

        width = max(1, self.width() - padding_left - padding_right)
        height = self.height()
        plot_height = max(1, height - padding_bottom)
        bar_width = max(1, width // 256)

        # Draw tick marks for value intervals
        painter.setPen(QColor(255, 255, 255, 120))
        tick_count = 2
        for i in range(tick_count + 1):
            x = padding_left + int(i * width / tick_count)
            painter.drawLine(x, plot_height + 4, x, height - 2)
            value = int(i * 255 / tick_count)
            painter.drawText(x - 14, height - 4, f"{value}")

        if self._is_color and self._hist_data is not None and self._hist_data.ndim == 2:
            colors = [QColor(255, 0, 0, 120), QColor(0, 255, 0, 120), QColor(0, 0, 255, 120)]
            for channel_index, channel_hist in enumerate(self._hist_data):
                if channel_index < len(self._channel_visibility) and not self._channel_visibility[channel_index]:
                    continue
                painter.setBrush(colors[channel_index])
                painter.setPen(Qt.NoPen)
                for i in range(256):
                    value = channel_hist[i]
                    bar_height = int(value * plot_height)
                    painter.drawRect(padding_left + i * bar_width, plot_height - bar_height, bar_width, bar_height)
        else:
            painter.setBrush(QColor(180, 180, 180))
            painter.setPen(Qt.NoPen)
            for i in range(256):
                value = self._hist_data[i]
                bar_height = int(value * plot_height)
                painter.drawRect(padding_left + i * bar_width, plot_height - bar_height, bar_width, bar_height)

        painter.end()

    def set_channel_visibility(self, channel: int, visible: bool):
        if 0 <= channel < len(self._channel_visibility):
            self._channel_visibility[channel] = bool(visible)
            self.update()

    def is_color(self) -> bool:
        return self._is_color


class FilterWindow(QMainWindow):
    """Main application window"""

    FILTER_NAMES = [
        "Nearest Neighbour Interpolation",
        "Bilinear interpolation",
        "Linear resize",
        "sepia",
        "vignette",
        "pixelation",
        "frame_simple",
        "frame_curvy",
        "glare",
        "watercolor_texture",
        "overlay_alpha"
    ]

    def __init__(self):
        super().__init__()
        self.current_filter = None
        self.filter_params = {}
        self.param_widgets: Dict[str, Tuple[Optional[QSlider], Optional[QSpinBox]]] = {}
        self.rgb_previews: Dict[str, QLabel] = {}
        self.original_image_path: Optional[Path] = None
        self.processed_temp_path: Optional[Path] = None
        self.vignette_pick_button: Optional[QPushButton] = None
        self.vignette_center_label: Optional[QLabel] = None
        self.pixelation_pick_button: Optional[QPushButton] = None
        self.pixelation_region_label: Optional[QLabel] = None
        self.glare_pick_button: Optional[QPushButton] = None
        self.glare_center_label: Optional[QLabel] = None
        self._pick_mode: Optional[str] = None
        self._pending_pixelation_point: Optional[Tuple[int, int]] = None
        self.overlay_image: Optional[np.ndarray] = None
        self.overlay_path: Optional[Path] = None
        self.overlay_button: Optional[QPushButton] = None
        self.overlay_preview: Optional[QLabel] = None
        self.use_overlay_for_frame: bool = False
        self.frame_overlay_checkbox: Optional[QCheckBox] = None
        self.original_histogram_checkboxes: List[QCheckBox] = []
        self.processed_histogram_checkboxes: List[QCheckBox] = []
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("lab1 - base matrix filters")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        image_layout = QVBoxLayout()

        original_label = QLabel("Исходное изображение")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setStyleSheet("font-size: 18px; font-weight: bold")
        self.original_viewer = ImageViewer()
        self.original_info_label = QLabel("Размер: —")
        self.original_info_label.setStyleSheet("font-size: 14px;")
        self.original_info_label.setAlignment(Qt.AlignLeft)
        self.original_histogram = HistogramWidget()

        processed_label = QLabel("Примененный фильтр")
        processed_label.setAlignment(Qt.AlignCenter)
        processed_label.setStyleSheet("font-size: 18px; font-weight: bold")
        self.processed_viewer = ImageViewer()
        self.processed_info_label = QLabel("Размер: —")
        self.processed_info_label.setStyleSheet("font-size: 14px;")
        self.processed_info_label.setAlignment(Qt.AlignLeft)
        self.processed_histogram = HistogramWidget()

        image_layout.addWidget(original_label)
        image_layout.addWidget(self.original_viewer)
        image_layout.addWidget(self.original_info_label)
        image_layout.addWidget(self.original_histogram)
        self.original_histogram_checkboxes = self._add_histogram_channel_controls(
            image_layout,
            self.original_histogram,
            "Каналы исходного изображения:"
        )
        image_layout.addWidget(processed_label)
        image_layout.addWidget(self.processed_viewer)
        image_layout.addWidget(self.processed_info_label)
        image_layout.addWidget(self.processed_histogram)
        self.processed_histogram_checkboxes = self._add_histogram_channel_controls(
            image_layout,
            self.processed_histogram,
            "Каналы результата:"
        )

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        file_group = QGroupBox("Files")
        file_layout = QVBoxLayout()

        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image)

        self.save_button = QPushButton("Сохранить измененное изображение")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)

        self.open_original_button = QPushButton("Открыть исходное изображение(внешнее прил.)")
        self.open_original_button.clicked.connect(self.open_original_external)
        self.open_original_button.setEnabled(False)

        self.open_processed_button = QPushButton("Открыть измененное изображение (внешнее прил.)")
        self.open_processed_button.clicked.connect(self.open_processed_external)
        self.open_processed_button.setEnabled(False)

        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.save_button)
        file_layout.addWidget(self.open_original_button)
        file_layout.addWidget(self.open_processed_button)
        file_group.setLayout(file_layout)

        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "Nearest Neighbour Interpolation",
            "Bilinear interpolation",
            "Linear resize",
            "Sepia",
            "Vignette",
            "Pixelation of rect area",
            "Frame (simple)",
            "Frame (curvy)",
            "Glare",
            "Watercolor paper texture",
            "Overlay (alpha)",
        ])
        self.filter_combo.currentIndexChanged.connect(self.on_filter_changed)

        filter_layout.addWidget(QLabel("Select filter:"))
        filter_layout.addWidget(self.filter_combo)

        self.params_group = QGroupBox("Filter Parameters")
        self.params_layout = QVBoxLayout()
        self.params_group.setLayout(self.params_layout)
        self.params_group.setVisible(False)

        filter_layout.addWidget(self.params_group)
        filter_group.setLayout(filter_layout)

        self.apply_button = QPushButton("Apply Filter")
        self.apply_button.clicked.connect(self.apply_filter)
        self.apply_button.setEnabled(False)

        control_layout.addWidget(file_group)
        control_layout.addWidget(filter_group)
        control_layout.addWidget(self.apply_button)

        self.overlay_button = QPushButton("Load Overlay")
        self.overlay_button.clicked.connect(self.load_overlay_image)
        self.overlay_preview = QLabel("Нет\nоверлея")
        self.overlay_preview.setAlignment(Qt.AlignCenter)
        self.overlay_preview.setFixedSize(140, 140)
        self.overlay_preview.setStyleSheet(
            "border: 1px solid gray; background-color: #222; color: #ccc;"
        )

        control_layout.addStretch()

        overlay_layout = QVBoxLayout()
        overlay_layout.setSpacing(6)
        overlay_layout.addWidget(self.overlay_button, alignment=Qt.AlignRight)
        overlay_layout.addWidget(self.overlay_preview, alignment=Qt.AlignRight)
        control_layout.addLayout(overlay_layout)

        main_layout.addLayout(image_layout, 2)
        main_layout.addWidget(control_panel, 1)

        # Initialize parameters for the default selection
        self.on_filter_changed(self.filter_combo.currentIndex())
        self._update_histograms()
        self._update_overlay_preview()
        self._update_overlay_visibility()
        self._update_overlay_labels()

    def load_image(self):
        """Loads image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )

        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.original_viewer.set_image(image)
                self.processed_viewer.set_image(image)
                self.processed_viewer._original_image = image.copy()
                self._set_pick_mode(None)
                self._update_image_info()
                if self.current_filter == "Linear resize":
                    self._update_resize_defaults(force=True)
                self.apply_button.setEnabled(True)
                self.save_button.setEnabled(False)
                self.original_image_path = Path(file_path)
                self.open_original_button.setEnabled(True)
                self.processed_temp_path = None
                self.open_processed_button.setEnabled(False)
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")

    def save_image(self):
        """Saves processed image"""
        image = self.processed_viewer.get_original_image()
        if image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )

        if file_path:
            cv2.imwrite(file_path, image)
            QMessageBox.information(self, "Success", "Image saved")
            self.processed_temp_path = Path(file_path)
            self.open_processed_button.setEnabled(True)

    def load_overlay_image(self):
        """Loads external overlay image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Overlay Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )

        if not file_path:
            return

        overlay = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            QMessageBox.warning(self, "Error", "Failed to load overlay image")
            return

        self.overlay_image = overlay
        self.overlay_path = Path(file_path)
        self._update_overlay_preview()

    def _update_overlay_preview(self):
        """Updates overlay preview thumbnail"""
        if self.overlay_preview is None:
            return

        if not self._is_overlay_allowed():
            self.overlay_preview.clear()
            self.overlay_preview.setText(self._get_overlay_placeholder_text())
            self.overlay_preview.setToolTip("")
            return

        if self.overlay_image is None:
            self.overlay_preview.setPixmap(QPixmap())
            self.overlay_preview.setText(self._get_overlay_placeholder_text())
            self.overlay_preview.setToolTip("")
            self._update_overlay_labels()
            return

        image = self.overlay_image
        height, width = image.shape[:2]

        if image.ndim == 2:
            q_image = QImage(
                image.data,
                width,
                height,
                image.strides[0],
                QImage.Format_Grayscale8
            )
        elif image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(
                rgb_image.data,
                width,
                height,
                rgb_image.strides[0],
                QImage.Format_RGB888
            )
        else:
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            q_image = QImage(
                rgba_image.data,
                width,
                height,
                rgba_image.strides[0],
                QImage.Format_RGBA8888
            )

        pixmap = QPixmap.fromImage(q_image.copy())
        scaled = pixmap.scaled(self.overlay_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.overlay_preview.setPixmap(scaled)
        self.overlay_preview.setText("")
        self.overlay_preview.setToolTip(str(self.overlay_path) if self.overlay_path else "")
        self._update_overlay_labels()

    def _update_overlay_labels(self):
        """Updates button caption and placeholder text depending on context"""
        if self.overlay_button is None or self.overlay_preview is None:
            return
        if not self._is_overlay_allowed():
            return

        if self.current_filter == "frame_curvy":
            self.overlay_button.setText("Load Frame")
        elif self.current_filter == "glare":
            self.overlay_button.setText("Load Glare")
        else:
            self.overlay_button.setText("Load Overlay")

        if self.overlay_image is None:
            self.overlay_preview.setText(self._get_overlay_placeholder_text())
            self.overlay_preview.setToolTip("")

    def _get_overlay_placeholder_text(self) -> str:
        """Returns context-specific placeholder for overlay preview"""
        if self.current_filter == "frame_curvy":
            return "Нет\nрамки"
        elif self.current_filter == "glare":
            return  "Нет\n блика"
        else:
            return "Нет\nоверлея"

    def on_filter_changed(self, index: int):
        """Handles filter selection change"""
        self.clear_params()

        if 0 <= index < len(self.FILTER_NAMES):
            self.current_filter = self.FILTER_NAMES[index]
            self.setup_filter_params()
            self._update_overlay_visibility()
            self._update_overlay_labels()
            self._update_overlay_preview()

    def clear_params(self):
        """Clears parameters panel"""
        def _clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.deleteLater()
                elif child_layout is not None:
                    _clear_layout(child_layout)
                    child_layout.deleteLater()

        _clear_layout(self.params_layout)
        self.params_group.setVisible(False)
        self.filter_params.clear()
        self.param_widgets.clear()
        self.rgb_previews.clear()
        self._set_pick_mode(None)
        self.vignette_pick_button = None
        self.vignette_center_label = None
        self.pixelation_pick_button = None
        self.pixelation_region_label = None
        self.glare_pick_button = None
        self.glare_center_label = None
        self._pending_pixelation_point = None
        self.use_overlay_for_frame = False
        self.frame_overlay_checkbox = None

    def setup_filter_params(self):
        """Sets up parameters for selected filter"""
        self.params_group.setVisible(True)

        if self.current_filter == "Nearest Neighbour Interpolation":
            self.add_slider_param("scale_factor", "Scale", 1, 5, 2)

        elif self.current_filter == "Bilinear interpolation":
            self.add_slider_param("scale_factor", "Scale:", 1, 5, 2)

        elif self.current_filter == "Linear resize":
            self.add_spinbox_param("new_width", "Width (px):", 1, 10000, 512)
            self.add_spinbox_param("new_height", "Height (px):", 1, 10000, 512)
            self._update_resize_defaults(force=True)

        elif self.current_filter == "vignette":
            self.add_slider_param("intensity", "Intensity:", 0, 100, 70)
            self.add_slider_param("radius", "Radius:", 0, 100, 50)
            center_layout = QHBoxLayout()
            self.vignette_pick_button = QPushButton("Pick center")
            self.vignette_pick_button.clicked.connect(self.activate_vignette_center_pick)
            self.vignette_center_label = QLabel("Center: n/a")
            center_layout.addWidget(self.vignette_pick_button)
            center_layout.addWidget(self.vignette_center_label)
            center_layout.addStretch()
            self.params_layout.addLayout(center_layout)
            self._update_vignette_center_defaults()

        elif self.current_filter == "pixelation":
            self.add_slider_param("x", "X:", 0, 1000, 0)
            self.add_slider_param("y", "Y:", 0, 1000, 0)
            self.add_slider_param("width", "Width:", 10, 500, 100)
            self.add_slider_param("height", "Height:", 10, 500, 100)
            self.add_slider_param("pixel_size", "Pixel size:", 5, 50, 10)
            region_layout = QHBoxLayout()
            self.pixelation_pick_button = QPushButton("Pick area")
            self.pixelation_pick_button.clicked.connect(self.activate_pixelation_pick)
            self.pixelation_region_label = QLabel("Area: n/a")
            region_layout.addWidget(self.pixelation_pick_button)
            region_layout.addWidget(self.pixelation_region_label)
            region_layout.addStretch()
            self.params_layout.addLayout(region_layout)
            self._update_pixelation_region_label()

        elif self.current_filter == "frame_simple":
            self.add_spinbox_param("frame_width", "Thickness (px):", 0, 255, 10)
            self.add_rgb_param("frame_color", "Color:", (0, 0, 0))

        elif self.current_filter == "frame_curvy":
            self.add_spinbox_param("frame_width", "Thickness (px):", 0, 255, 10)
            self.add_rgb_param("frame_color", "Color:", (0, 0, 0))
            self._add_frame_overlay_toggle()

        elif self.current_filter == "glare":
            self.add_slider_param("center_x", "Center X:", 0, 1000, 0)
            self.add_slider_param("center_y", "Center Y:", 0, 1000, 0)
            self.add_slider_param("radius", "Radius:", 10, 500, 100)
            self.add_slider_param("intensity", "Intensity:", 0, 100, 50)
            self.add_slider_param("overlay_scale", "Texture scale (%):", 10, 200, 100)
            self.filter_params["center_x"] = None
            self.filter_params["center_y"] = None
            center_layout = QHBoxLayout()
            self.glare_pick_button = QPushButton("Pick center")
            self.glare_pick_button.clicked.connect(self.activate_glare_center_pick)
            self.glare_center_label = QLabel("Center: n/a")
            center_layout.addWidget(self.glare_pick_button)
            center_layout.addWidget(self.glare_center_label)
            center_layout.addStretch()
            self.params_layout.addLayout(center_layout)
            self._update_glare_center_label()

        elif self.current_filter == "watercolor_texture":
            self.add_slider_param("texture_intensity", "Intensity:", 0, 100, 30)

        elif self.current_filter == "overlay_alpha":
            info_label = QLabel(
                "Наложение изображения с учётом альфа-канала. "
                "Загрузите изображение кнопкой в правом нижнем углу."
            )
            info_label.setWordWrap(True)
            self.params_layout.addWidget(info_label)

    def add_slider_param(self, param_name: str, label: str,
                        min_val: int, max_val: int, default: int, odd_only: bool = False):
        """Adds slider parameter"""
        param_layout = QHBoxLayout()

        label_widget = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)

        spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default)

        if odd_only:
            slider.valueChanged.connect(lambda v: slider.setValue(v if v % 2 == 1 else v + 1))
            spinbox.valueChanged.connect(lambda v: spinbox.setValue(v if v % 2 == 1 else v + 1))

        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(lambda v: self.update_param(param_name, v))

        param_layout.addWidget(label_widget)
        param_layout.addWidget(slider)
        param_layout.addWidget(spinbox)

        self.params_layout.addLayout(param_layout)
        self.filter_params[param_name] = default
        self.param_widgets[param_name] = (slider, spinbox)

    def add_spinbox_param(self, param_name: str, label: str,
                          min_val: int, max_val: int, default: int):
        """Adds parameter controlled by a single spinbox"""
        param_layout = QHBoxLayout()

        label_widget = QLabel(label)
        spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default)
        spinbox.valueChanged.connect(lambda v: self.update_param(param_name, v))

        param_layout.addWidget(label_widget)
        param_layout.addWidget(spinbox)
        param_layout.addStretch()

        self.params_layout.addLayout(param_layout)
        self.filter_params[param_name] = default
        self.param_widgets[param_name] = (None, spinbox)

    def add_rgb_param(self, base_name: str, label: str, default: Tuple[int, int, int]):
        """Adds RGB spinboxes with preview"""
        rgb_layout = QHBoxLayout()

        label_widget = QLabel(label)
        rgb_layout.addWidget(label_widget)

        channels = ("r", "g", "b")
        preview_label = QLabel()
        preview_label.setFixedSize(28, 28)

        for channel, default_value in zip(channels, default):
            channel_layout = QVBoxLayout()
            channel_label = QLabel(channel.upper())
            channel_label.setAlignment(Qt.AlignCenter)

            spinbox = QSpinBox()
            spinbox.setMinimum(0)
            spinbox.setMaximum(255)
            spinbox.setValue(int(default_value) % 256)
            param_name = f"{base_name}_{channel}"
            spinbox.valueChanged.connect(lambda v, name=param_name: self.update_param(name, v))

            channel_layout.addWidget(channel_label)
            channel_layout.addWidget(spinbox)
            rgb_layout.addLayout(channel_layout)

            self.filter_params[param_name] = int(default_value) % 256
            self.param_widgets[param_name] = (None, spinbox)

        rgb_layout.addWidget(preview_label)
        rgb_layout.addStretch()
        self.params_layout.addLayout(rgb_layout)

        self.rgb_previews[base_name] = preview_label
        self._update_color_preview(base_name)

    def _add_frame_overlay_toggle(self):
        """Adds checkbox to switch between procedural and custom frame"""
        checkbox = QCheckBox("Использовать внешнюю рамку (PNG с альфа-каналом)")
        checkbox.stateChanged.connect(self._toggle_frame_overlay)
        self.params_layout.addWidget(checkbox)
        self.frame_overlay_checkbox = checkbox
        self.use_overlay_for_frame = False
        self._update_overlay_labels()

    def _add_histogram_channel_controls(self,
                                        parent_layout: QVBoxLayout,
                                        histogram_widget: HistogramWidget,
                                        label_text: str) -> List[QCheckBox]:
        """Adds channel visibility controls for histogram"""
        control_layout = QHBoxLayout()
        title_label = QLabel(label_text)
        control_layout.addWidget(title_label)

        checkboxes: List[QCheckBox] = []
        channel_titles = [("R", "#ff4d4d"), ("G", "#4dff4d"), ("B", "#4d4dff")]

        for index, (channel_name, color) in enumerate(channel_titles):
            checkbox = QCheckBox(channel_name)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(f"color: {color}; font-weight: bold;")
            checkbox.toggled.connect(
                lambda checked, widget=histogram_widget, channel=index: self._on_histogram_channel_toggled(widget, channel, checked)
            )
            control_layout.addWidget(checkbox)
            checkboxes.append(checkbox)

        control_layout.addStretch()
        parent_layout.addLayout(control_layout)
        return checkboxes

    def _on_histogram_channel_toggled(self, histogram_widget: HistogramWidget, channel: int, checked: bool):
        """Updates histogram channel visibility when checkbox toggled"""
        histogram_widget.set_channel_visibility(channel, checked)

    def _sync_histogram_controls(self,
                                 histogram_widget: HistogramWidget,
                                 checkboxes: List[QCheckBox]):
        """Enables/disables histogram channel controls based on image mode"""
        if not checkboxes:
            return

        is_color = histogram_widget.is_color()
        for index, checkbox in enumerate(checkboxes):
            checkbox.blockSignals(True)
            checkbox.setEnabled(is_color)
            if not is_color:
                checkbox.setChecked(True)
            current_state = checkbox.isChecked()
            checkbox.blockSignals(False)
            histogram_widget.set_channel_visibility(index, current_state)

    def _is_overlay_allowed(self) -> bool:
        """Returns whether overlay controls should be visible for current filter"""
        return self.current_filter in {"frame_curvy", "glare", "watercolor_texture"}

    def _update_overlay_visibility(self):
        """Shows or hides overlay controls depending on current filter"""
        if self.overlay_button is None or self.overlay_preview is None:
            return

        allowed = self._is_overlay_allowed()
        self.overlay_button.setVisible(allowed)
        self.overlay_preview.setVisible(allowed)

        if not allowed:
            if self.frame_overlay_checkbox is not None:
                self.frame_overlay_checkbox.setVisible(False)
            if self.use_overlay_for_frame:
                self.use_overlay_for_frame = False
                if self.frame_overlay_checkbox is not None:
                    self.frame_overlay_checkbox.blockSignals(True)
                    self.frame_overlay_checkbox.setChecked(False)
                    self.frame_overlay_checkbox.blockSignals(False)
            self.overlay_preview.setText(self._get_overlay_placeholder_text())
            self.overlay_preview.setToolTip("")
        else:
            if self.frame_overlay_checkbox is not None:
                self.frame_overlay_checkbox.setVisible(True)

    def _toggle_frame_overlay(self, state: int):
        """Handles switching between procedural and custom frame"""
        if not self._is_overlay_allowed():
            self.use_overlay_for_frame = False
            if self.frame_overlay_checkbox is not None:
                self.frame_overlay_checkbox.blockSignals(True)
                self.frame_overlay_checkbox.setChecked(False)
                self.frame_overlay_checkbox.blockSignals(False)
            return

        self.use_overlay_for_frame = state == Qt.Checked
        if self.use_overlay_for_frame and self.overlay_image is None:
            QMessageBox.information(
                self,
                "Frame overlay",
                "Загрузите PNG с альфа-каналом, который будет использоваться как рамка."
            )
            if self.frame_overlay_checkbox is not None:
                self.frame_overlay_checkbox.blockSignals(True)
                self.frame_overlay_checkbox.setChecked(False)
                self.frame_overlay_checkbox.blockSignals(False)
            self.use_overlay_for_frame = False
        self._update_overlay_labels()
        self._update_overlay_preview()

    def _set_param_value(self, param_name: str, value: int):
        """Sets parameter value and updates associated widgets without recursion"""
        self.filter_params[param_name] = value
        widgets = self.param_widgets.get(param_name)
        if widgets is None:
            return
        slider, spinbox = widgets
        if slider is not None:
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)
        if spinbox is not None:
            spinbox.blockSignals(True)
            spinbox.setValue(int(value))
            spinbox.blockSignals(False)

    def update_param(self, param_name: str, value: int):
        """Updates parameter value"""
        sanitized_value = int(value)
        update_widget = False

        if param_name in {"new_width", "new_height"}:
            sanitized_value = max(1, sanitized_value)
            update_widget = sanitized_value != value
        elif param_name == "frame_width":
            sanitized_value = max(0, sanitized_value)
            update_widget = sanitized_value != value
        elif param_name.endswith(("_r", "_g", "_b")):
            sanitized_value %= 256
            update_widget = sanitized_value != value

        if update_widget:
            self._set_param_value(param_name, sanitized_value)
        else:
            self.filter_params[param_name] = sanitized_value

        if param_name.endswith(("_r", "_g", "_b")):
            base_name = param_name.rsplit("_", 1)[0]
            self._update_color_preview(base_name)
            return

        if self.current_filter == "pixelation" and param_name in {"x", "y", "width", "height"}:
            self._update_pixelation_region_label()
        elif self.current_filter == "glare" and param_name in {"center_x", "center_y"}:
            self._update_glare_center_label()
        elif self.current_filter == "glare" and param_name == "overlay_scale":
            self.filter_params[param_name] = np.clip(sanitized_value, 10, 200)

    def _update_color_preview(self, base_name: str):
        """Updates the RGB preview square"""
        preview = self.rgb_previews.get(base_name)
        if preview is None:
            return
        r = int(self.filter_params.get(f"{base_name}_r", 0)) % 256
        g = int(self.filter_params.get(f"{base_name}_g", 0)) % 256
        b = int(self.filter_params.get(f"{base_name}_b", 0)) % 256
        preview.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #444;")

    def _update_resize_defaults(self, force: bool = False):
        """Synchronizes resize parameters with current image size when available"""
        if self.current_filter != "Linear resize":
            return
        image = self.original_viewer.get_original_image()
        if image is None:
            return
        height, width = image.shape[:2]
        desired_width = max(1, int(width))
        desired_height = max(1, int(height))
        if "new_width" in self.param_widgets:
            current_width = self.filter_params.get("new_width")
            if force or current_width in (None, 0):
                self._set_param_value("new_width", desired_width)
        if "new_height" in self.param_widgets:
            current_height = self.filter_params.get("new_height")
            if force or current_height in (None, 0):
                self._set_param_value("new_height", desired_height)

    def apply_filter(self):
        """Applies selected filter"""
        original = self.original_viewer.get_original_image()
        if original is None:
            return

        if self.current_filter == "overlay_alpha" and self.overlay_image is None:
            QMessageBox.information(
                self,
                "Overlay",
                "Загрузите внешнее изображение с альфа-каналом, чтобы выполнить наложение."
            )
            return
        if self.current_filter in {"frame_simple", "frame_curvy"} and self.use_overlay_for_frame and self.overlay_image is None:
            QMessageBox.information(
                self,
                "Frame overlay",
                "Для кастомной рамки загрузите PNG с альфа-каналом или отключите внешний источник."
            )
            return
        if self.current_filter == "watercolor_texture" and self.overlay_image is None:
            QMessageBox.information(
                self,
                "Watercolor texture",
                "Для текстуры акварельной бумаги загрузите изображение (PNG с альфа-каналом)."
            )
            return

        try:
            processed = self._apply_filter_to_image(original)
            self.processed_viewer.update_image(processed)
            self.processed_viewer._original_image = processed.copy()
            self._update_image_info()
            self._save_temp_processed(processed)
            self.save_button.setEnabled(True)
            self.open_processed_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying filter: {str(e)}")

    def _apply_filter_to_image(self, image: np.ndarray) -> np.ndarray:
        """Applies filter to image"""
        if self.current_filter == "Nearest Neighbour Interpolation":
            scale = self.filter_params.get("scale_factor", 2)
            return ImageFilter.apply_Nearest_Neighbor_interpolation(image, scale)

        elif self.current_filter == "Bilinear interpolation":
            scale = self.filter_params.get("scale_factor", 2)
            return ImageFilter.apply_Bilinear_interpolation(image, scale)

        elif self.current_filter == "Linear resize":
            new_width = self.filter_params.get("new_width")
            new_height = self.filter_params.get("new_height")
            if new_width is None or new_height is None:
                h, w = image.shape[:2]
                new_width = w
                new_height = h
            return ImageFilter.apply_linear_resize(image, new_height, new_width)

        elif self.current_filter == "sepia":
            return ImageFilter.apply_sepia(image)

        elif self.current_filter == "vignette":
            intensity = self.filter_params.get("intensity", 70) / 100.0
            radius = self.filter_params.get("radius", 50) / 100.0
            center_x = self.filter_params.get("center_x")
            center_y = self.filter_params.get("center_y")
            if center_x is None or center_y is None:
                height, width = image.shape[:2]
                center_x = width // 2
                center_y = height // 2
            return ImageFilter.apply_vignette(image, intensity, radius, center_x, center_y)

        elif self.current_filter == "pixelation":
            return ImageFilter.apply_pixelation(
                image,
                self.filter_params.get("x", 0),
                self.filter_params.get("y", 0),
                self.filter_params.get("width", 100),
                self.filter_params.get("height", 100),
                self.filter_params.get("pixel_size", 10)
            )

        elif self.current_filter == "frame_simple":
            color = (
                int(self.filter_params.get("frame_color_r", 0)) % 256,
                int(self.filter_params.get("frame_color_g", 0)) % 256,
                int(self.filter_params.get("frame_color_b", 0)) % 256,
            )
            return ImageFilter.apply_frame_simple(
                image,
                self.filter_params.get("frame_width", 10),
                color
            )

        elif self.current_filter == "frame_curvy":
            color = (
                int(self.filter_params.get("frame_color_r", 0)) % 256,
                int(self.filter_params.get("frame_color_g", 0)) % 256,
                int(self.filter_params.get("frame_color_b", 0)) % 256,
            )
            if self.use_overlay_for_frame and self.overlay_image is not None:
                return ImageFilter.apply_overlay(image, self.overlay_image)
            return ImageFilter.apply_frame_curvy(
                image,
                self.filter_params.get("frame_width", 10),
                "wave",
                color
            )

        elif self.current_filter == "glare":
            h, w = image.shape[:2]
            center_x = self.filter_params.get("center_x", w // 2)
            center_y = self.filter_params.get("center_y", h // 2)
            radius = self.filter_params.get("radius", 100)
            intensity = self.filter_params.get("intensity", 50) / 100.0
            if self.overlay_image is not None:
                return ImageFilter.apply_overlay_centered(
                    image,
                    self.overlay_image,
                    center_x,
                    center_y,
                    intensity=intensity,
                    scale=self.filter_params.get("overlay_scale", 100) / 100.0
                )
            return ImageFilter.apply_glare(image, center_x, center_y, radius, intensity)

        elif self.current_filter == "watercolor_texture":
            intensity = self.filter_params.get("texture_intensity", 30) / 100.0
            return ImageFilter.apply_watercolor_texture(image, intensity, self.overlay_image)

        elif self.current_filter == "overlay_alpha":
            return ImageFilter.apply_overlay(image, self.overlay_image)

        return image

    def _update_image_info(self):
        """Updates information labels for images"""
        self.original_info_label.setText(self._format_image_info(self.original_viewer.get_original_image()))
        self.processed_info_label.setText(self._format_image_info(self.processed_viewer.get_original_image()))
        self._update_histograms()
        self._update_vignette_center_defaults(update_existing=True)
        self._update_pixelation_region_label()
        self._update_glare_center_label()
        self._update_resize_defaults()

    @staticmethod
    def _format_image_info(image: Optional[np.ndarray]) -> str:
        if image is None:
            return "Размер: —"
        height, width = image.shape[:2]
        return f"Размер: {width}×{height}"

    def _update_histograms(self):
        """Updates histograms for original and processed images"""
        self.original_histogram.update_histogram(self.original_viewer.get_original_image())
        self.processed_histogram.update_histogram(self.processed_viewer.get_original_image())
        self._sync_histogram_controls(self.original_histogram, self.original_histogram_checkboxes)
        self._sync_histogram_controls(self.processed_histogram, self.processed_histogram_checkboxes)

    def activate_vignette_center_pick(self):
        """Enables vignette center picking mode"""
        if self.original_viewer.get_original_image() is None:
            QMessageBox.information(self, "Info", "Load an image before picking the vignette center.")
            return
        self._pending_pixelation_point = None
        self._set_pick_mode("vignette")
        if self.vignette_center_label is not None:
            self.vignette_center_label.setText("Center: select a point...")

    def activate_pixelation_pick(self):
        """Enables rectangular pixelation picking mode"""
        if self.original_viewer.get_original_image() is None:
            QMessageBox.information(self, "Info", "Load an image before picking the region.")
            return
        self._pending_pixelation_point = None
        self._set_pick_mode("pixelation")
        if self.pixelation_region_label is not None:
            self.pixelation_region_label.setText("Area: select top-left...")

    def activate_glare_center_pick(self):
        """Enables glare center picking mode"""
        if self.original_viewer.get_original_image() is None:
            QMessageBox.information(self, "Info", "Load an image before picking the glare center.")
            return
        self._pending_pixelation_point = None
        self._set_pick_mode("glare")
        if self.glare_center_label is not None:
            self.glare_center_label.setText("Center: select a point...")

    def _set_pick_mode(self, mode: Optional[str]):
        """Sets current pick mode"""
        if mode == self._pick_mode:
            return

        if self._pick_mode is not None:
            self.original_viewer.unsetCursor()
            self.original_viewer.set_click_callback(None)
        self._pick_mode = None

        if mode is None:
            self._pending_pixelation_point = None
            return

        if self.original_viewer.get_original_image() is None:
            return

        self._pick_mode = mode
        self.original_viewer.set_click_callback(self._handle_image_click)
        self.original_viewer.setCursor(Qt.CrossCursor)

    def _handle_image_click(self, x: int, y: int):
        """Handles click events on the image for pick modes"""
        if self._pick_mode == "vignette":
            self.filter_params["center_x"] = x
            self.filter_params["center_y"] = y
            self._set_pick_mode(None)
            self._update_vignette_center_defaults(update_existing=True)

        elif self._pick_mode == "glare":
            self._set_param_value("center_x", x)
            self._set_param_value("center_y", y)
            self._set_pick_mode(None)
            self._update_glare_center_label()

        elif self._pick_mode == "pixelation":
            if self._pending_pixelation_point is None:
                self._pending_pixelation_point = (x, y)
                if self.pixelation_region_label is not None:
                    self.pixelation_region_label.setText(f"Area: ({x}, {y}) → …")
            else:
                x0, y0 = self._pending_pixelation_point
                x1, y1 = x, y
                left = min(x0, x1)
                top = min(y0, y1)
                right = max(x0, x1)
                bottom = max(y0, y1)
                width = max(1, right - left + 1)
                height = max(1, bottom - top + 1)
                self._set_param_value("x", left)
                self._set_param_value("y", top)
                self._set_param_value("width", width)
                self._set_param_value("height", height)
                self._pending_pixelation_point = None
                self._set_pick_mode(None)
                self._update_pixelation_region_label()

    def _update_vignette_center_defaults(self, update_existing: bool = False):
        """Sets default vignette center based on current image"""
        if self.current_filter != "vignette":
            return

        image = self.original_viewer.get_original_image()
        if image is None:
            self.filter_params["center_x"] = None
            self.filter_params["center_y"] = None
            if self.vignette_center_label is not None:
                self.vignette_center_label.setText("Center: n/a")
            if self.vignette_pick_button is not None:
                self.vignette_pick_button.setEnabled(False)
            return

        height, width = image.shape[:2]
        if update_existing:
            cx = self.filter_params.get("center_x")
            cy = self.filter_params.get("center_y")
        else:
            cx = width // 2
            cy = height // 2
            self.filter_params["center_x"] = cx
            self.filter_params["center_y"] = cy

        cx = int(np.clip(cx if cx is not None else width // 2, 0, width - 1))
        cy = int(np.clip(cy if cy is not None else height // 2, 0, height - 1))
        self.filter_params["center_x"] = cx
        self.filter_params["center_y"] = cy

        if self.vignette_center_label is not None:
            self.vignette_center_label.setText(f"Center: {cx}, {cy}")
        if self.vignette_pick_button is not None:
            self.vignette_pick_button.setEnabled(True)

    def _update_pixelation_region_label(self):
        """Updates pixelation region label state"""
        if self.current_filter != "pixelation":
            return

        if self.pixelation_region_label is None:
            return

        image = self.original_viewer.get_original_image()
        if image is None:
            self.pixelation_region_label.setText("Area: n/a")
            if self.pixelation_pick_button is not None:
                self.pixelation_pick_button.setEnabled(False)
            return

        if self.pixelation_pick_button is not None:
            self.pixelation_pick_button.setEnabled(True)

        x = self.filter_params.get("x")
        y = self.filter_params.get("y")
        width = self.filter_params.get("width")
        height = self.filter_params.get("height")

        if None in (x, y, width, height):
            self.pixelation_region_label.setText("Area: n/a")
        else:
            image_h, image_w = image.shape[:2]
            clamped_x = int(np.clip(x, 0, image_w - 1))
            clamped_y = int(np.clip(y, 0, image_h - 1))
            max_width = max(1, image_w - clamped_x)
            max_height = max(1, image_h - clamped_y)
            clamped_width = int(np.clip(width, 1, max_width))
            clamped_height = int(np.clip(height, 1, max_height))

            bottom_right_x = clamped_x + clamped_width - 1
            bottom_right_y = clamped_y + clamped_height - 1

            self._set_param_value("x", clamped_x)
            self._set_param_value("y", clamped_y)
            self._set_param_value("width", clamped_width)
            self._set_param_value("height", clamped_height)

            self.pixelation_region_label.setText(
                f"Area: ({clamped_x}, {clamped_y}) → ({bottom_right_x}, {bottom_right_y})"
            )

    def _update_glare_center_label(self):
        """Updates glare center label state"""
        if self.current_filter != "glare":
            return

        if self.glare_center_label is None:
            return

        image = self.original_viewer.get_original_image()
        if image is None:
            self.glare_center_label.setText("Center: n/a")
            if self.glare_pick_button is not None:
                self.glare_pick_button.setEnabled(False)
            return

        if self.glare_pick_button is not None:
            self.glare_pick_button.setEnabled(True)

        cx = self.filter_params.get("center_x")
        cy = self.filter_params.get("center_y")

        height, width = image.shape[:2]
        if cx is None or cy is None:
            cx = width // 2
            cy = height // 2

        cx = int(np.clip(cx, 0, width - 1))
        cy = int(np.clip(cy, 0, height - 1))
        self._set_param_value("center_x", cx)
        self._set_param_value("center_y", cy)
        self.glare_center_label.setText(f"Center: {cx}, {cy}")

    def _save_temp_processed(self, image: np.ndarray):
        """Saves processed image to a temporary location for external viewing"""
        try:
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / "cv_lab_processed.png"
            if cv2.imwrite(str(temp_path), image):
                self.processed_temp_path = temp_path
            else:
                QMessageBox.warning(self, "Error", "Failed to save processed image for external viewing.")
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to prepare processed image: {exc}")

    def open_original_external(self):
        """Opens the original image in the default image viewer"""
        image = self.original_viewer.get_original_image()
        if image is None:
            return

        target_path: Optional[Path] = None
        if self.original_image_path and self.original_image_path.exists():
            target_path = self.original_image_path
        else:
            try:
                temp_dir = Path(tempfile.gettempdir())
                temp_path = temp_dir / "cv_lab_original.png"
                if cv2.imwrite(str(temp_path), image):
                    target_path = temp_path
                    self.original_image_path = temp_path
                else:
                    QMessageBox.warning(self, "Error", "Failed to save original image for external viewing.")
                    return
            except Exception as exc:
                QMessageBox.warning(self, "Error", f"Failed to prepare original image: {exc}")
                return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target_path)))

    def open_processed_external(self):
        """Opens the processed image in the default image viewer"""
        image = self.processed_viewer.get_original_image()
        if image is None:
            return

        if not self.processed_temp_path or not self.processed_temp_path.exists():
            self._save_temp_processed(image)

        if self.processed_temp_path and self.processed_temp_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.processed_temp_path)))


def run_app():
    """Runs application"""
    app = QApplication(sys.argv)
    window = FilterWindow()
    window.show()
    sys.exit(app.exec_())
