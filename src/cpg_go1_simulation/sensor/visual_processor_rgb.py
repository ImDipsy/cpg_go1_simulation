from dataclasses import dataclass
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage import feature, measure, morphology


@dataclass
class VisualConfig:
    """Configurations for the visual processor"""

    # Debug configuration
    DEBUG: bool  # Whether to enable debug visualization

    # HSV color thresholding ranges for detecting black lines
    BLACK_HSV_LOW: tuple = (0, 0, 0)  # Hue 0-180, Saturation 0-100, Value 0-50
    BLACK_HSV_HIGH: tuple = (180, 100, 50)

    # Canny edge detection parameters (lowering thresholds)
    CANNY_SIGMA: float = 2.0
    CANNY_LOW_THRESHOLD: float = 0.05
    CANNY_HIGH_THRESHOLD: float = 0.2

    # Morphological processing parameters (reducing kernel size)
    MORPH_KERNEL_SIZE: int = 3  # Reducing kernel size to retain more details
    PATH_DILATION_SIZE: int = 2

    # Deviation calculation parameters
    POSITION_WEIGHT: float = 1.0  # Position deviation weight
    HEADING_WEIGHT: float = 0.2  # Heading deviation weight

    # Filter parameters
    FILTER_CUTOFF: float = 10.0  # Cutoff frequency (Hz)
    FILTER_ORDER: int = 4  # Filter order
    SAMPLE_RATE: float = 100.0  # Sampling rate (Hz)


class PathTracker:
    """Path tracking and visual processing class"""

    def __init__(self, DEBUG: bool = False):
        self.config = VisualConfig(DEBUG=DEBUG)
        self.fig = None
        self.axes = None
        if self.config.DEBUG:
            plt.ion()  # Enable interactive mode
        self._setup_filter()
        self.error_history = []
        self.ratio_history = []
        self.consecutive_direction = 0
        self.last_error_direction = 0
        self.lost_track_counter = 0
        self.max_lost_frames = 10
        self.last_valid_direction = 0
        self.last_valid_error = 0.0
        self.last_ratio_list = None
        self.stopFlag = False

    def process_frame(self, rgb_frame: np.ndarray) -> Tuple[float, list]:
        """Process a single RGB frame"""
        try:
            # 1. Get path mask and edges
            mask, edges = self._get_path_from_rgb(rgb_frame)

            if mask is None or not np.any(mask):
                self.lost_track_counter += 1
                if self.lost_track_counter <= self.max_lost_frames:
                    return (
                        self.last_valid_error * 1.2,
                        self._get_adaptive_ratio(self.last_valid_error * 1.2),
                        self.stopFlag,
                    )
                else:
                    self.stopFlag = self._detect_stop_condition(mask)
                    return self.last_valid_error, self.last_ratio_list, self.stopFlag
            else:
                self.lost_track_counter = 0
                error = self._calculate_error(mask)
                self.last_valid_error = error

                # 2. Calculate filtered error
                filtered_error = self._filter_error(error)

                # 3. Calculate adaptive velocity ratio
                velocity_ratio = self._get_adaptive_ratio(filtered_error)

                return filtered_error, velocity_ratio, self.stopFlag

        except Exception as e:
            print(f"RGB image error: {e}")
            import traceback

            traceback.print_exc()
            return 0.0, [0.8] * 8, self.stopFlag

    def process_straight_walking(self, imu_data: dict) -> Tuple[float, list]:
        """Process straight walking with IMU data"""
        try:
            # Get yaw angle from IMU data
            yaw = imu_data.get("orientation", [0, 0, 0])[2]

            # Calculate deviation from target direction (assumed to be 0)
            heading_error = yaw  # If yaw > 0 indicates left deviation, yaw < 0 indicates right deviation

            # Print debug information
            if self.config.DEBUG:
                print(
                    f"Current heading: {np.degrees(yaw):.1f}°, Deviation: {np.degrees(heading_error):.1f}°"
                )

            # Filter processing
            filtered_error = self._filter_error(heading_error)

            # Calculate velocity ratio based on deviation
            ratio_list = self._get_straight_walking_ratio(filtered_error)

            return filtered_error, ratio_list

        except Exception as e:
            print(f"Straight walking processing error: {e}")
            return 0.0, [0.8] * 8  # Return default values

    def _get_path_from_rgb(
        self, rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single RGB frame
        Args:
            rgb_frame: RGB image data
        Returns:
            mask: Path mask
            edges: Canny edges
        """
        try:
            # 1. Convert RGB to HSV and threshold for black lines
            hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
            black_mask = cv2.inRange(
                hsv,
                np.array(self.config.BLACK_HSV_LOW),
                np.array(self.config.BLACK_HSV_HIGH),
            )

            # 2. Morphological processing
            kernel = np.ones(
                (self.config.MORPH_KERNEL_SIZE, self.config.MORPH_KERNEL_SIZE), np.uint8
            )
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
            path_mask = black_mask

            # 3. Canny edge detection
            edges = feature.canny(
                path_mask,
                sigma=self.config.CANNY_SIGMA,
                low_threshold=self.config.CANNY_LOW_THRESHOLD,
                high_threshold=self.config.CANNY_HIGH_THRESHOLD,
            )

            # 4. Morphological processing to connect broken edges
            edges = morphology.binary_dilation(edges, morphology.disk(1))
            edges = morphology.skeletonize(edges)

            # 5. Retrieve connected components
            labels = measure.label(edges)
            regions = measure.regionprops(labels)

            if not regions:
                return np.zeros_like(path_mask, dtype=bool), edges

            # 6. Select the most probable path
            best_region = max(
                regions,
                key=lambda r: (r.area * 0.6 + r.extent * 0.4),
            )

            # 7. Create path mask
            path_mask = labels == best_region.label
            path_mask = morphology.dilation(
                path_mask, morphology.disk(self.config.PATH_DILATION_SIZE)
            )

            # 8. Retrieve endpoint coordinates
            if np.any(edges):
                edge_points = np.where(edges)
                if len(edge_points[0]) > 0:
                    top_idx = np.argmin(edge_points[0])
                    bottom_idx = np.argmax(edge_points[0])

                    self.top_point = (edge_points[1][top_idx], edge_points[0][top_idx])
                    self.bottom_point = (
                        edge_points[1][bottom_idx],
                        edge_points[0][bottom_idx],
                    )

                    if self.config.DEBUG:
                        print(
                            f"Top Coordinate: ({self.top_point[0]:.1f}, {self.top_point[1]:.1f})"
                        )
                        print(
                            f"Bottom Coordinate: ({self.bottom_point[0]:.1f}, {self.bottom_point[1]:.1f})"
                        )

            # 9. Visualize results
            if self.config.DEBUG:
                self._visualize_detection(rgb_frame, path_mask, edges)

            return path_mask, edges

        except Exception as e:
            print(f"Frame processing error: {e}")
            import traceback

            traceback.print_exc()
            return None, None

    def _detect_stop_condition(self, mask: np.ndarray) -> bool:
        """Detect stop condition: reached bottom of image or no path detected"""
        # Check if reached bottom of image
        # If there is a bottom point and the y-coordinate of the bottom point is over 98% of the image height, consider reached bottom
        if (
            hasattr(self, "bottom_point")
            and self.bottom_point[1] > mask.shape[0] * 0.98
        ):
            return True

        # Check if no path detected
        if not np.any(mask):
            return True

        return False

    def _calculate_error(self, mask: np.ndarray) -> float:
        """Calculate deviation error from the path"""
        if not np.any(mask):
            return self.last_valid_error * 1.2

        # Calculate path properties
        props = measure.regionprops(mask.astype(int))[0]
        cy, cx = props.centroid

        orientation = props.orientation

        # Get the center of the image and the bottom point
        image_center_x = mask.shape[1] / 2
        bottom_x = self.bottom_point[0] if hasattr(self, "bottom_point") else cx

        # 1. Calculate lateral offset (position deviation)
        pixel_error = bottom_x - image_center_x
        lateral_error = (
            pixel_error / mask.shape[1]
        )  # Reduce sensitivity to position error

        # 2. Calculate orientation deviation
        target_orientation = 0
        heading_error = target_orientation - orientation

        # 3. Edge detection compensation: increase turning tendency when the path approaches the edge of the image
        image_width = mask.shape[1]
        edge_threshold = (
            image_width * 0.3
        )  # 30% of the image width as the edge threshold

        if bottom_x < edge_threshold:  # Near the left edge
            lateral_error *= 10  # Increase the tendency to turn left
        elif bottom_x > (image_width - edge_threshold):  # Near the right edge
            lateral_error *= 10  # Increase the tendency to turn right

        # 4. Combine lateral and heading errors
        error = (
            self.config.POSITION_WEIGHT * lateral_error
            + self.config.HEADING_WEIGHT * heading_error
        )

        if self.config.DEBUG:
            print(
                f"Position: Center={image_center_x:.1f}, Current={cx:.1f}, Bottom X:{bottom_x:.1f}, Deviation={lateral_error:.6f}m"
            )
            print(
                f"Target: Heading={np.degrees(target_orientation):.1f}°, "
                f"Current={np.degrees(orientation):.1f}°"
            )
            print(
                f"Combined deviation: {error:.6f}",
                "Lateral error:",
                lateral_error,
                "Heading error:",
                heading_error,
            )

        self.last_valid_error = error

        # Save key point information
        self.path_info = {
            "centroid": (cx, cy),
            "orientation": orientation,
            "bottom_point": (bottom_x, cy),
        }

        return error

    def _filter_error(self, error: float) -> float:
        """Filter the error"""
        self.error_history.append(error)
        if len(self.error_history) > 50:
            self.error_history.pop(0)

        if len(self.error_history) >= 20:
            try:
                # Increase filter strength
                filtered = signal.filtfilt(self.b, self.a, self.error_history)
                # Add moving average for further smoothing
                window_size = 5
                filtered_with_ma = np.convolve(
                    filtered, np.ones(window_size) / window_size, mode="valid"
                )
                return filtered_with_ma[-1]
            except ValueError:
                return np.mean(self.error_history[-7:])  # Increase the mean window size
        else:
            return np.mean(self.error_history)

    def _get_adaptive_ratio(self, error: float) -> list:
        """Improved adaptive speed ratio calculation"""
        # Check if near the edge of the view
        is_near_edge = False
        if hasattr(self, "last_valid_position"):
            # Use fixed image width or get from configuration
            image_width = 640
            is_near_edge = (
                self.last_valid_position[0] < image_width * 0.2
                or self.last_valid_position[0] > image_width * 0.8
            )

        # Adjust base speed based on edge proximity
        base_speed = (
            0.7 if is_near_edge else 0.8
        )  # Reduce base speed when near the edge

        # Adjust turning intensity based on error magnitude

        if abs(error) < 0.05:
            ratio_list = [base_speed] * 8

        elif abs(error) < 0.4:
            # Determine base intensity for turning direction
            if abs(error) < 0.2:
                turn_intensity = 0.15 if is_near_edge else 0.1
            else:
                turn_intensity = 0.2 if is_near_edge else 0.15

            # Turn left
            if error < 0:
                ratio_list = [
                    base_speed - turn_intensity,
                    base_speed + turn_intensity,
                    base_speed + turn_intensity,
                    base_speed - turn_intensity,
                ] * 2
            # Turn right
            else:
                ratio_list = [
                    base_speed + turn_intensity,
                    base_speed - turn_intensity,
                    base_speed - turn_intensity,
                    base_speed + turn_intensity,
                ] * 2

        else:
            if error < 0:
                ratio_list = [0.8, 1.2, 1.2, 0.8] * 2
            else:
                ratio_list = [1.2, 0.8, 0.8, 1.2] * 2

        self.last_ratio_list = ratio_list

        return np.clip(self._smooth_ratios(ratio_list), 0.6, 1.2).tolist()

    def _get_straight_walking_ratio(self, error: float) -> list:
        """Calculate velocity ratio for straight walking"""
        if abs(error) <= 0.03:
            # Base speed
            ratio_list = [0.8] * 8
        elif abs(error) <= 0.05:
            # Turn Right
            if error > 0:
                ratio_list = [0.9, 0.7, 0.7, 0.9] * 2
            # Turn Left
            else:
                ratio_list = [0.7, 0.9, 0.9, 0.7] * 2
        else:  # 大偏差
            # Severe left deviation
            if error > 0:
                ratio_list = [1.0, 0.6, 0.6, 1.0] * 2
            # Severe right deviation
            else:
                ratio_list = [0.6, 1.0, 1.0, 0.6] * 2

        # Smooth the ratios
        self.ratio_history.append(ratio_list)
        if len(self.ratio_history) > 10:
            self.ratio_history.pop(0)

        # Calculate smoothed ratios
        if len(self.ratio_history) > 1:
            smoothed_ratios = np.mean(self.ratio_history, axis=0)
            return smoothed_ratios.tolist()

        return ratio_list

    def _smooth_ratios(self, ratio_list: list) -> list:
        """Smooth the velocity ratios"""
        self.ratio_history.append(ratio_list)
        if len(self.ratio_history) > 10:
            self.ratio_history.pop(0)

        if len(self.ratio_history) > 1:
            smoothed_ratios = np.mean(self.ratio_history, axis=0)
            return smoothed_ratios.tolist()

        return ratio_list

    def _setup_filter(self):
        """Setup Butterworth filter for error smoothing"""
        nyq = 0.5 * self.config.SAMPLE_RATE
        cutoff_norm = self.config.FILTER_CUTOFF / nyq
        self.b, self.a = signal.butter(self.config.FILTER_ORDER, cutoff_norm, "low")

    def _visualize_detection(self, rgb_frame, mask, edges):
        """Display the detection results"""
        try:
            if self.fig is None:
                self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 5))
                plt.ion()
            else:
                for ax in self.axes:
                    ax.clear()

            # 1. Original RGB frame
            self.axes[0].imshow(rgb_frame)
            self.axes[0].set_title("Original RGB")
            self.axes[0].axis("off")

            # 2. Path mask
            self.axes[1].imshow(mask, cmap="gray")
            self.axes[1].set_title("Path Mask")
            self.axes[1].axis("off")

            # 3. Canny edges
            self.axes[2].imshow(edges, cmap="gray")
            self.axes[2].set_title("Canny Edges")
            self.axes[2].axis("off")

            # 4. Overlay display
            overlay = rgb_frame.copy()
            for i in range(3):  # Process each RGB channel
                channel = overlay[:, :, i]  # Get current channel
                if i == 1:  # Green channel
                    channel[edges > 0] = 255  # Set to max value
                else:  # Red and blue channels
                    channel[edges > 0] = 0  # Set to 0
                overlay[:, :, i] = channel  # Update back to original
            self.axes[3].imshow(overlay)
            self.axes[3].set_title("Edge Overlay")
            self.axes[3].axis("off")

            plt.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback

            traceback.print_exc()
