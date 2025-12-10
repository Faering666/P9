import cv2
import numpy as np
from pathlib import Path
import rasterio
from typing import Tuple, List


class MultispectralAligner:
    """Align and resize images to match multispectral band dimensions"""

    def __init__(self, rgb_path: str, reference_band_path: str = None, center_crop_rgb: bool = True):
        """
        Initialize with RGB image and optional reference band for sizing

        Args:
            rgb_path: Path to RGB image file
            reference_band_path: Path to a multispectral band to use as size reference
                                If None, will be set when processing bands
            center_crop_rgb: If True, center-crop RGB to match multispectral aspect ratio
                           before resizing (recommended for DJI Mavic 3 MS)
        """
        self.rgb = cv2.imread(rgb_path)
        if self.rgb is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")

        self.rgb_gray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)
        self.center_crop_rgb = center_crop_rgb

        # If reference band provided, use its dimensions
        if reference_band_path:
            ref_band = cv2.imread(reference_band_path, cv2.IMREAD_UNCHANGED)
            if ref_band is None:
                raise ValueError(f"Could not load reference band: {reference_band_path}")
            self.target_height, self.target_width = ref_band.shape[:2]
        else:
            # Will be set from first band
            self.target_height = None
            self.target_width = None

        # Feature detector for alignment
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def _find_rgb_to_ms_transform(self, ms_band: np.ndarray) -> np.ndarray:
        """
        Find homography transformation from RGB to multispectral using feature matching

        Args:
            ms_band: Reference multispectral band

        Returns:
            3x3 homography matrix
        """
        # Downsample RGB to similar resolution as MS for better feature matching
        scale = min(ms_band.shape[1] / self.rgb.shape[1],
                   ms_band.shape[0] / self.rgb.shape[0])
        rgb_downsampled = cv2.resize(self.rgb, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
        rgb_gray = cv2.cvtColor(rgb_downsampled, cv2.COLOR_BGR2GRAY)

        # Ensure MS band is grayscale and 8-bit
        if len(ms_band.shape) == 3:
            ms_gray = cv2.cvtColor(ms_band, cv2.COLOR_BGR2GRAY)
        else:
            ms_gray = ms_band

        # Convert to 8-bit if needed (SIFT requires 8-bit images)
        if ms_gray.dtype != np.uint8:
            # Normalize to 0-255 range
            ms_gray = cv2.normalize(ms_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Detect features
        print("  Detecting features with SIFT...")
        kp1, des1 = self.detector.detectAndCompute(rgb_gray, None)
        kp2, des2 = self.detector.detectAndCompute(ms_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            raise ValueError("Not enough features detected for alignment")

        print(f"  Found {len(kp1)} features in RGB, {len(kp2)} in MS")

        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        print(f"  Found {len(good_matches)} good matches")

        if len(good_matches) < 10:
            raise ValueError(f"Not enough good matches ({len(good_matches)}) for alignment")

        # Extract matched keypoint coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("Failed to compute homography")

        # Scale homography to work with original RGB resolution
        # H maps from downsampled RGB -> MS
        # We need H' that maps from original RGB -> MS
        # H' = H @ S where S scales original RGB down to downsampled size
        scale_matrix = np.array([[scale, 0, 0],
                                 [0, scale, 0],
                                 [0, 0, 1]], dtype=np.float64)
        H_scaled = H @ scale_matrix

        inliers = np.sum(mask)
        print(f"  Homography computed with {inliers}/{len(good_matches)} inliers")

        return H_scaled

    def create_aligned_stack(self, band_paths: dict,
                             method: str = 'resize',
                             output_path: str = None,
                             output_dir: str = None) -> np.ndarray:
        """
        Create aligned multispectral stack
        All images will be resized to match the multispectral band dimensions

        NOTE: For DJI Mavic 3 MS, multispectral bands are from the same camera
        (already aligned). Only RGB needs to be cropped/resized to match.

        Args:
            band_paths: Dictionary with band names as keys and file paths as values
                       Example: {'red': 'path/to/red.tif', 'nir': 'path/to/nir.tif'}
            method: Alignment method - 'resize' (recommended for DJI M3M) or 'ecc' (experimental)
            output_path: Optional path to save the stack
            output_dir: Directory to save individual band files (required if save_individual_bands=True)

        Returns:
            Stacked array with shape (height, width, n_bands)
        """
        aligned_bands = []

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # First pass: determine target dimensions from first band if not set
        if self.target_width is None or self.target_height is None:
            first_band_path = list(band_paths.values())[0]
            first_band = cv2.imread(first_band_path, cv2.IMREAD_UNCHANGED)
            if first_band is None:
                raise ValueError(f"Could not load first band: {first_band_path}")
            self.target_height, self.target_width = first_band.shape[:2]
            print(f"Target dimensions set to: {self.target_width}x{self.target_height}")

        # Process and save RGB image
        print("Processing RGB image...")

        # Use feature matching to align RGB to multispectral FOV
        if self.center_crop_rgb and method == 'sift':
            # Load first MS band as reference (use green since it's similar to RGB green channel)
            ref_band_name = 'green' if 'green' in band_paths else list(band_paths.keys())[0]
            ref_band_path = band_paths[ref_band_name]
            ref_band = cv2.imread(ref_band_path, cv2.IMREAD_UNCHANGED)

            if ref_band is None:
                raise ValueError(f"Could not load reference band: {ref_band_path}")

            print(f"  Using '{ref_band_name}' band as reference for alignment")

            try:
                # Find homography from RGB to MS
                H = self._find_rgb_to_ms_transform(ref_band)

                # Warp RGB image using homography to match MS FOV
                rgb_aligned = cv2.warpPerspective(self.rgb, H,
                                                   (self.target_width, self.target_height),
                                                   flags=cv2.INTER_CUBIC)
                rgb_resized = rgb_aligned
                print("  RGB aligned to multispectral FOV using feature matching")

            except Exception as e:
                print(f"  Feature matching failed: {e}")
                print("  Falling back to simple resize")
                rgb_resized = cv2.resize(self.rgb, (self.target_width, self.target_height),
                                       interpolation=cv2.INTER_CUBIC)
        else:
            # Simple resize without alignment
            rgb_resized = cv2.resize(self.rgb, (self.target_width, self.target_height),
                                   interpolation=cv2.INTER_CUBIC)


        rgb_output_path = output_dir_path / "aligned_rgb.TIF"
        cv2.imwrite(str(rgb_output_path), rgb_resized)
        print(f"  Saved aligned RGB to {rgb_output_path}")

        # Process each multispectral band (already aligned to each other from same camera)
        for band_name, band_path in band_paths.items():
            print(f"Processing {band_name} band...")
            band = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)
            if band is None:
                raise ValueError(f"Could not load band: {band_path}")

            # Multispectral bands should already be at target size, but resize if needed
            if band.shape[:2] != (self.target_height, self.target_width):
                print(f"  Resizing {band_name} from {band.shape[:2]} to target size")
                aligned = cv2.resize(band, (self.target_width, self.target_height),
                                   interpolation=cv2.INTER_CUBIC)
            else:
                aligned = band

            # Ensure single channel
            if len(aligned.shape) == 3:
                aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

            aligned_bands.append(aligned)

            # Save individual band
            band_output_path = output_dir_path / f"aligned_{band_name}.TIF"
            cv2.imwrite(str(band_output_path), aligned)
            print(f"  Saved to {band_output_path}")

        # Stack bands
        stack = np.stack(aligned_bands, axis=-1)

        if output_path:
            # Save as multi-band TIFF
            self._save_multiband_tiff(stack, output_path, list(band_paths.keys()))

        return stack

    def _save_multiband_tiff(self, stack: np.ndarray,
                             output_path: str,
                             band_names: List[str]):
        """Save multiband array as GeoTIFF"""
        height, width, n_bands = stack.shape

        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=n_bands,
                dtype=stack.dtype,
                compress='lzw'
        ) as dst:
            for i in range(n_bands):
                dst.write(stack[:, :, i], i + 1)
                dst.set_band_description(i + 1, band_names[i])

def main():
    """Example usage"""

    # Paths to your images
    rgb_path = "data/MS_Sri_Lanka/DJI_20230814123320_0001_D.JPG"

    band_paths = {
        'red': 'data/MS_Sri_Lanka/DJI_20230814123320_0001_MS_R.TIF',
        'green': 'data/MS_Sri_Lanka/DJI_20230814123320_0001_MS_G.TIF',
        'red_edge': 'data/MS_Sri_Lanka/DJI_20230814123320_0001_MS_RE.TIF',
        'nir': 'data/MS_Sri_Lanka/DJI_20230814123320_0001_MS_NIR.TIF'
    }

    # Initialize aligner with center crop enabled
    aligner = MultispectralAligner(rgb_path, center_crop_rgb=True)

    # Create aligned stack and save individual bands
    # Method 'sift' uses feature matching to align RGB to MS FOV
    # Method 'resize' for simple resize without alignment
    stack = aligner.create_aligned_stack(
        band_paths,
        method='sift',
        output_path='aligned_stack_ecc.TIF',
        output_dir='aligned_bands'
    )

    print(f"\nAligned stack shape: {stack.shape}")
    print("Alignment complete!")

    # Optional: Calculate NDVI as example
    if 'nir' in band_paths and 'red' in band_paths:
        nir_idx = list(band_paths.keys()).index('nir')
        red_idx = list(band_paths.keys()).index('red')

        nir = stack[:, :, nir_idx].astype(float)
        red = stack[:, :, red_idx].astype(float)

        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)

        # Save NDVI
        cv2.imwrite('ndvi.TIF', ((ndvi + 1) * 127.5).astype(np.uint8))
        print("NDVI calculated and saved")


if __name__ == "__main__":
    main()