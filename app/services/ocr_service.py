
import cv2
import pytesseract
import numpy as np
from typing import Dict, Any, List, Tuple
import os
from datetime import datetime
from pathlib import Path
import hashlib
import logging
from app.core.config import settings
 
logger = logging.getLogger(__name__)


class OCRService:

    EXTRACT_DIR = Path("extract")
    MAX_WORKERS = 4  # Number of parallel OCR threads (if supported)

    def __init__(self, use_parallel: bool = True):
        # Create extract folder
        self.EXTRACT_DIR.mkdir(exist_ok=True)
        self.use_parallel = use_parallel

        # Check if threading is available
        if self.use_parallel:
            try:
                from concurrent.futures import ThreadPoolExecutor
                self._thread_executor_available = True
            except ImportError:
                logger.warning("ThreadPoolExecutor not available, using sequential processing")
                self._thread_executor_available = False
                self.use_parallel = False
    
    def extract_text(self, image_bytes: bytes, save_text: bool = True) -> Dict[str, Any]:

        img_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            return {"raw_text": "", "avg_confidence": 0.0, "words": [], "saved_file": None}

        # Run FULL multi-pass OCR (17 passes for accuracy)
        results = self._multi_pass_ocr(image)

        # Score and pick best result
        def score_result(r):
            text = r.get("raw_text", "")
            conf = r.get("avg_confidence", 0.0)
            words = r.get("words", [])
            alpha_count = sum(c.isalnum() for c in text)
            word_count = len(words)
            return (alpha_count * 0.4) + (conf * 100 * 0.4) + (word_count * 0.2)

        best_result = max(results, key=score_result)
        
        # Save OCR text to file
        saved_file = None
        if save_text and best_result.get("raw_text"):
            saved_file = self._save_ocr_text(image_bytes, best_result)
            best_result["saved_file"] = saved_file
        
        return best_result

    def _save_ocr_text(self, image_bytes: bytes, ocr_result: Dict[str, Any]) -> str:
        try:
            image_hash = hashlib.md5(image_bytes[:1024]).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ocr_{timestamp}_{image_hash}.txt"
            filepath = self.EXTRACT_DIR / filename
            
            # Prepare content
            content = []
            content.append("=" * 60)
            content.append(f"OCR Extraction - {datetime.now().isoformat()}")
            content.append("=" * 60)
            content.append(f"Average Confidence: {ocr_result.get('avg_confidence', 0):.2%}")
            content.append(f"Total Words Detected: {len(ocr_result.get('words', []))}")
            content.append("")
            content.append("--- RAW EXTRACTED TEXT ---")
            content.append("")
            content.append(ocr_result.get("raw_text", ""))
            content.append("")
            content.append("--- WORD-LEVEL DETAILS ---")
            content.append("")
            
            for i, word in enumerate(ocr_result.get("words", [])[:50]):  # First 50 words
                content.append(f"{i+1}. '{word.get('text', '')}' (conf: {word.get('confidence', 0):.2%})")
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            
            logger.info(f"Saved OCR text to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save OCR text: {e}")
            return None

    def _multi_pass_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:

        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        deskewed = self._deskew_image(gray)
        denoised = cv2.fastNlMeansDenoising(deskewed, None, 10, 7, 21)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        enhanced = self._remove_borders(enhanced)

        # Prepare all preprocessed images with their configs
        ocr_tasks: List[Tuple[np.ndarray, str]] = []
        
        # Pass 1-2: Basic + enhanced
        ocr_tasks.append((deskewed, "--oem 3 --psm 3"))
        ocr_tasks.append((enhanced, "--oem 3 --psm 3"))

        # Pass 3-4: Scaled (for small images)
        scale_factor = 2 if max(h, w) < 1500 else 1.5 if max(h, w) < 2500 else 1
        if scale_factor > 1:
            scaled = cv2.resize(enhanced, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            scaled = self._sharpen_image(scaled)
            ocr_tasks.append((scaled, "--oem 3 --psm 3"))
            ocr_tasks.append((clahe.apply(scaled), "--oem 3 --psm 3"))

        # Pass 5: Bilateral filter
        bilateral = cv2.bilateralFilter(deskewed, 9, 75, 75)
        ocr_tasks.append((bilateral, "--oem 3 --psm 3"))

        # Pass 6: Otsu binarization
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ocr_tasks.append((otsu, "--oem 3 --psm 6"))

        # Pass 7-9: Adaptive threshold (3 block sizes)
        for block_size in [11, 15, 21]:
            adaptive = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, 2
            )
            ocr_tasks.append((adaptive, "--oem 3 --psm 6"))

        # Pass 10-11: Morphological operations
        kernel_close = np.ones((2, 2), np.uint8)
        morph_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_close)
        ocr_tasks.append((morph_close, "--oem 3 --psm 6"))
        
        kernel_open = np.ones((1, 1), np.uint8)
        morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_open)
        ocr_tasks.append((morph_open, "--oem 3 --psm 6"))

        # Pass 12: Sharpened
        sharpened = self._sharpen_image(enhanced)
        ocr_tasks.append((sharpened, "--oem 3 --psm 3"))

        # Pass 13-16: Different PSM modes
        ocr_tasks.append((enhanced, "--oem 3 --psm 4"))   # Single column
        ocr_tasks.append((enhanced, "--oem 3 --psm 6"))   # Uniform block
        ocr_tasks.append((enhanced, "--oem 3 --psm 11"))  # Sparse text
        ocr_tasks.append((enhanced, "--oem 3 --psm 12"))  # Sparse with OSD

        # Pass 17: Inverted (for dark backgrounds)
        inverted = cv2.bitwise_not(enhanced)
        ocr_tasks.append((inverted, "--oem 3 --psm 3"))

        # === STAGE 2: Run OCR (Parallel or Sequential) ===
        results: List[Dict[str, Any]] = []

        if self.use_parallel and self._thread_executor_available:
            # Parallel execution
            logger.info(f"Running {len(ocr_tasks)} OCR passes in parallel...")
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                    # Submit all OCR tasks
                    futures = {
                        executor.submit(self._run_ocr, img, cfg): i
                        for i, (img, cfg) in enumerate(ocr_tasks)
                    }

                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"OCR pass failed: {e}")
                            results.append({"raw_text": "", "avg_confidence": 0.0, "words": []})
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}, falling back to sequential")
                self.use_parallel = False
                results = []

        if not self.use_parallel or not results:
            # Sequential execution (fallback)
            logger.info(f"Running {len(ocr_tasks)} OCR passes sequentially...")
            for i, (img, cfg) in enumerate(ocr_tasks):
                try:
                    result = self._run_ocr(img, cfg)
                    results.append(result)
                except Exception as e:
                    logger.error(f"OCR pass {i+1} failed: {e}")
                    results.append({"raw_text": "", "avg_confidence": 0.0, "words": []})

        logger.info(f"Completed {len(results)} OCR passes")
        return results
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        try:
            coords = np.column_stack(np.where(image < 128))
            if len(coords) < 100:
                return image
            
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            if abs(angle) > 0.5 and abs(angle) < 15:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, rotation_matrix, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
            
            return image
        except Exception:
            return image
    
    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        try:
            contours, _ = cv2.findContours(
                cv2.bitwise_not(image), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return image
            
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            img_h, img_w = image.shape[:2]
            if w > img_w * 0.5 and h > img_h * 0.5:
                pad = 5
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(img_w - x, w + 2*pad)
                h = min(img_h - y, h + 2*pad)
                return image[y:y+h, x:x+w]
            
            return image
        except Exception:
            return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    def _run_ocr(self, processed_image: np.ndarray, config: str) -> Dict[str, Any]:
        try:
            try:
                lang = "eng+hin"
                data = pytesseract.image_to_data(
                    processed_image,
                    output_type=pytesseract.Output.DICT,
                    config=config,
                    lang=lang
                )
            except pytesseract.TesseractError:
                lang = "eng"
                data = pytesseract.image_to_data(
                    processed_image,
                    output_type=pytesseract.Output.DICT,
                    config=config,
                    lang=lang
                )

            words: List[Dict[str, Any]] = []
            confidences: List[float] = []
            high_conf_count = 0

            for i, txt in enumerate(data.get("text", [])):
                txt = txt.strip()
                try:
                    conf = int(data["conf"][i])
                except (ValueError, KeyError):
                    continue

                if txt and conf > 0:
                    word_conf = conf / 100
                    words.append({
                        "text": txt,
                        "confidence": round(word_conf, 3),
                        "bbox": {
                            "x": data["left"][i],
                            "y": data["top"][i],
                            "w": data["width"][i],
                            "h": data["height"][i]
                        }
                    })
                    confidences.append(word_conf)
                    if word_conf >= 0.7:
                        high_conf_count += 1

            # Get raw text
            try:
                raw_text = pytesseract.image_to_string(
                    processed_image, config=config, lang=lang
                )
            except pytesseract.TesseractError:
                raw_text = pytesseract.image_to_string(
                    processed_image, config=config, lang="eng"
                )

            # Calculate weighted confidence
            if confidences:
                weighted_sum = sum(
                    c * (1.5 if c >= 0.7 else 1.0) for c in confidences
                )
                total_weight = sum(
                    1.5 if c >= 0.7 else 1.0 for c in confidences
                )
                avg_conf = weighted_sum / total_weight
                
                high_conf_ratio = high_conf_count / len(confidences) if confidences else 0
                if high_conf_ratio > 0.7:
                    avg_conf = min(avg_conf * 1.1, 1.0)
            else:
                avg_conf = 0.0

            return {
                "raw_text": raw_text.strip(),
                "avg_confidence": round(avg_conf, 3),
                "words": words,
                "word_count": len(words),
                "high_conf_words": high_conf_count
            }

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {"raw_text": "", "avg_confidence": 0.0, "words": [], "word_count": 0, "high_conf_words": 0}
ocr_service = OCRService(use_parallel=settings.ocr_use_parallel)
