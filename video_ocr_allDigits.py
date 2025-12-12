#3 Digits Together

import cv2
import pytesseract
import pandas as pd
import re
import math
from pathlib import Path


EXPECTED_MIN_VALUE = 0.0     
EXPECTED_MAX_VALUE = 200.0   


EXPECTED_DECIMALS = 1

# Invert thresholded image?
INVERT_FOR_OCR = False  # try True


VIDEO_PATH = r"experiment_1_2.mp4"

OUTPUT_EXCEL = r"Trial1_9Dec.xlsx"

SAMPLE_EVERY_SECONDS = 3.0  # e.g. every 1 second

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Region Of Interest (ROI) where the numbers are on screen
# (x, y, width, height) in pixels
# You MUST adjust these numbers to match your video.
ROI = (170, 150, 160, 100)  # example values – change!

# Whether to show a sample frame with ROI rectangle to help adjust ROI
SHOW_SAMPLE_FRAME = True

# OCR configuration: digits + decimal point only
TESS_CONFIG = r"--psm 7 -c tessedit_char_whitelist=0123456789."

DEBUG_SAVE_FIRST_N = 10



def clean_ocr_to_number(text: str) -> float | None:
    """
    Take OCR string and try to return a float.
    - Fix common misreads
    - Try to restore missing decimal (e.g. 521 -> 52.1)
    - Return None if nothing reasonable found
    """
    if not text:
        return None

    # Normalize
    text = text.strip()
    text = text.replace(",", ".")
    # Fix some common OCR confusions
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = text.replace("S", "5")

    # Keep only digits and dots
    cleaned = re.sub(r"[^0-9.]", "", text)
    if not cleaned:
        return None

    # Case 1: there is already a dot
    if "." in cleaned:
        try:
            val = float(cleaned)
            if EXPECTED_MIN_VALUE <= val <= EXPECTED_MAX_VALUE:
                return val
        except ValueError:
            pass  # fall through to digit-only handling

    # Case 2: no dot or the dot version failed
    # Extract just digits
    digits = re.sub(r"[^0-9]", "", cleaned)
    if not digits:
        return None

    # If we expect a decimal place, insert it from the right
    if EXPECTED_DECIMALS > 0 and len(digits) > EXPECTED_DECIMALS:
        idx = len(digits) - EXPECTED_DECIMALS
        candidate = digits[:idx] + "." + digits[idx:]
    else:
        candidate = digits  # fall back to integer

    try:
        val = float(candidate)
    except ValueError:
        return None

    # Final range check
    if EXPECTED_MIN_VALUE <= val <= EXPECTED_MAX_VALUE:
        return val

    # If out of range, treat as invalid
    return None



def main():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: could not read FPS from video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"Video FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_sec:.2f} seconds")

    # Frame step based on sampling interval
    frame_step = max(1, int(round(SAMPLE_EVERY_SECONDS * fps)))
    print(f"Sampling every {SAMPLE_EVERY_SECONDS}s -> every {frame_step} frames")

    # Optionally show a sample frame with ROI rectangle
    if SHOW_SAMPLE_FRAME:
        ret, frame = cap.read()
        if ret:
            x, y, w, h = ROI
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("Sample Frame with ROI (press any key to continue)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    readings = []

    current_frame = 0
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # Compute timestamp for this frame
        time_sec = current_frame / fps


        # Crop ROI
        x, y, w, h = ROI
        roi = frame[y:y + h, x:x + w]

        # ---- Preprocess for better OCR ----
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Upscale to make digits clearer
        scale = 3  # 3x enlargement
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Light blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive / Otsu threshold
        _, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if len(readings) < DEBUG_SAVE_FIRST_N:
            debug_path = f"debug_roi_{len(readings):03d}.png"
            cv2.imwrite(debug_path, thresh)


        if INVERT_FOR_OCR:
            thresh = cv2.bitwise_not(thresh)

        # ---- Run OCR ----
        ocr_text = pytesseract.image_to_string(thresh, config=TESS_CONFIG)
        ocr_text = ocr_text.strip()

        value = clean_ocr_to_number(ocr_text)

        ocr_text = ocr_text.strip()

        value = clean_ocr_to_number(ocr_text)

        readings.append({
            "time_seconds": round(time_sec, 3),
            "time_hhmmss": f"{int(time_sec//3600):02d}:{int((time_sec%3600)//60):02d}:{int(time_sec%60):02d}",
            "ocr_raw": ocr_text,
            "value": value,
        })

        print(f"{readings[-1]['time_hhmmss']}  raw='{ocr_text}'  value={value}")

        current_frame += frame_step

    cap.release()

    # Create DataFrame and save to Excel
    df = pd.DataFrame(readings)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nSaved {len(readings)} rows to '{OUTPUT_EXCEL}'.")


if __name__ == "__main__":
    main()


