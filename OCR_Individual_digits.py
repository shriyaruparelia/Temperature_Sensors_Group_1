#Separate digits

import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
from pathlib import Path



VIDEO_PATH = r"experiment_1_2.mp4"
OUTPUT_EXCEL = r"Trial1_9Dec.xlsx"


SAMPLE_EVERY_SECONDS = 3.0  # e.g. every 1 second

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Region Of Interest (ROI)

ROI_A = (180, 150, 50, 100) 
ROI_B = (230, 150, 50, 100)  
ROI_C = (280, 150, 50, 100)  

SHOW_SAMPLE_FRAME = True


INVERT_FOR_OCR = False

# Smoothing
MAX_STEP = 1.0

# Save first N crops for debugging (per digit)
DEBUG_SAVE_FIRST_N = 0  # e.g. 10 if you want debug images

# Tesseract config for single digit
TESS_CONFIG_DIGIT = r"--psm 10 -c tessedit_char_whitelist=0123456789"

# Fallback digit if we have absolutely nothing (first frames)
FALLBACK_DIGIT = 0


def ocr_single_digit(roi_bgr):
    """OCR a single digit from a small ROI. Returns int 0–9 or None."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale to help OCR
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Light blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if INVERT_FOR_OCR:
        thresh = cv2.bitwise_not(thresh)

    # Morphology to clean thin segments
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # OCR
    text = pytesseract.image_to_string(thresh, config=TESS_CONFIG_DIGIT).strip()

    # Fix common OCR mistakes
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = text.replace("S", "5")

    digits = re.findall(r"\d", text)
    if not digits:
        return None

    try:
        return int(digits[0])
    except ValueError:
        return None


def main():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print("ERROR: Video not found")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("ERROR: Could not read FPS")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.3f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} s")

    frame_step = max(1, int(round(SAMPLE_EVERY_SECONDS * fps)))
    print(f"Sampling every {SAMPLE_EVERY_SECONDS}s -> {frame_step} frames")

    # Show a sample frame with the three ROIs
    if SHOW_SAMPLE_FRAME:
        ret, frame = cap.read()
        if ret:
            for (x, y, w, h), color in zip(
                (ROI_A, ROI_B, ROI_C),
                ((0, 0, 255), (0, 255, 0), (255, 0, 0))
            ):
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.imshow("ROI Preview (press any key)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    readings = []
    last_value = None
    last_a = None
    last_b = None
    last_c = None

    current_frame = 0
    debug_saved = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = current_frame / fps
        hh = int(time_sec // 3600)
        mm = int((time_sec % 3600) // 60)
        ss = int(time_sec % 60)
        time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        xA, yA, wA, hA = ROI_A
        xB, yB, wB, hB = ROI_B
        xC, yC, wC, hC = ROI_C

        roiA = frame[yA:yA + hA, xA:xA + wA]
        roiB = frame[yB:yB + hB, xB:xB + wB]
        roiC = frame[yC:yC + hC, xC:xC + wC]

        # Optional debug crops
        if DEBUG_SAVE_FIRST_N > 0 and debug_saved < DEBUG_SAVE_FIRST_N:
            cv2.imwrite(f"debug_A_{debug_saved:03d}.png", roiA)
            cv2.imwrite(f"debug_B_{debug_saved:03d}.png", roiB)
            cv2.imwrite(f"debug_C_{debug_saved:03d}.png", roiC)
            debug_saved += 1

        a_raw = ocr_single_digit(roiA)
        b_raw = ocr_single_digit(roiB)
        c_raw = ocr_single_digit(roiC)

        # Confidence = how many digits we actually saw this frame
        confidence = sum(d is not None for d in (a_raw, b_raw, c_raw))

        # Use current OCR if available, otherwise last known, otherwise fallback digit
        a_used = a_raw if a_raw is not None else (last_a if last_a is not None else FALLBACK_DIGIT)
        b_used = b_raw if b_raw is not None else (last_b if last_b is not None else FALLBACK_DIGIT)
        c_used = c_raw if c_raw is not None else (last_c if last_c is not None else FALLBACK_DIGIT)

        # Update last known digits only when OCR sees something
        if a_raw is not None:
            last_a = a_raw
        if b_raw is not None:
            last_b = b_raw
        if c_raw is not None:
            last_c = c_raw

        number_str = f"{a_used}{b_used}.{c_used}"
        try:
            raw_value = float(number_str)
        except ValueError:
            raw_value = None

        value = raw_value

        # ---- confidence-aware smoothing ----
        if last_value is None:
            # first real value we see
            value_smoothed = value
        else:
            if confidence == 0:
                # this frame is pure fallback (no digit recognized) → just keep last_value
                value_smoothed = last_value
            else:
                # we saw at least one real digit; allow jump unless it's crazy
                if value is not None and abs(value - last_value) > MAX_STEP and confidence < 3:
                    # big jump but low confidence (e.g. only 1 digit) → keep last_value
                    value_smoothed = last_value
                else:
                    value_smoothed = value if value is not None else last_value

        if value_smoothed is not None:
            last_value = value_smoothed

        print(
            f"{time_str}  digits_raw=({a_raw}, {b_raw}, {c_raw})  "
            f"digits_used=({a_used}, {b_used}, {c_used})  "
            f"RAW={raw_value}  FINAL={value_smoothed}"
        )

        readings.append({
            "time_seconds": round(time_sec, 3),
            "time_hhmmss": time_str,
            "digit_a_raw": a_raw,
            "digit_b_raw": b_raw,
            "digit_c_raw": c_raw,
            "digit_a_used": a_used,
            "digit_b_used": b_used,
            "digit_c_used": c_used,
            "value_raw": raw_value,
            "value_smoothed": value_smoothed,
        })

        current_frame += frame_step

    cap.release()

    df = pd.DataFrame(readings)
    df.to_excel(OUTPUT_EXCEL, index=False)

    print(f"\n✅ Saved {len(readings)} rows to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()

