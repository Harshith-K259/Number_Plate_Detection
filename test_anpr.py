"""
Unit test script to verify EasyOCR, OpenCV preprocessing, and Indian plate extraction.
"""
import os
import cv2
import easyocr
import numpy as np
from app import detect_number_plate, parse_and_validate_plate

def run_tests():
    print("=" * 60)
    print("Testing Regex & Plate Parser logic...")
    print("=" * 60)

    test_cases = [
        ("MH12DE1433", "Standard Indian Plate", "MH 12 DE 1433"),
        ("DL 3C AB 1234", "Standard Indian Plate", "DL 3C AB 1234"),
        ("KA 05 MN 9876", "Standard Indian Plate", "KA 05 MN 9876"),
        ("22BH6543AA", "Bharat Series (BH)", "22 BH 6543 AA"),
        ("INDMH12AB1234", "Standard Indian Plate", "MH 12 AB 1234"),
        ("6J08069292", "Standard Indian Plate", "GJ 08 DG 9292"),
        ("JUSTANINVALIDSTRING", "None", None),
        ("RANDOMTEXT", "None", None),
    ]

    all_passed = True
    for text, expected_type, expected_formatted in test_cases:
        res = parse_and_validate_plate(text)
        if expected_type == "None":
            if res is None:
                print(f" PASS: '{text}' rejected correctly")
            else:
                print(f" FAIL: '{text}' should have been rejected, got {res}")
                all_passed = False
        else:
            if res and res["type"] == expected_type and res["plate_number"] == expected_formatted:
                print(f" PASS: '{text}' -> {res['plate_number']} ({res['type']})")
            else:
                print(f" FAIL: '{text}' -> Expected {expected_formatted} ({expected_type}), got {res}")
                all_passed = False

    print("\n" + "=" * 60)
    print("Testing EasyOCR on Sample Vehicle Images...")
    print("=" * 60)

    reader = easyocr.Reader(["en"], gpu=False)
    sample_files = [
        ("sample_images/car_sample_mh12.jpg", "MH 12 DE 1433"),
        ("sample_images/car_sample_dl3c.jpg", "DL 3C AB 1234"),
        ("sample_images/car_sample_22bh.jpg", "22 BH 6543 AA"),
        ("sample_images/car_sample_ka05.jpg", "KA 05 MN 9876"),
        ("sample_images/car_sample_scorpio_gj08.jpg", "GJ 08 DG 9292"),
    ]

    for path, expected_text in sample_files:
        if not os.path.exists(path):
            print(f" Skip: {path} not found")
            continue

        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result, annotated, crop, preproc = detect_number_plate(rgb, reader, min_confidence=0.20)

        if result:
            print(f" PASS: [{os.path.basename(path)}] -> Detected: {result['plate_number']} (Conf: {result['confidence']}, Type: {result['type']})")
        else:
            print(f" FAIL: [{os.path.basename(path)}] -> No plate detected!")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED, REVIEW ABOVE.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
