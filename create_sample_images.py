"""
Script to generate realistic test vehicle images with authentic Indian License Plates.
Creates samples for standard state plates (MH, DL, KA) and BH Series plates.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_sample_car(
    car_color: tuple,
    plate_text: str,
    output_filename: str,
    is_bh: bool = False
):
    # Canvas: 800 x 500 vehicle rear / front view mockup
    width, height = 800, 500
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Background (garage / road scenery)
    # Sky / background gradient
    for y in range(250):
        intensity = int(180 + (y / 250) * 40)
        img[y, :] = (intensity - 10, intensity, intensity + 5)
    # Asphalt road
    for y in range(250, height):
        intensity = int(50 + ((y - 250) / 250) * 25)
        img[y, :] = (intensity, intensity, intensity)

    # 2. Car Body (curved rear / bumper profile)
    # Car body contour
    car_contour = np.array([
        [150, 240], [220, 160], [320, 140], [480, 140], [580, 160],
        [650, 240], [670, 340], [650, 420], [150, 420], [130, 340]
    ], dtype=np.int32)
    cv2.fillPoly(img, [car_contour], car_color)

    # Car rear windshield
    windshield = np.array([
        [240, 165], [330, 150], [470, 150], [560, 165],
        [540, 220], [260, 220]
    ], dtype=np.int32)
    cv2.fillPoly(img, [windshield], (40, 45, 50))

    # Taillights (Red & amber glow)
    cv2.rectangle(img, (160, 270), (240, 310), (30, 30, 220), -1)
    cv2.rectangle(img, (560, 270), (640, 310), (30, 30, 220), -1)
    cv2.rectangle(img, (200, 280), (235, 305), (30, 180, 240), -1)
    cv2.rectangle(img, (565, 280), (600, 305), (30, 180, 240), -1)

    # Bumper recess / Number plate holder (dark grey matte)
    cv2.rectangle(img, (270, 320), (530, 400), (25, 25, 25), -1)

    # 3. Authentic Indian High Security Registration Plate (HSRP)
    # Plate dimensions: 230 x 60 inside the holder
    plate_x1, plate_y1 = 285, 330
    plate_x2, plate_y2 = 515, 390
    plate_w = plate_x2 - plate_x1
    plate_h = plate_y2 - plate_y1

    # White reflective plate surface
    cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (245, 245, 248), -1)
    # Black outer border
    cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (20, 20, 20), 2)
    cv2.rectangle(img, (plate_x1 + 2, plate_y1 + 2), (plate_x2 - 2, plate_y2 - 2), (20, 20, 20), 1)

    # Blue vertical strip on the left with "IND" (Standard HSRP)
    strip_w = 26
    cv2.rectangle(img, (plate_x1 + 3, plate_y1 + 3), (plate_x1 + strip_w, plate_y2 - 3), (160, 40, 10), -1) # BGR: blue is 160,40,10
    # Small chakra / hologram dot
    cv2.circle(img, (plate_x1 + 14, plate_y1 + 18), 5, (220, 220, 220), -1)
    # "IND" text in blue strip
    cv2.putText(img, "IND", (plate_x1 + 4, plate_y2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Stamp License Plate Characters
    # Convert OpenCV image to PIL for clean antialiased rendering
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Use default or scalable font
    try:
        # Try loading DejaVuSans-Bold or similar system monospace fonts
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 30)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 30)
        except Exception:
            font = ImageFont.load_default()

    # Draw the text in black
    text_x = plate_x1 + strip_w + 14
    text_y = plate_y1 + 14
    draw.text((text_x, text_y), plate_text, fill=(15, 15, 15), font=font)

    # Save final image
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    pil_img.save(output_filename, quality=95)
    print(f"Created: {output_filename} with Plate: {plate_text}")


def main():
    samples = [
        # Car color (BGR), Plate Text, Output path, is_bh
        ((230, 230, 230), "MH 12 DE 1433", "sample_images/car_sample_mh12.jpg", False),
        ((180, 70, 30), "DL 3C AB 1234", "sample_images/car_sample_dl3c.jpg", False),
        ((35, 40, 180), "22 BH 6543 AA", "sample_images/car_sample_22bh.jpg", True),
        ((30, 130, 30), "KA 05 MN 9876", "sample_images/car_sample_ka05.jpg", False),
    ]

    for car_color, plate_text, path, is_bh in samples:
        create_sample_car(car_color, plate_text, path, is_bh)


if __name__ == "__main__":
    main()
