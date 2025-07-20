
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Drone Show Mockup", layout="centered")
st.title("Realistic Drone Show Mockup")
st.markdown("Upload your logo to see how it might look with 1000 drones in the sky.")

def generate_sky_background(height, width):
    sky = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(500):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        cv2.circle(sky, (x, y), 1, (255, 255, 255), -1)
    gradient = np.linspace(0, 30, height).astype(np.uint8)
    for y in range(height):
        sky[y] = sky[y] + gradient[y]
    return sky

def generate_drone_mockup(image_np, spacing=2):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    height, width = image_np.shape[:2]
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    glow_layer = np.zeros((height, width, 3), dtype=np.uint8)

    for contour in contours:
        dist = 0
        last_point = contour[0][0]
        for point in contour[1:]:
            point = point[0]
            d = np.linalg.norm(np.array(point) - np.array(last_point))
            dist += d
            if dist >= spacing:
                x, y = point
                cv2.circle(glow_layer, (x, y), 6, (0, 255, 255), -1)
                cv2.circle(canvas, (x, y), 2, (255, 255, 255, 255), -1)
                dist = 0
                last_point = point

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        for y in range(0, height, spacing):
            for x in range(0, width, spacing):
                if mask[y, x]:
                    cv2.circle(glow_layer, (x, y), 6, (0, 255, 255), -1)
                    cv2.circle(canvas, (x, y), 2, (255, 255, 255, 255), -1)

    blurred_glow = cv2.GaussianBlur(glow_layer, (15, 15), 0)
    canvas[..., :3] = cv2.addWeighted(canvas[..., :3], 1.0, blurred_glow, 0.6, 0)
    return canvas

def apply_background_and_tilt(mockup_rgba):
    height, width = mockup_rgba.shape[:2]
    sky = generate_sky_background(height, width)
    overlay = mockup_rgba[..., :3]
    alpha = mockup_rgba[..., 3:] / 255.0
    composite = (overlay * alpha + sky * (1 - alpha)).astype(np.uint8)

    src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_pts = np.float32([[40, 0], [width - 40, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    tilted = cv2.warpPerspective(composite, M, (width, height))
    return tilted

uploaded_file = st.file_uploader("Upload a PNG or JPG logo", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGBA").resize((512, 512))
    image_np = np.array(image)

    st.image(image, caption="Uploaded Logo", use_container_width=True)

    st.subheader("Generating Drone Show Mockup...")
    mockup_rgba = generate_drone_mockup(image_np)
    final_output = apply_background_and_tilt(mockup_rgba)

    st.image(final_output, caption="1000 Drone Mockup", use_container_width=True)

    result = Image.fromarray(final_output)
    buf = BytesIO()
    result.save(buf, format="PNG")
    st.download_button("Download Mockup", data=buf.getvalue(), file_name="drone_show_mockup.png", mime="image/png")
