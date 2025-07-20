
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Drone Show Mockup", layout="wide")
st.title("Drone Show Mockup Grid")
st.markdown("Upload your logo to generate a realistic 4-panel drone light show mockup.")

def night_sky_background(h, w):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(300):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(bg, (x, y), 1, (np.random.randint(180, 255),) * 3, -1)
    gradient = np.linspace(20, 60, h).astype(np.uint8)
    for y in range(h):
        bg[y, :, :] += gradient[y]
    return bg

def draw_drone_show(contours, height, width, drone_count, fill=False):
    spacing = int(np.sqrt((height * width) / drone_count))
    canvas = night_sky_background(height, width)

    for contour in contours:
        dist = 0
        last_point = contour[0][0]
        for point in contour[1:]:
            point = point[0]
            d = np.linalg.norm(np.array(point) - np.array(last_point))
            dist += d
            if dist >= spacing:
                x, y = point
                cv2.circle(canvas, (x, y), 2, (0, 255, 255), -1)
                dist = 0
                last_point = point

        if fill:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            for y in range(0, height, spacing):
                for x in range(0, width, spacing):
                    if mask[y, x]:
                        cv2.circle(canvas, (x, y), 2, (0, 255, 255), -1)
    return canvas

uploaded_file = st.file_uploader("Upload PNG or JPG logo", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGBA").resize((512, 512))
    image_np = np.array(image)

    st.image(image, caption="Uploaded Logo", use_container_width=True)

    st.subheader("Generating Mockup Grid...")

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height, width = image_np.shape[:2]

    mockup_grid = {
        "100 Drones (Outline)": draw_drone_show(contours, height, width, 100, fill=False),
        "300 Drones": draw_drone_show(contours, height, width, 300, fill=True),
        "500 Drones": draw_drone_show(contours, height, width, 500, fill=True),
        "1000 Drones": draw_drone_show(contours, height, width, 1000, fill=True),
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, (title, img) in zip(axes, mockup_grid.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color='white')
        ax.axis("off")
        ax.set_facecolor("black")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor="black")
    plt.close()
    buf.seek(0)

    st.image(buf, caption="Mockup Preview", use_container_width=True)
    st.download_button("Download Full Mockup Grid", data=buf, file_name="drone_mockup_grid.png", mime="image/png")
