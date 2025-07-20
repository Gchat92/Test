
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Drone Logo Mockup", layout="wide")
st.title("Drone Logo Mockup Tool")
st.markdown("Upload your logo to see how it might look with 100 to 1000 drones.")

uploaded_file = st.file_uploader("Upload a PNG or JPG logo", type=["png", "jpg", "jpeg"])

def generate_mockup_grid(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def draw_drones(spacing, fill=False):
        canvas = np.zeros_like(image_np)
        for contour in contours:
            dist = 0
            last_point = contour[0][0]
            for point in contour[1:]:
                point = point[0]
                d = np.linalg.norm(np.array(point) - np.array(last_point))
                dist += d
                if dist >= spacing:
                    x, y = point
                    cv2.circle(canvas, (x, y), 2, (0, 255, 255, 255), -1)
                    dist = 0
                    last_point = point
            if fill:
                mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                for y in range(0, image_np.shape[0], spacing):
                    for x in range(0, image_np.shape[1], spacing):
                        if mask[y, x]:
                            cv2.circle(canvas, (x, y), 2, (0, 255, 255, 255), -1)
        return canvas

    mockups = {
        "100 Drones (Outline)": draw_drones(10, fill=False),
        "300 Drones": draw_drones(6, fill=True),
        "500 Drones": draw_drones(4, fill=True),
        "1000 Drones": draw_drones(2, fill=True),
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (label, mockup) in zip(axes, mockups.items()):
        ax.imshow(cv2.cvtColor(mockup, cv2.COLOR_BGRA2RGBA))
        ax.set_title(label, color='white')
        ax.axis("off")
        ax.set_facecolor("black")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor="black")
    plt.close()
    buf.seek(0)
    return buf

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGBA").resize((512, 512))
    image_np = np.array(image)
    st.image(image, caption="Uploaded Logo", use_container_width=True)

    st.subheader("Drone Mockup Grid")
    grid_buf = generate_mockup_grid(image_np)
    st.image(grid_buf, caption="Preview", use_container_width=True)
    st.download_button(
        label="Download Full Mockup Grid",
        data=grid_buf,
        file_name="drone_mockup_grid.png",
        mime="image/png"
    )
