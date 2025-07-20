
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Drone Show Builder", layout="centered")
st.title("Drone Show Mockup (Structured)")
st.markdown("Upload a logo and select the number of drones to see structured drone placement based on outline and fill logic.")

def extract_outline(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, []
    return max(contours, key=cv2.contourArea), contours

def distribute_outline_points(contour, num_points):
    arc_len = cv2.arcLength(contour, closed=True)
    interval = arc_len / num_points
    points = []
    prev = contour[0][0]
    acc_len = 0.0
    points.append(prev)
    for pt in contour[1:]:
        pt = pt[0]
        d = np.linalg.norm(np.array(pt) - np.array(prev))
        acc_len += d
        if acc_len >= interval:
            points.append(pt)
            acc_len = 0
        prev = pt
    return points[:num_points]

def fill_inside_shape(contour, spacing, shape_size):
    mask = np.zeros(shape_size[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    fill_points = []
    for y in range(0, shape_size[0], spacing):
        for x in range(0, shape_size[1], spacing):
            if mask[y, x] > 0:
                fill_points.append((x, y))
    return fill_points

def generate_mockup(image_np, drone_count):
    contour, all_contours = extract_outline(image_np)
    if contour is None:
        return None

    height, width = image_np.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if drone_count == 100:
        points = distribute_outline_points(contour, 100)
    elif drone_count == 300:
        outline = distribute_outline_points(contour, 100)
        fill = fill_inside_shape(contour, spacing=18, shape_size=image_np.shape)
        points = outline + fill[:200]
    elif drone_count == 500:
        outline = distribute_outline_points(contour, 150)
        fill = fill_inside_shape(contour, spacing=14, shape_size=image_np.shape)
        points = outline + fill[:350]
    elif drone_count == 1000:
        outline = distribute_outline_points(contour, 200)
        fill = fill_inside_shape(contour, spacing=10, shape_size=image_np.shape)
        points = outline + fill[:800]
    else:
        points = []

    for pt in points:
        cv2.circle(canvas, tuple(pt), 2, (0, 255, 255), -1)

    return canvas

uploaded_file = st.file_uploader("Upload PNG or JPG logo", type=["png", "jpg", "jpeg"])
drone_count = st.selectbox("Select drone count", [100, 300, 500, 1000])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGBA").resize((512, 512))
    image_np = np.array(image)
    st.image(image, caption="Uploaded Logo", use_container_width=True)

    st.subheader(f"Generating {drone_count} Drone Mockup...")
    mockup = generate_mockup(image_np, drone_count)
    if mockup is not None:
        st.image(mockup, caption=f"{drone_count} Drone Mockup", use_container_width=True)
        result = Image.fromarray(cv2.cvtColor(mockup, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        result.save(buf, format="PNG")
        st.download_button("Download Mockup", data=buf.getvalue(), file_name=f"drone_mockup_{drone_count}.png", mime="image/png")
    else:
        st.error("Could not extract outline from the uploaded image.")
