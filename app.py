
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Drone Logo Mockup", layout="centered")
st.title("FlightShows: Drone Logo Visualiser")
st.markdown("Upload your logo and see how it might look with 100–1000 drones in the sky.")

uploaded_file = st.file_uploader("Upload a PNG or JPG logo", type=["png", "jpg", "jpeg"])

@st.cache_data
def process_logo(image_file):
    image = Image.open(image_file).convert("RGBA")
    image = image.resize((512, 512))
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def generate_mockup(spacing):
        canvas = np.zeros_like(image_np)
        for contour in contours:
            dist = 0
            last_point = contour[0][0]
            for point in contour[1:]:
                point = point[0]
                d = np.linalg.norm(np.array(point) - np.array(last_point))
                dist += d
                if dist >= spacing:
                    cv2.circle(canvas, tuple(point), 1, (0, 255, 255, 255), -1)
                    dist = 0
                    last_point = point
        result = np.zeros_like(image_np)
        result[:, :, 3] = 255
        mask = canvas[:, :, 0] > 0
        result[mask] = canvas[mask]
        return Image.fromarray(result)

    return {
        100: generate_mockup(10),
        300: generate_mockup(6),
        500: generate_mockup(4),
        1000: generate_mockup(2)
    }

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Logo", use_column_width=True)
    st.write("Generating drone show mockups...")
    mockups = process_logo(uploaded_file)

    for count, img in mockups.items():
        st.subheader(f"{count} Drones")
        st.image(img, use_column_width=True)
        buf = BytesIO()
        img.save(buf, format="PNG")
        st.download_button(f"Download {count} Drones Mockup", data=buf.getvalue(), file_name=f"drone_mockup_{count}.png", mime="image/png")
