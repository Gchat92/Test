
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Drone Logo Visualiser", layout="centered")
st.title("Drone Logo Visualiser")
st.markdown("Upload your logo and see how it could look made of 100–1000 drones.")

uploaded_file = st.file_uploader("Upload a PNG or JPG logo", type=["png", "jpg", "jpeg"])

def generate_mockup(image_np, spacing):
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
    except Exception as e:
        st.error(f"Error generating mockup: {e}")
        return None

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGBA").resize((512, 512))
        image_np = np.array(image)

        st.image(image, caption="Original Logo", use_column_width=True)
        st.write("Generating mockups...")

        drone_counts = [100, 300, 500, 1000]
        spacings = {100: 10, 300: 6, 500: 4, 1000: 2}

        for count in drone_counts:
            result = generate_mockup(image_np, spacings[count])
            if result:
                st.subheader(f"{count} Drones")
                st.image(result, use_column_width=True)
                buf = BytesIO()
                result.save(buf, format="PNG")
                st.download_button(
                    label=f"Download {count} Drone Mockup",
                    data=buf.getvalue(),
                    file_name=f"drone_mockup_{count}.png",
                    mime="image/png"
                )
    except Exception as err:
        st.error(f"Something went wrong: {err}")
