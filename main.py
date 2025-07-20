
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_mockup(file: UploadFile = File(...), drones: int = Form(300)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGBA").resize((300, 300))
    np_image = np.array(image)

    gray = cv2.cvtColor(np_image, cv2.COLOR_RGBA2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "No contour found"}
    contour = max(contours, key=cv2.contourArea)

    arc_len = cv2.arcLength(contour, True)
    interval = arc_len / drones
    points = []
    acc_len = 0.0
    last = contour[0][0]
    points.append(tuple(last))

    for pt in contour[1:]:
        pt = pt[0]
        d = np.linalg.norm(pt - last)
        acc_len += d
        if acc_len >= interval:
            points.append(tuple(pt))
            acc_len = 0
        last = pt

    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    for x, y in points:
        cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 255), -1)

    filename = f"/tmp/mockup_{uuid.uuid4()}.png"
    cv2.imwrite(filename, canvas)
    return FileResponse(filename, media_type="image/png", filename="mockup.png")
