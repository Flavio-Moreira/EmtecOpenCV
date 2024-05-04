from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
from starlette.responses import StreamingResponse
import io
from fastapi.middleware.cors import CORSMiddleware
from sympy import Idx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(file: UploadFile):
    # Lecture de l'image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    unique_labels = set()

    for i, contour in enumerate(contours):
        if i == 0:
            continue

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Extract bounding box coordinates
        x, y, w, h = cv2.boundingRect(approx)
        x_mid = int(x + (w / 3))
        y_mid = int(y + (h / 1.5))
        coords = (x_mid, y_mid)


        # Determine shape label based on the number of vertices
        if len(approx) == 3:
            shape_label = "Triangle"
        elif len(approx) == 4:
            shape_label = "Quadrilateral"
        elif len(approx) == 5:
            shape_label = "Pentagon"
        elif len(approx) == 6:
            shape_label = "Hexagon"
        else:
            shape_label = "Circle"

        # Draw contour and add shape label to the set
        cv2.drawContours(image, [contour], 0, (0, 0, 0), 4)
        cv2.putText(image, shape_label, coords, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        unique_labels.add(shape_label)

    # Draw unique shape labels on the image
    for label in unique_labels:
        cv2.putText(image, label, (10, 50 + 30 * Idx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    _, img_encoded = cv2.imencode(".jpg", image)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")