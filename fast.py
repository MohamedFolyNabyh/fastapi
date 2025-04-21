
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64

app = FastAPI()

# تحميل النموذج
model = tf.keras.models.load_model(r"C:\Users\DELL\Downloads\model9.h5")

# أسماء الفئات
categories = [
    'Myocardial Infarction Patients',
    'Patient with History of Myocardial Infarction',
    'Abnormal Heartbeat',
    'Normal Person'
]

# GradCAM class
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName or self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("No 4D layer found for Grad-CAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
        heatmap = np.uint8(255 * heatmap)

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap_color = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap_color, 1 - alpha, 0)
        return output

# الدالة الرئيسية للمعالجة والتنبؤ
async def preprocess_and_predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig = np.array(image)
    resized_img = cv2.resize(orig, (128, 128))
    input_img = resized_img.reshape(1, 128, 128, 3) / 255.0

    prediction = model.predict(input_img)
    predicted_index = int(np.argmax(prediction))
   

    cam = GradCAM(model, predicted_index)
    heatmap = cam.compute_heatmap(input_img)

    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_overlay = cam.overlay_heatmap(heatmap_resized, orig, alpha=0.5)

    _, buffer = cv2.imencode('.jpg', heatmap_overlay)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "predicted_class": categories[predicted_index],
        "heatmap_image_base64": heatmap_base64
    }

# نقطة النهاية لاستقبال صورة مباشرة
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = await preprocess_and_predict(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


  
