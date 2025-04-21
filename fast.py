# from fastapi import FastAPI, File, UploadFile
# import numpy as np
# import cv2
# import tensorflow as tf
# from PIL import Image
# import io

# app = FastAPI()

# # تحميل النموذج
# model = tf.keras.models.load_model(r"C:\Users\DELL\Downloads\model9.h5")  # غيّر الاسم لو مختلف

# # تعريف الكلاسات
# categories = [
#     'Myocardial Infarction Patients',
#     'Patient with History of Myocardial Infarction',
#     'Abnormal Heartbeat',
#     'Normal Person'
# ]

# # دالة لعمل preprocessing والتنبؤ
# async def preprocess_and_predict(image_bytes, model):
#     try:
#         # تحميل الصورة وتحويلها إلى RGB
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         image = np.array(image)

#         # تغيير الحجم
#         resized_img = cv2.resize(image, (128, 128))

#         # تطبيع وتحويل إلى batch
#         input_img = resized_img.reshape(1, 128, 128, 3) / 255.0

#         # التنبؤ
#         prediction = model.predict(input_img)
#         predicted_index = np.argmax(prediction)
#         confidence = float(np.max(prediction))

#         return {
#             "predicted_class": categories[predicted_index],
#             "confidence": confidence
#         }

#     except Exception as e:
#         return {"error": str(e)}

# # نقطة النهاية لاستقبال الصورة والتنبؤ
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     result = preprocess_and_predict(contents, model)
#     return result
# # uvicorn fast:app --reload
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import numpy as np
# import cv2
# import tensorflow as tf
# from PIL import Image
# import io
# import base64

# app = FastAPI()

# # تحميل النموذج
# model = tf.keras.models.load_model(r"C:\Users\DELL\Downloads\model9.h5")  # غيّر الاسم لو مختلف

# # تعريف الكلاسات
# categories = [
#     'Myocardial Infarction Patients',
#     'Patient with History of Myocardial Infarction',
#     'Abnormal Heartbeat',
#     'Normal Person'
# ]

# # دالة لعمل preprocessing والتنبؤ
# async def preprocess_and_predict(image_bytes, model):
#     try:
#         # تحميل الصورة وتحويلها إلى RGB
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         image = np.array(image)

#         # تغيير الحجم
#         resized_img = cv2.resize(image, (128, 128))

#         # تطبيع وتحويل إلى batch
#         input_img = resized_img.reshape(1, 128, 128, 3) / 255.0

#         # التنبؤ
#         prediction = model.predict(input_img)
#         predicted_index = np.argmax(prediction)
#         confidence = float(np.max(prediction))

#         return {
#             "predicted_class": categories[predicted_index],
#             "confidence": confidence
#         }

#     except Exception as e:
#         return {"error": str(e)}

# # نقطة النهاية لاستقبال صورة بتنسيق base64 والتنبؤ
# @app.post("/predict")
# async def predict(file: str):
#     try:
#         # فك تشفير base64 إلى بايتات
#         image_bytes = base64.b64decode(file)
#         result = await preprocess_and_predict(image_bytes, model)
#         return JSONResponse(content=result)
    
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})

# لاختبار: 
# يجب إرسال صورة بتنسيق base64 في body عند استخدام هذه الدالة.
# على سبيل المثال:
# {
#     "file": "<الصورة بتنسيق base64>"
# }

# لتشغيل الخادم:
# uvicorn main:app --reload
# from fastapi import FastAPI
# from fastapi.responses import JSONResponse
# import numpy as np
# import tensorflow as tf
# import cv2
# from PIL import Image
# import base64
# import io
# import matplotlib.pyplot as plt

# app = FastAPI()

# # تحميل النموذج
# model = tf.keras.models.load_model(r"C:\Users\DELL\Downloads\model9.h5")

# # أسماء الفئات
# categories = [
#     'Myocardial Infarction Patients',
#     'Patient with History of Myocardial Infarction',
#     'Abnormal Heartbeat',
#     'Normal Person'
# ]

# # كلاس GradCAM
# class GradCAM:
#     def __init__(self, model, classIdx, layerName=None):
#         self.model = model
#         self.classIdx = classIdx
#         self.layerName = layerName or self.find_target_layer()

#     def find_target_layer(self):
#         for layer in reversed(self.model.layers):
#             if len(layer.output.shape) == 4:
#                 return layer.name
#         raise ValueError("No 4D layer found for Grad-CAM.")

#     def compute_heatmap(self, image, eps=1e-8):
#         gradModel = tf.keras.models.Model(
#             inputs=[self.model.inputs],
#             outputs=[self.model.get_layer(self.layerName).output, self.model.output]
#         )
#         with tf.GradientTape() as tape:
#             inputs = tf.cast(image, tf.float32)
#             (convOutputs, predictions) = gradModel(inputs)
#             loss = predictions[:, self.classIdx]

#         grads = tape.gradient(loss, convOutputs)
#         castConvOutputs = tf.cast(convOutputs > 0, "float32")
#         castGrads = tf.cast(grads > 0, "float32")
#         guidedGrads = castConvOutputs * castGrads * grads

#         convOutputs = convOutputs[0]
#         guidedGrads = guidedGrads[0]

#         weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
#         cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

#         (w, h) = (image.shape[2], image.shape[1])
#         heatmap = cv2.resize(cam.numpy(), (w, h))

#         heatmap = np.maximum(heatmap, 0)
#         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
#         heatmap = np.uint8(255 * heatmap)

#         return heatmap

#     def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
#         heatmap_color = cv2.applyColorMap(heatmap, colormap)
#         output = cv2.addWeighted(image, alpha, heatmap_color, 1 - alpha, 0)
#         return output

# # الدالة الرئيسية للمعالجة والتنبؤ
# async def preprocess_and_predict(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     orig = np.array(image)
#     resized_img = cv2.resize(orig, (128, 128))
#     input_img = resized_img.reshape(1, 128, 128, 3) / 255.0

#     # التنبؤ
#     prediction = model.predict(input_img)
#     predicted_index = int(np.argmax(prediction))
#     confidence = float(np.max(prediction))

#     # Grad-CAM
#     cam = GradCAM(model, predicted_index)
#     heatmap = cam.compute_heatmap(input_img)

#     heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
#     heatmap_overlay = cam.overlay_heatmap(heatmap_resized, orig, alpha=0.5)

#     # تحويل الصورة الناتجة إلى base64
#     _, buffer = cv2.imencode('.jpg', heatmap_overlay)
#     heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

#     return {
#         "predicted_class": categories[predicted_index],
#         "confidence": confidence,
#         "heatmap_image_base64": heatmap_base64
#     }

# # # نقطة النهاية الرئيسية
# # @app.post("/predict")
# # async def predict_base64(file: str):
# #     try:
# #         image_bytes = base64.b64decode(file)
# #         result = await preprocess_and_predict(image_bytes)
# #         return JSONResponse(content=result)
# #     except Exception as e:
# #         return JSONResponse(status_code=400, content={"error": str(e)})
# from pydantic import BaseModel

# class ImageData(BaseModel):
#     file: str

# @app.post("/predict")
# async def predict_base64(data: ImageData):
#     try:
#         image_bytes = base64.b64decode(data.file)
#         result = await preprocess_and_predict(image_bytes)
#         return JSONResponse(content=result)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})
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


    
# fast.py
# from fastapi import FastAPI, File, UploadFile

# app = FastAPI()

# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     # نعيد اسم الملف فقط للتجربة
#     return {"filename": file.filename}
