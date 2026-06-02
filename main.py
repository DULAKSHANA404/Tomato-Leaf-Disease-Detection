from fastapi import FastAPI,Request,File,UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import os

import kagglehub
import os

os.environ['KAGGLEHUB_CACHE'] = "./models/"
path = kagglehub.model_download("domindu3/tomato-leaf-disease-detecter/tensorFlow2/default")
print(f"model downloaded to {path}")

app = FastAPI(title='none')

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

model_path = os.path.join(path,os.listdir(path)[0])
labels = {0: 'Tomato___Bacterial_spot',
 1: 'Tomato___Early_blight',
 2: 'Tomato___Late_blight',
 3: 'Tomato___Leaf_Mold',
 4: 'Tomato___Septoria_leaf_spot',
 5: 'Tomato___Spider_mites Two-spotted_spider_mite',
 6: 'Tomato___Target_Spot',
 7: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 8: 'Tomato___Tomato_mosaic_virus',
 9: 'Tomato___healthy'}

@app.get('/')
async def home_page(responce:Request):
    return templates.TemplateResponse(request=responce,name='home.html') 

class Model():

    def __init__(self,model_path =model_path):
        self.W,self.H,self.D = 128,128,3
        try:
            self.model = tf.keras.models.load_model(model_path)
            print('Model loaded')
        except Exception as e:
            print(f'Model not loaded {e}')
            raise

    def preprocess(self,img):
        img = tf.io.decode_jpeg(img,channels=self.D)
        img = tf.image.resize(img,[self.H,self.W])
        img = tf.cast(img,dtype=tf.float32)

        return img

    def get_result(self,img):
        pred = self.model.predict(tf.expand_dims(img,axis=0))
        result = np.argmax(pred)
        label = labels[result]
        conf = pred[0][result]
        print(conf)

        return label,conf*100

model = Model()

@app.post('/result')
async def result_page(request:Request,img:UploadFile=File(...)):

    img_path = os.path.join('./static',"user_img.jpeg")

    img_renderd = await img.read()

    with open(img_path,'wb') as f:
        f.write(img_renderd)

    img = model.preprocess(img_renderd)
    
    result,confidence = model.get_result(img)

    return templates.TemplateResponse(request=request,name="result.html",context={"label":result,"img":img_path,"conf":f"{confidence}"})