from django.http import HttpResponse,JsonResponse

from django.core.files.storage import FileSystemStorage

from django.views.decorators.csrf import csrf_exempt

from keras.preprocessing.image import img_to_array



import tensorflow as tf
import pickle
import numpy as np
import keras
import os
import cv2

default_image_size = tuple((256, 256))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            image=image/255
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

@csrf_exempt
def predict(req):
    from predictor.urls import model,labels
    imageFile = req.FILES['plant_image']
    try:
        os.chdir(os.getcwd()+"/predictor")
    except:
        print("Already in directory")
    fs=FileSystemStorage()
    filename = fs.save(imageFile.name, imageFile)
    file_url = fs.url(filename)
    data = convert_image_to_array(imageFile.name)
    data = data.reshape((256,256,3))
    
    print(labels)

    disease=labels[np.argmax(model.predict(np.array([data])))]
    
    os.remove(filename)
    res=JsonResponse({"dis":disease})
    res["Access-Control-Allow-Origin"] = "*"
    return res
