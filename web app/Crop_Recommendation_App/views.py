from django.shortcuts import render
import os
import pickle
import numpy as np
# Create your views here.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = pickle.load(open(os.path.join(BASE_DIR, 'model/best_crop_recommendation_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'model/scaler.pkl'), 'rb'))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, 'model/label_encoder.pkl'), 'rb'))

def predict_crop(request):
    if request.method == 'POST':
        temp = float(request.POST['temperature'])
        hum = float(request.POST['humidity'])
        moist = float(request.POST['moisture'])
        nitro = float(request.POST['nitrogen'])
        potas = int(request.POST['potassium'])
        phos = int(request.POST['phosphorous'])

        data = np.array([[temp, hum, moist, nitro, potas, phos]])
        scaled = scaler.transform(data)
        prediction = model.predict(scaled)
        crop_name = label_encoder.inverse_transform([prediction[0]])[0]

        return render(request, 'form.html', {'prediction': crop_name})

    return render(request, 'form.html')
