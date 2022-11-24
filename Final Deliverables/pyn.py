
import os
import numpy as np import pandas as pd
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image from werkzeug.utils import secure_filename

from flask import Flask, render_template, request app = Flask( name )
#load both the vegetable and fruit models model = load_model("vegetable.h5") model1=load_model("fruit.h5")

#home page @app.route('/') def home():
return render_template('home.html')

#prediction page @app.route('/prediction') def prediction():
return render_template('predict.html')

@app.route('/predict',methods=['POST']) def predict():
if request.method == 'POST':
# Get the file from post request f = request.files['image']

# Save the file to ./uploads
basepath = os.path.dirname( file ) file_path = os.path.join basepath, 'uploads', secure_filename(f.filename)) f.save(file_path)
img = image.load_img(file_path, target_size=(128, 128))
x = image.img_to_array(img) x = np.expand_dims(x, axis=0)
plant=request.form['plant'] print(plant) if(plant=="vegetable"):
preds = model.predict(x) preds=np.argmax(preds) print(preds)
df=pd.read_excel('precautions - veg.xlsx') print(df.iloc[preds]['caution'])
else:
preds = model1.predict(x) preds=np.argmax(preds) df=pd.read_excel('precautions - fruits.xlsx') print(df.iloc[preds]['caution'])
return df.iloc[preds]['caution'] if  name  == " main ":
app.run(debug=False)
