from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from sentence_transformers import SentenceTransformer

image_model = Sequential()
image_model.add(ResNet50(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3)))
image_model.add(GlobalAveragePooling2D())

print(image_model.summary())
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
print(bert_model)
def cosine_similarity_arranged(df, arr, feature_space):
    dist = []
    for i in range(feature_space.shape[0]):
        sim = np.dot(arr, feature_space[i])/(np.linalg.norm(arr)*np.linalg.norm(feature_space[i]))
        dist.append([i, sim])
    dist = sorted(dist, key = lambda x:x[1])
    dist = np.array(dist)
    return df.iloc[dist[-8:, 0]][::-1]


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "AdoptIt\\static\\Images\\"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dogs', methods = ['GET', 'POST'])
def get_recommendations():
    df_dogs = pd.read_csv("Dogs_From_MSPCA.csv")
    df_dogs["Description"] = list(map(str, df_dogs["Description"].values))
    df_dogs["Desc_Len"] = list(map(len, df_dogs["Description"].values))
    df_dogs = df_dogs[df_dogs["Desc_Len"] > 3]
    df_dogs = df_dogs.reset_index()
    if request.method == 'POST':
        f = request.files['dog_image']
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_FOLDER'] + filename)
    dog_img = app.config['UPLOAD_FOLDER'] + filename

    dog_img_f = image.load_img(dog_img, target_size=(224, 224))
    dog_img_np = image.img_to_array(dog_img_f)
    dog_img_np = np.expand_dims(dog_img_np, axis = 0)
    image_features = image_model.predict(dog_img_np)
    dog_desc = request.form.get('dog_desc')

    text_embeddings_features = bert_model.encode(dog_desc)

    combined_features = np.concatenate((image_features, text_embeddings_features), axis=None)
    feature_space = np.load("AdoptIt\\dog_combined_features.npy")

    res_df = cosine_similarity_arranged(df_dogs, combined_features, feature_space)
    print(res_df)
    records = res_df.to_dict(orient='records')
    for record in records:
        record['index'] = str(record['index'])
    return render_template('dogs.html', records = records)


if __name__ == '__main__':
    app.run(debug = True)