from flask import Flask, render_template, request
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)


def load_model():
    class_names = [
        "african-wildcat",
        "blackfoot-cat",
        "chinese-mountain-cat",
        "domestic-cat",
        "european-wildcat",
        "jungle-cat",
        "sand-cat",
    ]

    global model
    model = Sequential()

    vgg = VGG16(include_top=False, weights="imagenet", input_shape=(227, 227, 3))

    vgg.trainable = True
    for layer in vgg.layers[:15]:
        layer.trainable = False

    model.add(vgg)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(class_names), activation="softmax"))

    opt = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=["accuracy"]
    )

    model.load_weights("./weights.h5")


load_model()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    file_path = "./static/images/file.jpg"
    if request.method == "POST":
        file = request.files["image"]
        if file:
            file.save(file_path)
            result = process_image((file_path))

            return render_template("predict.html", result=result)

    if os.path.exists(file_path):
        os.remove(file_path)
    return render_template("predict.html")


@app.route("/discover")
def discover():
    return render_template("discover.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # send_message(request)
        message = "Thank you for contacting us! We'll get back to you soon."
        return render_template("contact.html", message=message)

    return render_template("contact.html", message="")


@app.route("/faq")
def faq():
    return render_template("faq.html")


def send_message(request):
    name = request.form["firstname"]
    email = request.form["email"]
    message = request.form["message"]

    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = ""
    smtp_password = ""

    msg = MIMEMultipart()
    msg["From"] = smtp_username
    msg["To"] = smtp_username
    msg["Subject"] = "Contact Form Submission"

    body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
    msg.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_username, smtp_password)
    server.sendmail(smtp_username, smtp_username, msg.as_string())
    server.quit()


def process_image(file_path):
    class_names = [
        "african-wildcat",
        "blackfoot-cat",
        "chinese-mountain-cat",
        "domestic-cat",
        "european-wildcat",
        "jungle-cat",
        "sand-cat",
    ]

    img = load_img(file_path)
    img = img.resize((227, 227))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    class_idx = np.argmax(y_pred, axis=1)[0]
    class_name = class_names[class_idx]
    result = class_name.split("-")
    result = result[0].capitalize() + " " + result[1].capitalize()
    return result


if __name__ == "__main__":
    app.run(debug=True)
