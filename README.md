# Dog vs Cat Classifier 🐶🐱

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images as either **dog** or **cat**.
It trains the model on a dataset of dog and cat images, saves the model, and allows you to predict new images.

---

## 📂 **Project Structure**

```
ML GLOB/
├── dog-vs-cat/               # Folder containing training images (organized in subfolders for each class)
├── dog_vs_cat_model.h5       # Saved trained model (auto-generated after training)
├── 1.jpg                     # Example test image
└── your_script.py             # Your Python code (this code)
```

---

## ⚙ **How it works**

* If a trained model exists (`dog_vs_cat_model.h5`), it loads the model.
* If not, it:

  * Loads and splits the dataset (90% training, 10% validation).
  * Builds a CNN with multiple convolutional + pooling layers.
  * Trains the model for 10 epochs.
  * Saves the trained model.
* Finally, it loads a test image, predicts whether it’s a dog or a cat, and displays the image with the prediction and confidence score.

---

## 🚀 **Requirements**

Install the required packages:

```bash
pip install tensorflow matplotlib numpy pillow
```

---

## 🖼 **Input data**

Place your dataset in:

```
C:/Users/nihal/PycharmProjects/ML GLOB/ML GLOB/dog-vs-cat
```

Expected folder structure:

```
dog-vs-cat/
├── Dog/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
└── Cat/
    ├── cat1.jpg
    ├── cat2.jpg
    └── ...
```

Place your test image (e.g. `1.jpg`) at:

```
C:/Users/nihal/PycharmProjects/ML GLOB/ML GLOB/1.jpg
```

---

## 📝 **Usage**

Run the script:

```bash
python your_script.py
```

✅ Replace `your_script.py` with your actual Python filename.
✅ The script will train or load the model, predict the test image, and display the result.

---

## ⚠ **Note**

* The file paths are hardcoded for local Windows environment. Update them if running elsewhere.
* The model uses binary crossentropy loss and sigmoid activation for binary classification.

---

## ✨ **Example output**

You will see your test image displayed with a title like:

```
Prediction: Dog (0.95)
```

where `0.95` is the confidence score.

---

## 📌 **Author**

Nihal Yerra
