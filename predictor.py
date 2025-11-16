import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2

# Load model
model = load_model('model/brain_tumor_model.h5')

# Preprocessing
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # adjust size if different
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

# Prediction
def predict_tumor(img_path):
    x = preprocess(img_path)
    preds = model.predict(x)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)
    return pred_class, confidence

# Grad-CAM (optional — copy your Colab’s GradCAM logic here)
def generate_gradcam(img_path, layer_name='block5_conv3'):
    img = preprocess(img_path)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)

        # Handle list-like outputs safely
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Create heatmap
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (224, 224))       # ✅ fixed line
    heatmap = np.uint8(255 * heatmap)

    # Superimpose heatmap on original image
    img_orig = cv2.imread(img_path)
    img_orig = cv2.resize(img_orig, (224, 224))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_orig, 0.5, heatmap, 0.8, 0)

    output_path = 'gradcam_output.jpg'
    cv2.imwrite(output_path, superimposed_img)
    return output_path



