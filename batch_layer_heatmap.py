# simple_densenet_gradcam.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def simple_densenet_gradcam(model_path, image_path, output_dir="simple_results"):
    """Simple Grad-CAM for DenseNet Alzheimer's model"""
    
    # Load model and image
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    
    # Get prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    
    print(f"Prediction: Class {class_idx}, Confidence: {confidence:.3f}")
    
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        print("No convolutional layer found!")
        return
    
    print(f"Using layer: {last_conv_layer.name}")
    
    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute Grad-CAM
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    # Create visualization
    os.makedirs(output_dir, exist_ok=True)
    
    heatmap = heatmap.numpy()
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], 
                                    [img.height, img.width]).numpy().squeeze()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Grad-CAM Heatmap\n{last_conv_layer.name}")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    plt.title("Overlay (Alpha=0.5)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gradcam_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Grad-CAM result saved to: {output_dir}/gradcam_result.png")

# Run the simple version
if __name__ == "__main__":
    simple_densenet_gradcam(
        model_path="models/Early_Alzheimers_DenseNet121_Augmented.keras",
        image_path="test/alz/cn/cn_ADNI_002_S_4225_MR_MT1__N3m_Br_20120420150648868_S147169_I299292_s113.png",  # Replace with your image
        output_dir="simple_gradcam_results"
    )