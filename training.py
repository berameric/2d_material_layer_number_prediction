import streamlit as st
import numpy as np
from skimage import io, transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import os

# RGB to Luv conversion functions
def linearize_rgb(rgb):
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return rgb

def rgb_to_xyz(rgb_linear):
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return np.dot(rgb_linear, M.T)

def rgb_to_xyz_illuminant_a(rgb):
    rgb_linear = linearize_rgb(rgb)





    matrix = np.array([
        [0.4002, 0.7075, -0.0808],
        [-0.2263, 1.1653, 0.0457],
        [0.0000, 0.0000, 0.8253]
    ])





    return np.dot(rgb_linear, matrix.T)

def xyz_to_luv(xyz, reference_white):
    xyz = np.maximum(xyz, 1e-6)  # Avoid division by zero
    Xr, Yr, Zr = reference_white

    L = np.where(xyz[:,:,1]/Yr > 216/24389,
                 116 * (xyz[:,:,1]/Yr)**(1/3) - 16,
                 24389/27 * xyz[:,:,1]/Yr)

    d = xyz[:,:,0] + 15 * xyz[:,:,1] + 3 * xyz[:,:,2]
    ur_prime = 4 * Xr / (Xr + 15 * Yr + 3 * Zr)
    vr_prime = 9 * Yr / (Xr + 15 * Yr + 3 * Zr)

    u_prime = 4 * xyz[:,:,0] / d
    v_prime = 9 * xyz[:,:,1] / d

    u = 13 * L * (u_prime - ur_prime)
    v = 13 * L * (v_prime - vr_prime)

    return np.dstack((L, u, v))

def rgb_to_luv(rgb):
    reference_white_A = np.array([1.09850, 1.00000, 0.35585])
    xyz = rgb_to_xyz_illuminant_a(rgb)
    return xyz_to_luv(xyz, reference_white_A)

def convert_image_rgb_to_luv(image_path):
    img = Image.open(image_path)
    rgb_array = np.array(img).astype(np.float32) / 255.0
    luv_array = rgb_to_luv(rgb_array)
    return luv_array

def open_image(image_path):
    return io.imread(image_path)

def create_mask(image, color, tolerance):
    return np.all((image >= color - tolerance) & (image <= color + tolerance), axis=-1).astype(int)

def most_common_color(image):
    image = image.reshape(-1, 3)
    mask = ~((image == [255, 255, 255]).all(axis=1) | (image == [0, 0, 0]).all(axis=1))
    image = image[mask]

    red, green, blue = image[:, 0], image[:, 1], image[:, 2]

    hist_r = np.histogram(red[red != 0], bins=256, range=(0, 256))
    hist_g = np.histogram(green[green != 0], bins=256, range=(0, 256))
    hist_b = np.histogram(blue[blue != 0], bins=256, range=(0, 256))

    hist_r_filtered = hist_r[0]
    hist_g_filtered = hist_g[0]
    hist_b_filtered = hist_b[0]

    hist_r_normalized = hist_r_filtered / np.sum(hist_r_filtered)
    hist_g_normalized = hist_g_filtered / np.sum(hist_g_filtered)
    hist_b_normalized = hist_b_filtered / np.sum(hist_b_filtered)

    weighted_mean_red = np.average(hist_r[1][:-1], weights=hist_r_normalized)
    weighted_mean_green = np.average(hist_g[1][:-1], weights=hist_g_normalized)
    weighted_mean_blue = np.average(hist_b[1][:-1], weights=hist_b_normalized)

    return (round(weighted_mean_red), round(weighted_mean_green), round(weighted_mean_blue))

def iterative_weighted_mean(delta_e, iterations=10, bins=100, hist_range=(1, 100)):
    hist, bin_edges = np.histogram(delta_e, bins=bins, range=hist_range)
    hist_filtered = hist.copy()
    weighted_means = []

    for _ in range(iterations):
        hist_normalized = hist_filtered / np.sum(hist_filtered)
        weighted_mean = np.average(bin_edges[:-1], weights=hist_normalized)
        weighted_means.append(weighted_mean)
        most_common_index = np.argmax(hist_filtered)
        hist_filtered[most_common_index] = 0

    hist_filtered = hist.copy()

    for _ in range(iterations):
        hist_normalized = hist_filtered / np.sum(hist_filtered)
        weighted_mean = np.average(bin_edges[:-1], weights=hist_normalized)
        weighted_means.append(weighted_mean)
        least_common_index = np.argmin(hist_filtered + (hist_filtered == 0) * np.inf)
        hist_filtered[least_common_index] = 0

    mean = np.mean(bin_edges)
    return weighted_means, mean

def process_images(masked_image, original_image, material_name, substrate_name):
    # Make sure original image matches mask size
    if masked_image.shape != original_image.shape:
        original_image = transform.resize(original_image, masked_image.shape,
                                          anti_aliasing=True,
                                          preserve_range=True).astype(masked_image.dtype)

    # [Include the mask creation code here]
    monolayer_mask_color = np.array([0, 0, 254])
    bilayer_mask_color = np.array([254, 0, 0])
    trilayer_mask_color = np.array([0, 255, 3])
    fourlayer_mask_color = np.array([255, 156, 0])
    five_layer_mask_color = np.array([183, 1, 254])
    sixlayer_mask_color = np.array([255, 255, 0])
    not_segmented_color = np.array([255, 255, 255])

    tolerance = 10

    masks = {
        "1": create_mask(masked_image, monolayer_mask_color, tolerance),
        "2": create_mask(masked_image, bilayer_mask_color, tolerance),
        "3": create_mask(masked_image, trilayer_mask_color, tolerance),
        "4": create_mask(masked_image, fourlayer_mask_color, tolerance),
        "5": create_mask(masked_image, five_layer_mask_color, tolerance),
        "6": create_mask(masked_image, sixlayer_mask_color, tolerance),
        "not_segmented": create_mask(masked_image, not_segmented_color, tolerance)
    }

    # [Include the data processing code here]
    variables_b = [[], [], []]
    targets_b = []

    substrate_area = original_image * (1 - (masks["1"] + masks["2"] + masks["3"] + masks["4"] + masks["5"] + masks["6"] + masks["not_segmented"])[:, :, np.newaxis])
    substrate_color = np.array(most_common_color(substrate_area))
    substrate_color_luv = rgb_to_luv(substrate_color[np.newaxis, np.newaxis, :] / 255)[0, 0]
    xyz_tristimulus_values = rgb_to_xyz_illuminant_a(substrate_color[np.newaxis, np.newaxis, :] / 255)[0, 0]

    for key, mask in masks.items():
        if key == "not_segmented" or np.sum(mask) == 0:
            continue

        layer_area = original_image * mask[:, :, np.newaxis]
        black_areas_mask = (layer_area[:, :, 0] == 0) & (layer_area[:, :, 1] == 0) & (layer_area[:, :, 2] == 0)
        layer_area[layer_area.sum(axis=2) == 0] = substrate_color
        layer_area_luv = rgb_to_luv(layer_area / 255)
        layer_area_xyz = rgb_to_xyz_illuminant_a(layer_area / 255)
        layer_area_luv = layer_area_luv[~black_areas_mask]
        layer_area_xyz = layer_area_xyz[~black_areas_mask]
        layer_area_xyz = layer_area_xyz.reshape(-1, 3)
        layer_area_luv = layer_area_luv.reshape(-1, 3)

        l_part = layer_area_luv[:, 0]
        u_part = layer_area_luv[:, 1]
        v_part = layer_area_luv[:, 2]

        delta_e_l_squared = (l_part** 2 - substrate_color_luv[0]** 2)
        delta_e_u_squared = (u_part** 2 - substrate_color_luv[1]** 2)
        delta_e_v_squared = (v_part** 2 - substrate_color_luv[2]** 2)

        delta_e_l = np.sqrt(delta_e_l_squared)
        delta_e_u = np.sqrt(delta_e_u_squared)
        delta_e_v = np.sqrt(delta_e_v_squared)

        variables_b[0].extend(delta_e_l)
        variables_b[1].extend(delta_e_u)
        variables_b[2].extend(delta_e_v)

        targets_b.extend([int(key)] * len(delta_e_l))

    variables_a = np.array(variables_b).T
    targets_a = np.array(targets_b)

    df_a = pd.DataFrame({'delta_e_L': variables_a[:,0], 'delta_e_u': variables_a[:,1], 'delta_e_v': variables_a[:,2], 'layer': targets_a})

    df_a['delta_e_L_squared'] = df_a['delta_e_L'] ** 2
    df_a['delta_e_u_squared'] = df_a['delta_e_u'] ** 2
    df_a['delta_e_v_squared'] = df_a['delta_e_v'] ** 2

    X_a = df_a[['delta_e_L', 'delta_e_L_squared', 'delta_e_u', 'delta_e_u_squared', 'delta_e_v', 'delta_e_v_squared']]
    y_a = df_a['layer']

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.20, random_state=42, stratify=y_a, shuffle=True)

    rf_model_a = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42
    )

    rf_model_a.fit(X_train_a, y_train_a)

    # Save the model
    model_name = f"{material_name}-{substrate_name}"
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model_a, f"models/{model_name}.joblib")

    # Get predictions and classification report
    y_pred_a = rf_model_a.predict(X_test_a)
    report = classification_report(y_test_a, y_pred_a, output_dict=True)

    return report, y_test_a, y_pred_a, model_name

def main():
    st.title("Flake Analysis App")

    # User inputs
    material_name = st.text_input("Enter material name:")
    substrate_name = st.text_input("Enter substrate name:")

    # File uploader for masked image
    masked_image_file = st.file_uploader("Upload masked image", type=["jpg", "png"])

    # File uploader for original image
    original_image_file = st.file_uploader("Upload original image", type=["jpg", "png"])

    # Add a button to trigger the processing
    if st.button("Process Images"):
        if masked_image_file is not None and original_image_file is not None and material_name and substrate_name:
            with st.spinner("Processing images..."):
                masked_image = open_image(masked_image_file)
                original_image = open_image(original_image_file)

                report, y_test_a, y_pred_a, model_name = process_images(masked_image, original_image, material_name, substrate_name)

                # Display results
                st.subheader("Model Performance")
                st.write(f"Model saved as: models/{model_name}.joblib")

                # Display classification report
                st.write("Classification Report:")
                st.table(pd.DataFrame(report).transpose())

                # Create and display confusion matrix
                cm = confusion_matrix(y_test_a, y_pred_a)
                class_labels = sorted(set(y_test_a) | set(y_pred_a))
                cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
                cm_percent = cm_df.div(cm_df.sum(axis=1), axis=0) * 100

                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cm_df, annot=False, cmap="YlOrBr", ax=ax)

                for i in range(len(cm)):
                    for j in range(len(cm)):
                        text = f"{cm[i, j]}\n{cm_percent.iloc[i, j]:.1f}%"
                        ax.text(j + 0.5, i + 0.5, text, ha='center', va='center')

                ax.set_ylabel('Output Class')
                ax.set_xlabel('Target Class')
                ax.set_title('Confusion Matrix')

                for i in range(len(cm)):
                    correct = cm[i, i]
                    total = np.sum(cm[i, :])
                    accuracy = correct / total * 100
                    ax.text(len(cm), i + 0.5, f'{accuracy:.1f}%\n{100-accuracy:.1f}%',
                            ha='left', va='center')

                for i in range(len(cm)):
                    correct = cm[i, i]
                    total = np.sum(cm[:, i])
                    accuracy = correct / total * 100
                    ax.text(i + 0.5, len(cm), f'{accuracy:.1f}%\n{100-accuracy:.1f}%',
                            ha='center', va='top')

                overall_accuracy = np.trace(cm) / np.sum(cm) * 100
                ax.text(len(cm), len(cm), f'{overall_accuracy:.1f}%\n{100-overall_accuracy:.1f}%',
                        ha='left', va='top')

                st.pyplot(fig)
        else:
            st.error("Please upload both images and enter material and substrate names before processing.")

if __name__ == "__main__":
    main()
