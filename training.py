import streamlit as st
import numpy as np
from skimage import io as ios, transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import io
import cv2 as cv


def linearize_rgb(rgb):
    rgb = rgb / 255.0  # 0-255 ölçeğini 0-1'e çevir
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def rgb_to_xyz_illuminant_a(rgb):
    rgb_linear = linearize_rgb(rgb)
    matrix = np.array([[0.4965, 0.2520, 0.6000],
                       [0.2560, 0.5040, 0.2400],
                       [0.0233, 0.0840, 3.1600]])
    if rgb_linear.ndim == 1:  # Tek piksel
        return np.dot(rgb_linear, matrix)
    else:  # Görüntü: [height, width, 3]
        return np.einsum('ijk,kl->ijl', rgb_linear, matrix)

def xyz_to_luv(xyz, reference_white):
    xyz = np.maximum(xyz, 1e-6)
    Xr, Yr, Zr = reference_white
    Y_ratio = xyz[..., 1] / Yr
    L = np.where(Y_ratio > (6/29)**3,
                 116 * Y_ratio**(1/3) - 16,
                 (29/3)**3 * Y_ratio / 27)

    d = xyz[..., 0] + 15 * xyz[..., 1] + 3 * xyz[..., 2]
    d = np.maximum(d, 1e-6)
    ur_prime = 4 * Xr / (Xr + 15 * Yr + 3 * Zr)
    vr_prime = 9 * Yr / (Xr + 15 * Yr + 3 * Zr)

    u_prime = 4 * xyz[..., 0] / d
    v_prime = 9 * xyz[..., 1] / d

    u = 13 * L * (u_prime - ur_prime)
    v = 13 * L * (v_prime - vr_prime)

    return np.stack((L, u, v), axis=-1)

def rgb_to_luv(rgb):
    reference_white_A = np.array([1.09850, 1.00000, 0.35585])
    xyz = rgb_to_xyz_illuminant_a(rgb)
    return xyz_to_luv(xyz, reference_white_A)

def open_image(image_file):
    return ios.imread(image_file)

def create_mask(image, color, tolerance):
    lower_bound = color - tolerance
    upper_bound = color + tolerance
    mask = np.logical_and(image >= lower_bound, image <= upper_bound)
    return np.all(mask, axis=-1).astype(int)

def most_common_color(image):
    image = image.reshape(-1, 3)
    mask = ~((image == [255, 255, 255]).all(axis=1) | (image == [0, 0, 0]).all(axis=1))
    image = image[mask]
    
    hist, edges = np.histogramdd(image, bins=256, range=((0, 256), (0, 256), (0, 256)))
    weighted_mean = [np.average(edges[i][:-1], weights=np.sum(hist, axis=tuple(j for j in range(3) if j != i)))
                     for i in range(3)]
    return tuple(map(round, weighted_mean))

def process_single_image(masked_image, original_image):

    if masked_image.shape != original_image.shape:
        original_image = transform.resize(original_image, masked_image.shape,
                                       anti_aliasing=True,
                                       preserve_range=True).astype(masked_image.dtype)
    



    # Define mask colors
    mask_colors = {
        "1": np.array([0, 0, 254]),
        "2": np.array([254, 0, 0]),
        "3": np.array([0, 255, 3]),
        "4": np.array([255, 156, 0]),
        "6": np.array([255, 255, 0]),
        "5": np.array([200, 1, 254]),
        "not_segmented": np.array([255, 255, 255])
    }

    tolerance = 10
    masks = {key: create_mask(masked_image, color, tolerance) 
             for key, color in mask_colors.items()}

    # Calculate substrate area
    all_masks = sum(masks[key] for key in masks if key != "not_segmented")
    substrate_area = original_image * (1 - all_masks[:, :, np.newaxis])
    substrate_color = np.array(most_common_color(substrate_area))
    substrate_color_luv = rgb_to_luv(substrate_color[np.newaxis, np.newaxis, :] / 255)[0, 0]

    variables = []
    targets = []

    for key, mask in masks.items():
        if key == "not_segmented" or np.sum(mask) == 0:
            continue

        layer_area = original_image * mask[:, :, np.newaxis]
        black_areas_mask = np.all(layer_area == 0, axis=2)
        layer_area[black_areas_mask] = substrate_color

        layer_area_luv = rgb_to_luv(layer_area / 255)
        layer_area_luv = layer_area_luv[~black_areas_mask]
        
        if len(layer_area_luv) == 0:
            continue

        delta_e = layer_area_luv - substrate_color_luv
        variables.extend(delta_e)
        targets.extend([int(key)] * len(delta_e))

    return variables, targets

def process_multiple_images(masked_images, original_images, material_name, substrate_name):
    all_variables = []
    all_targets = []

    for masked_image, original_image in zip(masked_images, original_images):
        variables, targets = process_single_image(masked_image, original_image)
        all_variables.extend(variables)
        all_targets.extend(targets)

    variables_array = np.array(all_variables)
    targets_array = np.array(all_targets)

    df = pd.DataFrame({
        'delta_e_L': variables_array[:,0],
        'delta_e_u': variables_array[:,1],
        'delta_e_v': variables_array[:,2],
        'layer': targets_array
    })

    X = df[['delta_e_L',"delta_e_u","delta_e_v"]]
    y = df['layer']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.75,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report['feature_importances'] = dict(zip(X.columns, rf_model.feature_importances_))

    model_name = f"{material_name}-{substrate_name}.joblib"
    model_bytes = io.BytesIO()
    joblib.dump(rf_model, model_bytes)
    model_bytes.seek(0)

    dataset_filename = f"{material_name}-{substrate_name}-dataset.csv"
    df.to_csv(dataset_filename, index=False)

    return report, y_test, y_pred, model_name, model_bytes

def main():
    st.title("2D Material Flake Analysis App")

    material_name = st.text_input("Enter material name:")
    substrate_name = st.text_input("Enter substrate name:")

    masked_image_files = st.file_uploader("Upload masked images", type=["jpg", "png"], accept_multiple_files=True)
    original_image_files = st.file_uploader("Upload original images", type=["jpg", "png"], accept_multiple_files=True)

    if st.button("Process Images"):
        if (masked_image_files and original_image_files and 
            len(masked_image_files) == len(original_image_files) and
            material_name and substrate_name):
            
            with st.spinner("Processing images..."):
                masked_images = [open_image(f) for f in masked_image_files]
                original_images = [open_image(f) for f in original_image_files]

                report, y_test, y_pred, model_name, model_bytes = process_multiple_images(
                    masked_images, original_images, material_name, substrate_name
                )

                st.subheader("Model Performance")
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name=model_name,
                    mime="application/octet-stream"
                )

                st.write("Classification Report:")
                st.table(pd.DataFrame(report).transpose())

                # Create confusion matrix visualization
                cm = confusion_matrix(y_test, y_pred)
                class_labels = sorted(set(y_test) | set(y_pred))
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

                # Add accuracy percentages
                for i in range(len(cm)):
                    correct = cm[i, i]
                    total = np.sum(cm[i, :])
                    accuracy = correct / total * 100
                    ax.text(len(cm), i + 0.5, f'{accuracy:.1f}%\n{100-accuracy:.1f}%',
                           ha='left', va='center')

                    total_col = np.sum(cm[:, i])
                    accuracy_col = correct / total_col * 100
                    ax.text(i + 0.5, len(cm), f'{accuracy_col:.1f}%\n{100-accuracy_col:.1f}%',
                           ha='center', va='top')

                overall_accuracy = np.trace(cm) / np.sum(cm) * 100
                ax.text(len(cm), len(cm), f'{overall_accuracy:.1f}%\n{100-overall_accuracy:.1f}%',
                       ha='left', va='top')

                st.pyplot(fig)
        else:
            st.error("Please upload equal numbers of masked and original images and enter material and substrate names.")

if __name__ == "__main__":
    main()
