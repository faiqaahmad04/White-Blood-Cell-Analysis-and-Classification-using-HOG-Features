import cv2
import numpy as np
import os
from collections import defaultdict
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# ----------------- HOG Parameters ----------------- #
HOG_IMG_SIZE = (200, 200)
HOG_WIN_SIZE = (128, 128)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
NBINS = 9

# ----------------- HOG Descriptor ----------------- #
hog_descriptor = cv2.HOGDescriptor(
    _winSize=HOG_WIN_SIZE,
    _blockSize=HOG_BLOCK_SIZE,
    _blockStride=HOG_BLOCK_STRIDE,
    _cellSize=HOG_CELL_SIZE,
    _nbins=NBINS
)


# ----------------- HOG Extraction for Color Images ----------------- #
def extract_hog_features(image):
    resized = cv2.resize(image, HOG_IMG_SIZE)
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)
    descriptors = []

    for channel in channels:
        channel_descriptors = []
        for y in range(0, channel.shape[0] - HOG_WIN_SIZE[1] + 1, 16):
            for x in range(0, channel.shape[1] - HOG_WIN_SIZE[0] + 1, 16):
                patch = channel[y:y + HOG_WIN_SIZE[1], x:x + HOG_WIN_SIZE[0]]
                if patch.shape == HOG_WIN_SIZE:
                    desc = hog_descriptor.compute(patch)
                    if desc is not None:
                        channel_descriptors.append(desc.flatten())
        if channel_descriptors:
            descriptors.append(np.concatenate(channel_descriptors))

    if descriptors:
        return np.concatenate(descriptors)
    else:
        return np.zeros((hog_descriptor.getDescriptorSize() * 3,))  # 3 channels fallback


# ----------------- PCA for Dimensionality Reduction ----------------- #
def apply_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


# ----------------- Visualize HOG Features with PCA ----------------- #
def visualize_hog_features_with_pca(train_dir):
    all_features = []
    labels = []

    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    features = extract_hog_features(img)
                    all_features.append(features)
                    labels.append(class_name)

    all_features = np.array(all_features)
    reduced_features = apply_pca(all_features, n_components=2)

    plt.figure(figsize=(8, 6))
    for class_name in set(labels):
        class_indices = np.where(np.array(labels) == class_name)[0]

        # Plot the reduced features for these images (belonging to this class)
        plt.scatter(reduced_features[class_indices, 0], reduced_features[class_indices, 1], label=class_name)

    plt.title("HOG Feature Visualization with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


# ----------------- Visualize One HOG Image per Class ----------------- #
def plot_hog_per_class(train_dir):
    classes = sorted(os.listdir(train_dir))
    plt.figure(figsize=(15, 8))

    for idx, class_name in enumerate(classes):
        class_path = os.path.join(train_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            continue

        img_path = os.path.join(class_path, image_files[0])
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, HOG_WIN_SIZE)

        features, hog_image = hog(
            gray_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            block_norm='L2-Hys'
        )
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        plt.subplot(2, 3, idx + 1)
        plt.imshow(hog_image, cmap='gray')
        plt.title(class_name)
        plt.axis('off')

    plt.suptitle("HOG Visualizations – One per WBC Class", fontsize=16)
    plt.tight_layout()
    plt.show()


# ----------------- Training ----------------- #
def compute_class_averages(train_dir):
    class_hogs = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            hogs = []
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    features = extract_hog_features(img)
                    hogs.append(features)
            if hogs:
                avg_hog = np.mean(hogs, axis=0)
                class_hogs[class_name] = avg_hog
                print(f"Processed {class_name}: {len(hogs)} images")

                # Display the first few entries of the averaged HOG vector
                print(
                    f"First few entries of the averaged HOG vector for {class_name}: {avg_hog[:10]}")  # Display first 10 entries

    return class_hogs


# ----------------- Prediction ----------------- #
def classify_image(image, class_hogs):
    test_hog = extract_hog_features(image)
    min_mse = float('inf')
    predicted_class = None
    for class_name, avg_hog in class_hogs.items():
        min_len = min(len(test_hog), len(avg_hog))
        mse = mean_squared_error(test_hog[:min_len], avg_hog[:min_len])
        if mse < min_mse:
            min_mse = mse
            predicted_class = class_name
    return predicted_class


# ----------------- Evaluation ----------------- #
def evaluate(test_dir, class_hogs):
    y_true = []
    y_pred = []

    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    pred = classify_image(img, class_hogs)
                    y_true.append(class_name)
                    y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    class_wise_accuracy(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    return y_true, y_pred


# ----------------- Class-wise Accuracy ----------------- #
def class_wise_accuracy(y_true, y_pred):
    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    for actual, predicted in zip(y_true, y_pred):
        class_total[actual] += 1
        if actual == predicted:
            class_correct[actual] += 1

    print("\nClass-wise Accuracy:")
    for cls in class_total:
        acc = 100 * class_correct[cls] / class_total[cls]
        print(f"{cls:12s} : {acc:.2f}%")


# ----------------- Confusion Matrix Plot ----------------- #
def plot_confusion_matrix(y_true, y_pred):
    # Define consistent class order
    classes = ['Monocyte', 'Lymphocyte', 'Neutrophil', 'Basophil', 'Eosinophil']

    # Create confusion matrix with fixed label order
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Plotting using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the matrix as a heatmap
    cax = ax.matshow(cm, cmap='Blues')

    # Add a color bar to the heatmap
    fig.colorbar(cax)

    # Set the labels for the x and y axes
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Rotate the tick labels and set their alignment
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adding labels and title
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')

    # Annotating the matrix cells with the counts
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.show()


# ----------------- Run Pipeline ----------------- #
if __name__ == "__main__":
    train_path = r"D:\Sem 6\DIP\wbc_data\Train"
    test_path = r"D:\Sem 6\DIP\wbc_data\Test"

    print("Extracting HOG features from training set...")
    class_averages = compute_class_averages(train_path)

    print("\nEvaluating on test set...")
    y_true, y_pred = evaluate(test_path, class_averages)

    # Sample Prediction
    print("\nShowing prediction on a sample test image...")
    test_img_path = r"D:\Sem 6\DIP\wbc_data\Test\Eosinophil\Eosinophil_50.jpg"
    if os.path.exists(test_img_path):
        test_img = cv2.imread(test_img_path)
        pred_class = classify_image(test_img, class_averages)
        print(f"Predicted Class for '{os.path.basename(test_img_path)}': {pred_class}")
        cv2.imshow(f"Predicted: {pred_class}", test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Test image not found: {test_img_path}")

    # Visualizations
    print("\nVisualizing one HOG image per WBC class...")
    plot_hog_per_class(train_path)

    print("\nVisualizing all training HOG features using PCA...")
    visualize_hog_features_with_pca(train_path)
