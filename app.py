from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/upload/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def pca_denoise(img, k, patch_size=8):
    """
    Patch-based PCA denoising (works for heavy lines/noise).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    h, w = gray.shape
    restored = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)

    # Slide over the image in patches
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):

            patch = gray[y:y+patch_size, x:x+patch_size]
            U, S, VT = np.linalg.svd(patch, full_matrices=False)

            S[k:] = 0  # keep only top-k eigen components
            patch_restored = np.dot(U * S, VT)

            restored[y:y+patch_size, x:x+patch_size] += patch_restored
            count[y:y+patch_size, x:x+patch_size] += 1

    restored = restored / count
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    return cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files['image']
    k = int(request.form['components'])

    img_path = UPLOAD_FOLDER + "uploaded.jpg"
    output_path = UPLOAD_FOLDER + "restored.jpg"

    file.save(img_path)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 400))

    restored = pca_denoise(img, k)

    cv2.imwrite(output_path, restored)

    return render_template("index.html",
                           original=img_path,
                           restored=output_path)


if __name__ == "__main__":
    app.run(debug=True)
