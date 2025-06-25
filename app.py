import streamlit as st
import nibabel as nib
import numpy as np
import cv2
import tempfile
import os
import shutil
from inference_sdk import InferenceHTTPClient
import supervision as sv
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment variables from a .env file if present
# --------------------------------------------------
load_dotenv()

API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY is not set. Create a .env file with your key or set it as an environment variable."
    )

# Initialize Roboflow inference client
CLIENT = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("🧠 Brain Tumor Segmentation")

uploaded_files = st.file_uploader(
    "Upload one or more .nii, .nii.gz, or image files (.jpg/.png)",
    type=["nii", "nii.gz", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

def run_inference(image_np):
    """Run segmentation inference on a single image (NumPy array, BGR)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, image_np)

    result = CLIENT.infer(temp_path, model_id="brain-tumor-segmentation-model-1/1")
    os.remove(temp_path)

    if not result["predictions"]:
        # No detections, return original image with message
        no_detection_img = image_np.copy()
        cv2.putText(
            no_detection_img,
            "No tumor detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        return no_detection_img

    xyxy, confidence, class_id, masks = [], [], [], []
    height, width = image_np.shape[:2]

    for pred in result["predictions"]:
        x1 = pred["x"] - pred["width"] / 2
        y1 = pred["y"] - pred["height"] / 2
        x2 = pred["x"] + pred["width"] / 2
        y2 = pred["y"] + pred["height"] / 2
        xyxy.append([x1, y1, x2, y2])
        confidence.append(pred["confidence"])
        class_id.append(pred["class_id"])

        polygon = np.array([[p["x"], p["y"]] for p in pred["points"]], dtype=np.int32)
        blank_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(blank_mask, [polygon], 1)
        masks.append(blank_mask.astype(bool))

    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidence),
        class_id=np.array(class_id),
        mask=np.array(masks),
    )

    labels = [pred["class"] for pred in result["predictions"]]
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    annotated = mask_annotator.annotate(scene=image_np.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    return annotated


if uploaded_files:
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
        st.divider()
        st.markdown(f"## 🗂️ File: `{uploaded_file.name}`")

        if file_ext in [".nii", ".nii.gz"]:
            st.info("Reading NIfTI file...")

            suffix = ".nii.gz" if file_ext == ".nii.gz" else ".nii"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name
                shutil.copyfileobj(uploaded_file, tmp)

            try:
                nii_img = nib.load(temp_path)
                data = nii_img.get_fdata()
                total_slices = data.shape[2]
                max_slices_to_process = 20  # You can change this if needed

                selected_indices = np.linspace(0, total_slices - 1, max_slices_to_process, dtype=int)

                st.success(
                    f"Found {total_slices} slices. Processing {len(selected_indices)} evenly spaced slices..."
                )

                for i in selected_indices:
                    slice_2d = data[:, :, i]
                    norm_slice = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX)
                    slice_img = cv2.cvtColor(norm_slice.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    annotated = run_inference(slice_img)

                    st.markdown(f"### Frame {i + 1}")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("Original")
                        st.image(slice_img, channels="BGR", use_container_width=True)
                    with col2:
                        st.markdown("Segmented")
                        st.image(annotated, channels="BGR", use_container_width=True)

            except Exception as e:
                st.error(f"Failed to load NIfTI file: {str(e)}")
            finally:
                os.remove(temp_path)

        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Failed to read image file.")
            else:
                annotated = run_inference(image)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("Original")
                    st.image(image, channels="BGR", use_container_width=True)
                with col2:
                    st.markdown("Segmented")
                    st.image(annotated, channels="BGR", use_container_width=True)
