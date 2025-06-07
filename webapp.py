from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import datetime
import psycopg2
from torchvision import transforms
from mnist_cnn import Net  # your trained model

# Load model
device = torch.device("cpu")
model = Net()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Define transform for preprocessing canvas image
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
])

# Connect to PostgreSQL
import os

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )


# Log prediction + user feedback
def log_prediction(pred, true_label, confidence):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (timestamp, pred, true_label, confidence)
            VALUES (%s, %s, %s, %s)
        """, (datetime.datetime.now(), pred, true_label, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"❌ Failed to log prediction: {e}")

# Load recent prediction history
def load_history():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT timestamp, pred, true_label, confidence FROM predictions ORDER BY timestamp DESC LIMIT 10")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"❌ Failed to load history: {e}")
        return []

# --- Streamlit UI ---

st.title("🧠 PyTorch MNIST Digit Recognizer")
st.markdown("Draw a digit (0–9) below, click **Predict**, then submit feedback.")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # 1) Convert canvas result to PIL image, grayscale + resize
        pil_img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
        pil_img = pil_img.convert("L")
        pil_img = preprocess(pil_img)  # now a 28×28 PIL image

        # 2) Convert to numpy array
        arr = np.array(pil_img, dtype=np.float32)  # shape (28,28)

        # 3) Normalize & convert to Tensor [1,1,28,28] without from_numpy
        # Convert arr (NumPy array) to a plain list to avoid the NumPy-specific path:
        tensor_img = torch.tensor(arr.tolist(), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor_img = (tensor_img - 0.1307) / 0.3081

        # 4) Run prediction
        with torch.no_grad():
            output = model(tensor_img)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item() * 100

        # 5) Store in session state and display
        st.session_state.prediction = pred
        st.session_state.confidence = confidence

        st.subheader("Prediction")
        st.metric("Digit", pred, f"{confidence:.1f}%", delta_color="normal")




# Feedback + Logging
if "prediction" in st.session_state and "confidence" in st.session_state:
    true_label = st.number_input("Enter True Label:", min_value=0, max_value=9, step=1, value=st.session_state.prediction)

    if st.button("Submit Feedback"):
        try:
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            log_prediction(prediction, true_label, confidence)
            st.success("✅ Feedback logged to database!")
            st.write(f"Logged: prediction={prediction}, true_label={true_label}, confidence={confidence:.2f}%")
        except Exception as e:
            st.error(f"❌ Error while logging: {e}")

# Show prediction history
st.markdown("## Prediction History")
history = load_history()
if history:
    st.table({
        "Timestamp": [str(row[0]) for row in history],
        "Pred": [row[1] for row in history],
        "True": [row[2] for row in history],
        "Conf": [f"{row[3]:.1f}%" for row in history],
    })
else:
    st.info("No predictions logged yet.")
