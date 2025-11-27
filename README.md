# Galaxy Morphological Classification Pipeline

## **Quick Start**

* **Watch Demo Video:** [YouTube Demo](https://youtu.be/your-video-link)
* **Streamlit UI Dashboard:** [Open Dashboard](https://galaxymorphologicalclassificationsummativeassignment.streamlit.app/)
* **API Endpoint (Render Deployment):** [API Link](https://galaxymorphologicalclassification.onrender.com/)

## **Project Description**

This project is a Machine Learning pipeline for classifying galaxy images based on their morphology. It includes:

* **Image Data Acquisition**
* **Data Preprocessing**
* **Model Creation and Training** (using MobileNetV2)
* **Prediction API** (via FastAPI)
* **Model Retraining** (with new data upload and trigger endpoint)
* **UI Dashboard** (via Streamlit)
* **Load Testing Simulation** (using Locust)
* **Deployment** (Render for API, Streamlit Cloud for UI)

The pipeline allows prediction of individual images, retraining with new images, and simulation of high-traffic requests to assess performance.

---

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/e_dania/GalaxyMorphologicalClassification_SummativeAssignment.git
cd GalaxyMorphologicalClassification_SummativeAssignment
```

### **2. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Prepare Data**

* Place raw galaxy images in `data/raw/images_train/` and `data/raw/images_test/`.
* For retraining, create `data/new/` with subfolders `class_0/`, `class_1/`, `class_2/` containing new images.

### **5. Run the API**

```bash
uvicorn api:app --reload
```

* Prediction endpoint: `/predict`
* Retrain endpoint: `/retrain`
* Upload new data endpoint: `/upload_new_data`

### **6. Run Streamlit Dashboard**

```bash
streamlit run streamlit_app.py
```

* Provides: model uptime, prediction interface, retraining trigger, and data visualizations.

### **7. Load Testing (Optional)**

* Install Locust:

```bash
pip install locust
```

* Create a `locustfile.py` simulating multiple users sending requests to the API.
* Run:

```bash
locust -f locustfile.py
```

* Open [http://localhost:8089](http://localhost:8089) to start load testing.

---
