# 🌍 Natural disaster alert: Flood & Fire Prediction

The **Natural Disaster Alert System** is an end-to-end **data pipeline and machine learning system** designed to ingest **historical and real-time data**, process it at scale, and generate **real-time flood and fire predictions**.

The system follows a **modern data architecture (Bronze / Silver / Gold)** and provides **real-time dashboards** to support timely decision-making for disaster monitoring and risk prevention.


---

## 🏗️ 1. High-Level Architecture
The system follows the **Medallion (Lakehouse) Architecture**, ensuring data evolves from raw logs to high-value features for Machine Learning.

![Architecture](image_113b75.png)

| Stage | Technology | Purpose |
| :--- | :--- | :--- |
| **Ingestion** | Kafka + Spark | Hybrid Batch & Streaming ingestion. |
| **Processing** | Apache Spark | Distributed cleaning and feature engineering. |
| **Storage** | Parquet + InfluxDB | Efficient data lake storage + Time-series performance. |
| **Analytics** | ML Models | Real-time hazard classification. |
| **Visualization**| Grafana | Live monitoring and alerting dashboards. |

---

## 📥 2. Data Ingestion & Lifecycle
We manage data from diverse sources including NASA satellite feeds, weather APIs, and IoT sensor simulations.

![Data Ingestion](image_113b78.png)

* **Ingestion Layer:** Real-time data is funneled through specialized **Kafka Topics** (`flood_topic`, `fire_topic`).
* **Initial Storage:** Data is first persisted in **Bronze (Raw)** Parquet format to ensure auditability.
* **Downstream Flow:** Validated data is transformed into **Silver (Cleaned)** and **Gold (Aggregated)** layers for consumption.

---

## 🛠️ 3. Preprocessing & Feature Engineering
Data quality is paramount. Our pipeline automates the transition from noisy sensor data to ML-ready features.

![Uploading image.png…]()


### **Three-Step Refinement:**
1.  **Data Loading:** High-speed retrieval from Parquet using Spark's distributed engine.
2.  **Quality Processing:** Standardizing column names, handling missing values (via Median/Mode), and schema validation.
3.  **Variable Creation:** Engineering high-impact features such as:
    * **Rain × Water Level** (Interaction effect)
    * **Flow / Rain** (Ratio analysis)
    * **Global Risk Score** (Composite indicator)

---

## 🤖 4. Model Performance & Evaluation
The pipeline feeds two specialized Random Forest models. Both models demonstrate high reliability for early warning systems.

![Models](image_113b7b.png)

| Feature | 💧 Flood Prediction | 🔥 Fire Prediction |
| :--- | :--- | :--- |
| **Variables** | 9 (Rainfall, Water Level, etc.) | 167 (Weather, Vegetation, Spatial) |
| **Complexity** | 200 Trees | 100 Trees |
| **Accuracy** | **85.0%** | **89.6%** |
| **F1-Score** | **0.842** | **0.8855** |

---

## 💾 5. Storage & Monitoring Strategy
We utilize a dual-layer storage approach to support both deep historical analysis and sub-second real-time alerting.

![Data Storage](image_113bb4.png)

### **Historical Lake (Parquet)**
* `*_ingested/`: Source of truth (Bronze).
* `*_cleaned/`: Validated/standardized (Silver).
* `*_features/`: Transformed for training (Gold).

### **Real-time Sink (InfluxDB)**
* **Granularity:** Time-series storage for live predictions.
* **Schema:** Tracks `location_id`, `coordinates`, `probability`, and `alert_level`.
* **Alerting:** Logic triggers when the `extreme_event` flag returns `True`.

---

## 🚀 6. Installation & Execution

### **Quick Setup**
```bash
# 1. Start Dockerized services (Kafka, InfluxDB, Grafana)
docker-compose up -d

# 2. Install dependencies
pip install pyspark confluent-kafka influxdb-client

# 3. Trigger Ingestion
python src/ingestion/kafka_producer.py

Spark Submission
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.x.x src/processing/main_pipeline.py
