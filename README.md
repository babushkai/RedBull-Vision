# RedBull Vision

A comprehensive end-to-end machine learning system for RedBull can classification, leveraging Google Cloud Platform and Vertex AI.

## Background

This project demonstrates an end-to-end machine learning system that combines my experience as a Computer Vision Engineer with current Cloud Engineering expertise. The system is built on Google Cloud Platform and showcases various ML pipeline patterns.

## Methodology

- **Data Collection**: Images collected from the internet and personal photography
- **Data Augmentation**: iPhone's burst mode used to collect 100+ images for each RedBull type
- **Classification**: Binary classification between regular and sugar-free RedBull variants

## Technical Stack

- **Algorithm**: InceptionResNetV2 (pre-trained on ImageNet)
- **Orchestration**: Kubeflow Pipelines
- **Platform**: Google Cloud Platform (Vertex AI)
- **Pipeline Types**: 
  - Custom Training
  - AutoML (see [AutoML notebook](https://github.com/kwdaisuke/MLOps/blob/main/autoML_pipeline.ipynb))

## Dataset Management

### DVC Setup

This project uses DVC (Data Version Control) for dataset management. Reference: [DVC Documentation](https://dvc.org/doc/start/data-and-model-access)

```shell
# From source repository
pip install dvc
dvc init
dvc status
git commit -m "Initialize DVC"

dvc add redbull
git add redbull.dvc .gitignore
git commit -m "Add raw data"

dvc remote add -d drive gdrive://1J8A8XaIzrUp87_OAalmlqYD1S8uXpU5t
pip install 'dvc[gdrive]'
dvc push

git add .dvc/config
git commit -m "Configure remote storage"

# From destination repository
git clone <repository-url>
dvc remote list
dvc get https://github.com/kwdaisuke/test redbull

gsutil cp redbull gs://source_bucket/redbull
```

### Data Processing

```python
from google.cloud import storage
import pandas as pd

client = storage.Client()
bucket = "your-source-bucket-name"  # Set source bucket name here

# Create dataset schema
df = pd.DataFrame({"path": [file.name for file in client.list_blobs(bucket)]})
df["type"] = df.path.apply(lambda x: x.split("/")[-2])  # Extract folder name (e.g., sugar_free, normal)
df.to_csv("schema.csv", index=False)

# Upload schema to GCS
fullpath = f"gs://{bucket}/schema.csv"
!gsutil cp schema.csv {fullpath}
```

### Dataset Visualization

![Dataset Overview](image/dataset.png)

## Model Performance

- **AutoML**: Used for this demonstration (custom training also available)
- **Test Accuracy**: 100% on test dataset

![Performance Metrics](image/performance.png)

## Inference

The model successfully predicts RedBull type with high confidence:
- **Prediction**: Sugar-free RedBull
- **Confidence**: 89.9%

![Prediction Example](image/prediction.png)

## Design Patterns

This project showcases various ML pipeline patterns:

1. **End-to-End Notebook**: [RedBull Classification Pipeline](https://github.com/kwdaisuke/MLOps/blob/main/transfer_learning_custom_pipeline.ipynb)
2. **AutoML Pipeline**: Automated model training and deployment
3. **Function-Based Components**: [Lightweight Functions Component I/O](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/notebooks/official/pipelines/lightweight_functions_component_io_kfp.ipynb)
4. **Control Structure**: [Control Flow KFP](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/notebooks/official/pipelines/control_flow_kfp.ipynb)
5. **Scheduled Execution**: [Pipeline Scheduling with Cloud Scheduler](https://cloud.google.com/vertex-ai/docs/pipelines/schedule-cloud-scheduler)
6. **Data Management Pipeline**: Data preprocessing and validation workflows
7. **TensorFlow Advanced Pipeline**: Custom TensorFlow training pipelines
8. **Batch Prediction**: Large-scale batch inference processing

## References

- [Google Cloud Pipeline Components - Model Train, Upload, Deploy](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/notebooks/official/pipelines/google_cloud_pipeline_components_model_train_upload_deploy.ipynb)
- [Vertex AI Quickstart Lab](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/self-paced-labs/vertex-ai/vertex-ai-qwikstart/lab_exercise.ipynb)
- [Custom Image Classification Batch Processing](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/master/notebooks/official/custom/sdk-custom-image-classification-batch.ipynb)
- [Vertex AI Pipelines Documentation](https://cloud.google.com/vertex-ai/docs/pipelines/notebooks)

## Development Notes

- Importing dataset from Vertex Dataset is not necessary when using GCS directly
- Focus on pipeline components without additional Docker setup or YAML configurations
