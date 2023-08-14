import os
from sensor.constants.s3_bucket import TRAINING_BUCKET_NAME

# Defining common constatnt variables for a training pipeline
TARGET_COLUMN = 'class'
PIPELINE_NAME: str = 'sensor'
ARTIFACT_DIR: str = 'artifact'
FILE_NAME: str = 'sensor.csv'

TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'

PREPROCSSING_OBJECT_FILE_NAME = 'preprocessing.pkl'
MODEL_FILE_NAME = 'model.pkl'
SCHEMA_FILE_PATH = os.path.join('config', 'schema.yaml')
SCHEMA_DROP_COLS = 'drop_columns'
