from sensor.entity.config_entity import (TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig,
                                         DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig,
                                         ModelPusherConfig)
from sensor.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact,
                                           ModelTrainerArtifact, ModelEvaluationArtifact)
from sensor.logger import logging
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.exception import SensorException
import sys
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.cloud_storage.s3_syncer import S3Sync
from sensor.constants.s3_bucket import TRAINING_BUCKET_NAME
from sensor.constants.training_pipeline import SAVED_MODEL_DIR


class TrainPipeline:
    is_pipeline_running = False

    def __init__(self):
        self.data_ingestion_config = None
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info('Starting data ingestion')

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f'Data ingestion completed and artifact: {data_ingestion_artifact}')

            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_data_validaton(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config
                                             )

            logging.info('Starting data validation')
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info('Data validation stage completed')

            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config
                                                     )

            logging.info('Starting data transformation')
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info('Data transformation stage completed')

            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)

            logging.info('Starting model trainer')
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info('Model trainer stage completed')

            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_model_evaluation(self, data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact):
        try:
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config, data_validation_artifact, model_trainer_artifact)

            logging.info('Starting model evaluation')
            model_eval_artifact = model_eval.initiate_model_evaluation()
            logging.info('Model evaluation stage completed')

            return model_eval_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_model_pusher(self, data_transform_artifact: DataTransformationArtifact, model_eval_artifact: ModelEvaluationArtifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, data_transform_artifact, model_eval_artifact)

            logging.info('Starting model pusher')
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info('Model pusher stage completed')

            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f's3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}'
            logging.info(f'aws bucket url: {aws_bucket_url}')
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir,
                                           aws_bucket_url=aws_bucket_url)
            logging.info('Successfully able to synch artifact directory to s3')
        except Exception as e:
            raise SensorException(e, sys)

    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f's3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}'
            logging.info(f'aws bucket url: {aws_bucket_url}')
            self.s3_sync.sync_folder_to_s3(folder=SAVED_MODEL_DIR, aws_bucket_url=aws_bucket_url)
            logging.info('Successfully able to synch saved model directory to s3')
        except Exception as e:
            raise SensorException(e, sys)

    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)

            if not model_eval_artifact.is_model_accepted:
                raise Exception('Trained model is not better than the best model')

            model_pusher_artifact = self.start_model_pusher(data_transformation_artifact, model_eval_artifact)
            TrainPipeline.is_pipeline_running = False

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
        except Exception as e:
            self.sync_artifact_dir_to_s3()

            TrainPipeline.is_pipeline_running = False
            raise SensorException(e, sys)
