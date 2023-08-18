from sensor.entity.config_entity import ModelPusherConfig
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from sensor.exception import SensorException
import sys
import os
import shutil


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 data_transform_artifact: DataTransformationArtifact,
                 model_eval_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.data_transform_artifact = data_transform_artifact
            self.model_eval_artifact = model_eval_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            transformer_path = self.data_transform_artifact.transformed_object_file_path
            trained_model_path = self.model_eval_artifact.trained_model_path

            # Creating transformer pusher dir to save transformer
            transformer_file_path = self.model_pusher_config.transformer_file_path
            os.makedirs(os.path.dirname(transformer_file_path), exist_ok=True)
            shutil.copy(src=transformer_path, dst=transformer_file_path)

            # saved transformer dir
            saved_transformer_path = self.model_pusher_config.saved_transformer_path
            os.makedirs(os.path.dirname(saved_transformer_path), exist_ok=True)
            shutil.copy(src=transformer_path, dst=saved_transformer_path)

            # Creating model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

            # saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            # prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_transformer_path=saved_transformer_path,
                                                        transformer_file_path=transformer_file_path,
                                                        saved_model_path=saved_model_path,
                                                        model_file_path=model_file_path)
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e, sys)
