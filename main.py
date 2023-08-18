import os
import pandas as pd
from sensor.utils.main_utils import read_yaml_file, load_object
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from sensor.pipeline.training_pipeline import TrainPipeline
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.constants.training_pipeline import SAVED_TRANSFORMER_DIR, SAVED_MODEL_DIR
from sensor.logger import logging
from uvicorn import run as app_run
from sensor.constants.application import APP_HOST, APP_PORT
import numpy as np

env_file_path = os.path.join(os.getcwd(), 'env.yaml')


def set_env_variable(env_file_path):
    if os.getenv('MONGO_DB_URL', None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL'] = env_config['MONGO_DB_URL']


app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')


@app.get('/train')
async def train_route():
    try:
        train_pipeline = TrainPipeline()

        if train_pipeline.is_pipeline_running:
            return Response('Training pipeline is already running.')
        train_pipeline.run_pipeline()

        return Response('Training successful !!')
    except Exception as e:
        return Response(f'Error Occurred! {e}')


@app.post('/predict')
async def predict_route(user_file: UploadFile = File(...)):
    try:
        # get data from user csv file
        # conver csv file to dataframe
        model_resolver = ModelResolver(transformer_dir=SAVED_TRANSFORMER_DIR, model_dir=SAVED_MODEL_DIR)
        df = pd.read_csv(user_file.file)
        df.replace({'na': np.NAN}, inplace=True)

        if not model_resolver.is_transformer_exists():
            return Response('Transformer is not available')
        elif not model_resolver.is_model_exists():
            return Response('Model is not available')

        best_transformer_path = model_resolver.get_best_transformer_path()
        transformer = load_object(file_path=best_transformer_path)

        input_feature_names = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)

        y_pred = model.predict(input_arr)

        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(), inplace=True)

        # Convert the modified DataFrame back to CSV
        predicted_csv = df.to_csv(index=False)

        # Return the CSV content as a downloadable file
        response = Response(content=predicted_csv)
        response.headers['Content-Disposition'] = 'attachment; filename=predicted_results.csv'

        return response
    except Exception as e:
        return Response(f'Error Occured! {e}')


def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)


if __name__ == '__main__':
    # main()
    # set_env_variable(env_file_path)
    app_run(app, host=APP_HOST, port=APP_PORT)
