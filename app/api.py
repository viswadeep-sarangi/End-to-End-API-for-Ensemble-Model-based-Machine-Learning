from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from typing import Optional
import uvicorn
import os
import training
from data_structures import ModelResponse, PredictRequest, ModelName, TrainResponse, TrainRequest
from train import save_model
import prediction as pred

app = FastAPI()


@app.get("/", response_model=ModelResponse)
async def root() -> ModelResponse:
    return ModelResponse(error="This is a test endpoint.")


@app.get("/predict", response_model=ModelResponse)
async def explain_api() -> ModelResponse:
    return ModelResponse(
        error="Send a POST request to this endpoint with 'features' data."
    )


@app.post("/predict")
async def get_model_predictions(request: PredictRequest) -> ModelResponse:
    print("Got Request: {0}".format(request))

    if pred.ensemble_models is None:
        return ModelResponse(
            error="The models couldn't be loaded properly. Please consider retraining and saving the models")

    prediction_valid, model_prediction = pred.predict(request)
    if prediction_valid:
        return ModelResponse(predictions=[model_prediction], error="Success")
    else:
        return ModelResponse(error="Error in getting prediction from the models")


@app.post("/train")
async def train_model(model_name: ModelName, data: UploadFile = File(...),
                      save_model: Optional[bool] = False) -> TrainResponse:
    if not os.path.exists(training.base_address):
        os.mkdir(training.base_address)

    contents = await data.read()

    f = open(os.path.join(training.base_address, data.filename), "wb")
    f.write(contents)
    f.close()

    [model_filename, accuracy, model_desc] = training.train_specific_model(f.name, model_name)

    if save_model:
        return TrainResponse(accuracy=str(float(accuracy) * 100) + "%", unique_model_name=model_filename,
                             details="{0}. Call the /downloadmodel API endpoint with the unique_model_name to save the model to disk".format(model_desc))
    else:
        return TrainResponse(accuracy=str(float(accuracy) * 100) + "%")


@app.post("/downloadmodel")
async def download_model(unique_model_name: str) -> FileResponse:
    model_exists, model_path = save_model.check_model_exists(unique_model_name)
    if model_exists:
        return FileResponse(path=model_path, filename=unique_model_name)
    else:
        return None


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
