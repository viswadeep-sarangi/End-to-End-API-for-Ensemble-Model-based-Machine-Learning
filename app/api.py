from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import uvicorn
import os
from app import training
from app.data_structures import ModelResponse, PredictRequest, ModelName, TrainResponse, TrainRequest
from train import save_model
import app.prediction as pred

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
        return ModelResponse(predictions=[model_prediction])
    else:
        return ModelResponse(error="Error in getting prediction from the models. Please check the JSON data again")


@app.get("/train")
async def explain_api_train():
    raise HTTPException(status_code=400,
                        detail="Send a POST request to this endpoint with 'model_name' (svm, decisiontree or neuralnetwork) and 'csv_data_file' data with a 'save_model_to_disk' option")


@app.post("/train", response_model=TrainResponse)
async def train_model(model_name: ModelName, csv_data_file: UploadFile = File(...),
                      save_model_to_disk: Optional[bool] = False) -> TrainResponse:
    if not os.path.exists(training.base_address):
        os.mkdir(training.base_address)

    contents = await csv_data_file.read()

    f = open(os.path.join(training.base_address, csv_data_file.filename), "wb")
    f.write(contents)
    f.close()

    csv_valid, [model_filename, accuracy, model_desc] = training.train_specific_model(f.name, model_name)

    if not csv_valid:
        return TrainResponse(error="Error in parsing the .csv file. Please make sure its valid")

    if save_model_to_disk:
        return TrainResponse(accuracy=str(float(accuracy) * 100) + "%", unique_model_name=model_filename,
                             details="{0}. Call the /downloadmodel API endpoint with the unique_model_name to save the model to disk".format(model_desc))
    else:
        return TrainResponse(accuracy=str(float(accuracy) * 100) + "%")


@app.get("/downloadmodel")
async def explain_api_downloadmodel():
    raise HTTPException(status_code=400,
                        detail="Send a POST request to this endpoint with 'unique_model_name' of the previously trained model")


@app.post("/downloadmodel")
async def download_model(unique_model_name: str) -> FileResponse:
    model_exists, model_path = save_model.check_model_exists(unique_model_name)
    if model_exists:
        return FileResponse(path=model_path, filename=unique_model_name)
    else:
        return None


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
