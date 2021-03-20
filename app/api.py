
from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
import prediction as pred
import training
from data_structures import ModelResponse, PredictRequest, ModelName, TrainResponse, TrainRequest

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

    if not pred.models_loaded:
        return ModelResponse(
            error="The models couldn't be loaded properly. Please consider retraining and saving the models")

    model_prediction = pred.predict(request)
    return ModelResponse(error="Hit the 'predict' function")


@app.post("/train")
async def train_model(model_name: ModelName, data: UploadFile = File(...)) -> TrainResponse:
    if not os.path.exists(training.base_address):
        os.mkdir(training.base_address)

    contents = await data.read()

    f = open(os.path.join(training.base_address, data.filename), "wb")
    f.write(contents)
    f.close()

    training.train_specific_model(f.name, model_name)

    return TrainResponse(accuracy="")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
