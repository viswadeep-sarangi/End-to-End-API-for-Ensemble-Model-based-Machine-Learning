from fastapi.testclient import TestClient
import api

client = TestClient(api.app)

sample_json_for_prediction = {
    "features": {
        "age": 75,
        "anaemia": 0,
        "creatinine_phosphokinase": 582,
        "diabetes": 0,
        "ejection_fraction": 20,
        "high_blood_pressure": 1,
        "platelets": 265000,
        "serum_creatinine": 1.9,
        "serum_sodium": 130,
        "sex": 1,
        "smoking": 0,
        "time": 4
    }
}


def test_good_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"predictions": None, "error": "This is a test endpoint."}


def test_bad_predict_get_request():
    response = client.get("/predict")
    assert response.status_code == 200
    assert response.json() == {"predictions": None,
                               "error": "Send a POST request to this endpoint with 'features' data."}


def test_bad_train_get_request():
    response = client.get("/train")
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Send a POST request to this endpoint with 'model_name' (svm, decisiontree or neuralnetwork) and 'csv_data_file' data with a 'save_model_to_disk' option"}


def test_bad_downloadmodel_get_request():
    response = client.get("/downloadmodel")
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Send a POST request to this endpoint with 'unique_model_name' of the previously trained model"}


def test_good_predict_post_request():
    response = client.post(
        "/predict",
        json=sample_json_for_prediction
    )
    assert response.status_code == 200
    assert response.json()["error"] is None


def test_bad_predict_post_request():
    response = client.post(
        "/predict",
        json={"Incorrect Key": "Incorrect Value"}
    )
    assert response.status_code == 422


def test_bad_predict_post_request_2():
    response = client.post(
        "/predict",
        json={"features": {
            "incorrect feature key": 1.0
        }
        }
    )
    assert response.status_code == 200
    assert response.json()["error"] is not None
