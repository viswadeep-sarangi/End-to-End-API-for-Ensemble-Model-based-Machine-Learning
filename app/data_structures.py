from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi import UploadFile, File

'''
This file houses all the classes used for communication for the API calls (extending the Base Model)
The class names suffixed with 'Request' define the formats used for accepting the API calls
the class names suffixed with 'Response' define the formats used for returning a result of an API call 
'''


class PredictRequest(BaseModel):
    features: Dict[str, float]


class ModelResponse(BaseModel):
    predictions: Optional[List[Dict[str, float]]]
    error: Optional[str]


class TrainRequest(BaseModel):
    data: UploadFile = File(...)
    model_name: str
    save_model: Optional[bool]


class TrainResponse(BaseModel):
    accuracy: Optional[str]
    unique_model_name: Optional[str]
    details: Optional[str]
    error: Optional[str]
