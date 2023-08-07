import logging
import azure.functions as func
from io import BytesIO
from tensorflow.keras.models import load_model
from azure.storage.blob import BlobClient
from h5py import File as h5py_File
from json import dumps as json_dumps
from numpy import array as numpy_array, argmax as numpy_argmax
from os import environ as env_var


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    logging.info(req_body)

    # Get data from request
    age = float(req_body.get('age'))
    presenting_problem = float(req_body.get('presentingProblem'))
    positive_discriminator = float(req_body.get('positiveDiscriminator'))
    respiratory_rate = float(req_body.get('respiratoryRate'))
    heart_rate = float(req_body.get('heartRate'))
    oxigen_saturation = float(req_body.get('oxygenSaturation'))
    temperature = float(req_body.get('temperature'))

    # Load the model
    sas_url = env_var['MODEL_SAS_URL']
    blob_service_client = BlobClient.from_blob_url(sas_url)

    model_file = h5py_File(
        BytesIO(blob_service_client.download_blob().content_as_bytes(max_concurrency=1)), 'r')
    model = load_model(model_file)

    # Predict
    parameters = numpy_array([[age, respiratory_rate, heart_rate, temperature,
                               oxigen_saturation, positive_discriminator, presenting_problem]])
    result = model.predict(parameters)

    # Get result name
    if numpy_argmax(result) == 0:
        result_name = 'NonUrgent'
    elif numpy_argmax(result) == 1:
        result_name = 'Standard'
    elif numpy_argmax(result) == 2:
        result_name = 'Urgent'
    elif numpy_argmax(result) == 3:
        result_name = 'VeryUrgent'
    elif numpy_argmax(result) == 4:
        result_name = 'Emergent'

    # Return result
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }
    return func.HttpResponse(json_dumps({"result": result_name}), status_code=200, headers=headers)
