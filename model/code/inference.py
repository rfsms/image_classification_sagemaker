
import logging
from torch import nn
import torch
import torchvision.models as models
import os
from torchvision import transforms
from PIL import Image
import json
import subprocess
import sys
import s3fs

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("s3fs")



fs = s3fs.S3FileSystem()


logger = logging.getLogger(__name__)

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    logger.info('Done loading model')
    return model





def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data["url"]
        logger.info(url)
        with fs.open(url) as f:
            image_data = Image.open(f)
            image_transform = transforms.Compose([
                transforms.CenterCrop(size=224),
                transforms.ToTensor()
            ])
            img = image_transform(image_data)

        return img
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')





def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)
    return ps


def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    classes = {0: 'error', 1: 'no_error'}
    _, pred = torch.max(prediction_output, 1)
    result = classes[int(pred)]
    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')








