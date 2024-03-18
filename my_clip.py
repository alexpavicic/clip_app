import os
import clip
import torch
from torchvision.datasets import CIFAR100

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import base64
from PIL import Image
import io

# Load the CLIP model
def load_clip_model(device):
    return clip.load('ViT-B/32', device)

# Process image and generate predictions
def process_image(image_bytes, model, text_inputs, cifar100):
    image = Image.open(io.BytesIO(image_bytes))
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Prepare predictions for display
    predictions = [
        {"class": cifar100.classes[index], "confidence": 100 * value.item()}
        for value, index in zip(values, indices)
    ]

    return image, predictions

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Load the model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_clip_model(device)

# Create text inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Dash app initialization
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Image')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-image-upload'),
])

# Callback to process uploaded image and display predictions
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def update_output(content, filename):
    if content is not None:
        # Decode and process image
        content_type, content_string = content.split(',')
        image_bytes = base64.b64decode(content_string)
        image, predictions = process_image(image_bytes, model, text_inputs, cifar100)

        # Prepare predictions for display
        predictions_html = html.Div([
            html.H5('Top predictions:'),
            html.Ul([
                html.Li(f"{pred['class']}: {pred['confidence']:.2f}%")
                for pred in predictions
            ])
        ])

        # Display image and predictions
        return html.Div([
            html.H5(filename),
            html.Img(src=content, style={'width': '50%'}),
            predictions_html
        ])
    else:
        raise PreventUpdate

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
