import torch
from torchvision import transforms, models
import json
import numpy as np
import torch.nn as nn
from PIL import Image
import sys
from openai import OpenAI



model_path = 'best_model_weights.pth'

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the same model we trained on
# and load in our trained layer
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_classes = 4  
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load our pretrained layer
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


def diagnose(image_path):

    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0) # put it into dimensions to act as a batch
    img_tensor = img_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.cpu().numpy()[0]

def get_llm_rec(json_data):
    # Use of Generative AI
    # (Bonus Points)

    ## BE SURE that you have your own API key if you would like generative reommendations
    # We removed our own due to privacy reasons. 
    key = "put your key here and comment the next line" 
    key = "hiding_key_for_submission"
    
    if(key == "hiding_key_for_submission"):
        return("API Key removed for submission for privacy reasons. The Safe bet is to go to a doctor!")
    
    client = OpenAI(
        api_key=key
    )
    # Format the prompt and call the model
    response = client.responses.create(
        model="gpt-3.5-turbo", 
        instructions = "Here are the classification results for an MRI image of a brain tumor. Based on this, recommend a course of action in simple language. Keep the Response within a few sentences.",
        input = json.dumps(json_data, indent=2)
    )

    return(response.output_text)
    

if __name__ == "__main__":

    # to run the system from the command line

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    probs = diagnose(image_path)
    probs = [float(p) for p in probs]

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # turn outputs into JSON format for parsability
    out = dict(zip(classes,probs))
    json_data = json.dumps(out)

    rec = get_llm_rec(json_data)


    out['rec'] = rec

    final_output = json.dumps(out)

    # print("Class probabilities:", probs)
    # print("Predicted Class: ", classes[np.argmax(probs)])

    # predicted = classes[np.argmax(probs)]



    print(final_output)