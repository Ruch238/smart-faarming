
from django.shortcuts import render, redirect, HttpResponse
from django.http import HttpResponse, HttpResponseRedirect
import torch
from torchvision import transforms
from .models import ResNet9
from PIL import Image
import joblib
import io
from pathlib import Path
import os

from subapp.diseases import disease_dic

from .models import Userr
from mainapp.forms import NewUserForm
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, authenticate, logout
from django.conf import settings
from django.contrib.auth.decorators import login_required


def home(request):
    return render(request, "home.html")


def about(request):
    return render(request, "about.html")


def tutorial(request):
    return render(request, "tutorial.html")


def ourteams(request):
    return render(request, "ourteams.html")


@login_required
def ourservices(request):
    return render(request, "ourservices.html")


@login_required
def diseases(request):
    from .models import Userr
    userrs = Userr.objects.all()
    p = userrs[len(userrs)-1].pic
    print(p.url)
    return render(request, 'diseases.html', {'pic': p.url})


def uploadImage(request):
    print("Request handling......")
    p = request.FILES['image']
    from .models import Userr
    userr = Userr(pic=p)
    userr.save()
    return render(request, "diseases.html")


def d_result(request):

    title = 'Disease Detection'

    if request.method == 'POST':
        file = request.FILES.get('image')
        # if not file:
        #     return render('disease.html', title=title)
        print("IMAGE============", file)
        pic = file.read()

        prediction = predict_image(pic)

        prediction_data = str(disease_dic[prediction])
        print("PREDICTION++++++++++++++++++++++", prediction)
        # return render('disease_result.html', prediction=prediction_data, title=title)

    return render(request, 'diseases_result.html', {"prediction": prediction_data})


def result(request):

    CL = joblib.load("crop_prediction.sav")
    lst = []

    lst.append(float(request.GET["N"]))
    lst.append(float(request.GET["P"]))
    lst.append(float(request.GET["K"]))
    lst.append(float(request.GET["temperature"]))
    lst.append(float(request.GET["ph"]))
    lst.append(float(request.GET["rainfall"]))

    print(lst)
    ans = CL.predict([lst])
    A = ans.tolist()
    ANS = ' '.join(map(str, A))
    print(ANS)

    return render(request, "result.html", {"ans": ANS, "lst": lst})


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = settings.PY_MODEL+'plant-disease-model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


def register_request(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect("home")
        messages.error(
            request, "Unsuccessful registration. Invalid information.")
    form = NewUserForm()
    return render(request=request, template_name="register.html", context={"register_form": form})


def login_request(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                return redirect("home")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request=request, template_name="login.html", context={"login_form": form})


def logout_request(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect("home")
