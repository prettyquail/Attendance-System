from django.shortcuts import render
from .training_model import result
import datetime
import json


# Create your views here.
def home(request):
    date = request.GET.get("date")
    image = request.GET.get("images")
    date_param = 0
    if date:
        date_param = datetime.datetime.strptime(str(date), "%Y-%m-%d").strftime(
            "%d/%m/%Y"
        )
    # print("*********Date***********", date_param)
    # print("*********Image**********", image)
    context = {}
    if date_param and image:
        df = result(str(image), str(date_param))
        context = {"d": df, "date": date_param, "image": image}
    return render(request, "home.html", context)
