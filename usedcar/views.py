from django.shortcuts import render, redirect
from django.urls import reverse
from usedcar import input_features
from usedcar import ml_model

# Create your views here.


def home(request):
    if request.method == 'POST':
        car_brand = request.POST.get('car_brand')
        return redirect(reverse('feature') + f'?car_brand={car_brand}')
        
    return render(request, 'home.html')

def feature(request):
    if request.method == 'POST':
        car_brand = request.POST.get('car_brand')
        model = request.POST.get('model')
        year = request.POST.get('year')
        transmission = request.POST.get('transmission')
        mileage = request.POST.get('mileage')
        fueltype = request.POST.get('fueltype')
        tax = request.POST.get('tax')
        miles_per_gallon = request.POST.get('miles_per_gallon')
        engine_size = request.POST.get('engine_size')
        
        price_prediction = ml_model.regression_model(car_brand.lower(),model, year, transmission, mileage, fueltype, tax, miles_per_gallon, engine_size)
        inr = (int(price_prediction)*90)
        return render(request, 'predict.html',{"price":int(price_prediction),"inr":inr})
    
    car_brand = request.GET.get('car_brand')
    input_feature = input_features.features(car_brand.lower())
    return render(request, 'features.html', input_feature)

