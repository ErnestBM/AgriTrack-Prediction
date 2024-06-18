# predictions_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('predict/<str:commodity>/', views.predict_commodity, name='predict_commodity'),
]
