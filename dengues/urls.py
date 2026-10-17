from django.urls import path
from .views import DenguePredictionView, DenguePredictionHistoryView

urlpatterns = [
    path('predict/', DenguePredictionView.as_view(), name='dengue-predict'),
    path('history/', DenguePredictionHistoryView.as_view(), name='dengue-history'),
]