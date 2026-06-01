from django.urls import path

from . import api_views, views

app_name = "base"

urlpatterns = [
    path("", views.Home, name="home"),
    path("api/predict/", api_views.predict_api, name="api_predict"),
    path("api/upload/", api_views.upload_api, name="api_upload"),
]
