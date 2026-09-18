from django.urls import path

from .views import LandslideDetailView, LandslideListView

urlpatterns = [
    path("landslides/", LandslideListView.as_view(), name="landslide-list"),
    path("landslides/<str:id>/", LandslideDetailView.as_view(), name="landslide-detail"),
]
