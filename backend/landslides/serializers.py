from rest_framework import serializers

from .models import Landslide


class LandslideListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Landslide
        fields = ("id", "name", "type", "lat", "lon", "length_m", "mission")


class LandslideDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Landslide
        fields = (
            "id",
            "name",
            "type",
            "lat",
            "lon",
            "length_m",
            "width_m",
            "slope_deg",
            "mission",
            "region",
            "features",
            "description",
            "source_url",
        )
