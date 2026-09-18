from django.contrib import admin

from .models import Landslide


@admin.register(Landslide)
class LandslideAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "type", "lat", "lon", "mission", "region")
    list_filter = ("type", "mission", "region")
    search_fields = ("id", "name", "description")
