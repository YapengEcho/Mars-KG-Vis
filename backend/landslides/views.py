from django.http import Http404
from rest_framework.exceptions import NotFound, ValidationError
from rest_framework.generics import ListAPIView, RetrieveAPIView

from .models import Landslide
from .serializers import LandslideDetailSerializer, LandslideListSerializer


def _float_param(params, name):
    raw = params.get(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError({"detail": f"参数 {name} 必须是数字"}) from exc


class LandslideListView(ListAPIView):
    serializer_class = LandslideListSerializer

    def get_queryset(self):
        qs = Landslide.objects.all()
        params = self.request.query_params

        landslide_type = params.get("type")
        if landslide_type:
            qs = qs.filter(type=landslide_type)

        region = params.get("region")
        if region:
            qs = qs.filter(region=region)

        min_lat = _float_param(params, "min_lat")
        max_lat = _float_param(params, "max_lat")
        min_lon = _float_param(params, "min_lon")
        max_lon = _float_param(params, "max_lon")
        if min_lat is not None:
            qs = qs.filter(lat__gte=min_lat)
        if max_lat is not None:
            qs = qs.filter(lat__lte=max_lat)
        if min_lon is not None:
            qs = qs.filter(lon__gte=min_lon)
        if max_lon is not None:
            qs = qs.filter(lon__lte=max_lon)
        return qs


class LandslideDetailView(RetrieveAPIView):
    queryset = Landslide.objects.all()
    serializer_class = LandslideDetailSerializer
    lookup_field = "id"
    lookup_url_kwarg = "id"

    def get_object(self):
        try:
            return super().get_object()
        except Http404:
            raise NotFound("未找到该滑坡") from None
