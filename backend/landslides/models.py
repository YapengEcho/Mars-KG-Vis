from django.db import models


class Landslide(models.Model):
    """一行一条滑坡。id 与图谱 / RAG 引用共用同一套 ls_xxxx。"""

    class Type(models.TextChoices):
        SLUMP = "slump", "slump"
        FALL = "fall", "fall"
        FLOW = "flow", "flow"
        SLIDE = "slide", "slide"
        UNKNOWN = "unknown", "unknown"

    id = models.CharField("编号", max_length=32, primary_key=True)
    name = models.CharField("名称", max_length=200)
    type = models.CharField("类型", max_length=16, choices=Type.choices, db_index=True)
    lat = models.FloatField("纬度")
    lon = models.FloatField("经度")
    length_m = models.FloatField("长度(米)", null=True, blank=True)
    width_m = models.FloatField("宽度(米)", null=True, blank=True)
    slope_deg = models.FloatField("坡度(度)", null=True, blank=True)
    mission = models.CharField("探测任务", max_length=64, blank=True, default="", db_index=True)
    region = models.CharField("区域", max_length=64, blank=True, default="", db_index=True)
    features = models.JSONField("地貌特征", default=list, blank=True)
    description = models.TextField("描述", blank=True, default="")
    source_url = models.URLField("来源", max_length=500, blank=True, default="")

    class Meta:
        db_table = "landslide"
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.id} {self.name}"
