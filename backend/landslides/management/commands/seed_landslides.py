from django.core.management.base import BaseCommand

from landslides.models import Landslide
from landslides.seed_data import SEED_LANDSLIDES


class Command(BaseCommand):
    help = "写入演示用火星滑坡样本（可重复执行，按主键更新）。"

    def handle(self, *args, **options):
        created = 0
        updated = 0
        for row in SEED_LANDSLIDES:
            payload = {key: value for key, value in row.items() if key != "id"}
            _, is_created = Landslide.objects.update_or_create(
                id=row["id"],
                defaults=payload,
            )
            if is_created:
                created += 1
            else:
                updated += 1
        self.stdout.write(
            self.style.SUCCESS(f"种子数据完成：新增 {created}，更新 {updated}，合计 {len(SEED_LANDSLIDES)}")
        )
