from django.test import TestCase
from rest_framework.test import APIClient

from landslides.models import Landslide
from landslides.seed_data import SEED_LANDSLIDES


class LandslideApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        for row in SEED_LANDSLIDES[:8]:
            Landslide.objects.create(**row)

    def test_list_shape_and_fields(self):
        response = self.client.get("/api/landslides/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 8)
        self.assertEqual(body["page"], 1)
        self.assertEqual(body["page_size"], 20)
        self.assertEqual(len(body["results"]), 8)
        item = body["results"][0]
        self.assertEqual(
            set(item),
            {"id", "name", "type", "lat", "lon", "length_m", "mission"},
        )

    def test_filter_type(self):
        response = self.client.get("/api/landslides/", {"type": "fall"})
        self.assertEqual(response.status_code, 200)
        types = {row["type"] for row in response.json()["results"]}
        self.assertTrue(types)
        self.assertEqual(types, {"fall"})

    def test_bbox_filter(self):
        response = self.client.get(
            "/api/landslides/",
            {"min_lat": -10, "max_lat": -5, "min_lon": -90, "max_lon": -80},
        )
        self.assertEqual(response.status_code, 200)
        ids = {row["id"] for row in response.json()["results"]}
        self.assertIn("ls_0001", ids)
        self.assertNotIn("ls_0002", ids)

    def test_empty_filter_returns_200(self):
        response = self.client.get("/api/landslides/", {"type": "not-a-type"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 0)
        self.assertEqual(response.json()["results"], [])

    def test_detail_extra_fields(self):
        response = self.client.get("/api/landslides/ls_0001/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in ("width_m", "features", "description", "source_url", "region"):
            self.assertIn(key, body)
        self.assertEqual(body["lat"], -7.2)

    def test_detail_404_chinese(self):
        response = self.client.get("/api/landslides/ls_9999/")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "未找到该滑坡"})

    def test_cors_allows_vite(self):
        response = self.client.get(
            "/api/landslides/",
            HTTP_ORIGIN="http://localhost:5173",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Origin"),
            "http://localhost:5173",
        )
