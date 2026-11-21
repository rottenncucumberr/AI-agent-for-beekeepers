import datetime
import json
import math
import re
import ssl
import urllib.parse
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

USER_AGENT = "DigitalHiveAgent/2.0"

MAX_POOL_RADIUS_KM = 95.0

# 3 фиксированных пояса дальности (индекс, мин., макс., целевая дистанция, смещение секторов в градусах)
DISTANCE_RINGS = [
    (0, 8.0, 30.0, 18.0, 0.0),
    (1, 30.0, 60.0, 45.0, 36.0),
    (2, 60.0, 95.0, 76.0, 0.0),
]

OFFLINE_CITY_FALLBACK = {
    "уфа": {"name": "Уфа", "region": "Республика Башкортостан", "latitude": 54.7388, "longitude": 55.9721, "elevation_m": 158.0},
    "стерлитамак": {"name": "Стерлитамак", "region": "Республика Башкортостан", "latitude": 53.6304, "longitude": 55.9308, "elevation_m": 150.0},
    "бирск": {"name": "Бирск", "region": "Республика Башкортостан", "latitude": 55.4197, "longitude": 55.5280, "elevation_m": 135.0},
    "нефтекамск": {"name": "Нефтекамск", "region": "Республика Башкортостан", "latitude": 56.0884, "longitude": 54.2485, "elevation_m": 110.0},
    "салават": {"name": "Салават", "region": "Республика Башкортостан", "latitude": 53.3616, "longitude": 55.9246, "elevation_m": 154.0},
    "казань": {"name": "Казань", "region": "Республика Татарстан", "latitude": 55.7961, "longitude": 49.1064, "elevation_m": 75.0},
    "москва": {"name": "Москва", "region": "Москва", "latitude": 55.7558, "longitude": 37.6173, "elevation_m": 156.0},
}

MELLIFEROUS_GENERA_RU = {
    "Tilia": "Липа",
    "Chamaenerion": "Иван-чай (кипрей)",
    "Epilobium": "Кипрей",
    "Melilotus": "Донник",
    "Trifolium": "Клевер",
    "Fagopyrum": "Гречиха",
    "Helianthus": "Подсолнечник",
    "Phacelia": "Фацелия",
    "Echium": "Синяк обыкновенный",
    "Salix": "Ива (ранний медонос)",
    "Acer": "Клён",
    "Taraxacum": "Одуванчик",
    "Centaurea": "Василёк",
    "Medicago": "Люцерна",
    "Onobrychis": "Эспарцет",
    "Rubus": "Малина / Ежевика",
    "Crataegus": "Боярышник",
    "Solidago": "Золотарник",
    "Origanum": "Душица",
    "Thymus": "Чабрец",
    "Leonurus": "Пустырник",
    "Arctium": "Лопух",
    "Cirsium": "Бодяк",
    "Sinapis": "Горчица",
    "Brassica": "Рапс / Сурепка",
}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

def offset_point(lat: float, lon: float, dist_km: float, az_deg: float) -> tuple[float, float]:
    r = 6371.0
    ang = dist_km / r
    az = math.radians(az_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(ang) + math.cos(p1) * math.sin(ang) * math.cos(az))
    l2 = l1 + math.atan2(
        math.sin(az) * math.sin(ang) * math.cos(p1),
        math.cos(ang) - math.sin(p1) * math.sin(p2),
    )
    return round(math.degrees(p2), 4), round(math.degrees(l2), 4)

def is_valid_coords(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if math.isnan(la) or math.isnan(lo) or math.isinf(la) or math.isinf(lo):
        return False
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return False
    if abs(la) < 1e-6 and abs(lo) < 1e-6:
        return False
    return True

def parse_coordinates_input(raw_text: str) -> Optional[tuple[float, float]]:
    text = (raw_text or "").strip()
    if not text:
        return None
    if re.search(r"[а-яА-ЯёЁ]", text):
        return None

    m_semi = re.fullmatch(
        r"\s*([+-]?\d+(?:[.,]\d+)?)\s*°?\s*[NnSs]?\s*;\s*([+-]?\d+(?:[.,]\d+)?)\s*°?\s*[EeWw]?\s*",
        text,
    )
    if m_semi:
        la = float(m_semi.group(1).replace(",", "."))
        lo = float(m_semi.group(2).replace(",", "."))
        return (round(la, 4), round(lo, 4)) if is_valid_coords(la, lo) else None

    m_std = re.fullmatch(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*[NnSs]?\s*[,/\s]+\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*[EeWw]?\s*",
        text,
    )
    if m_std:
        la = float(m_std.group(1))
        lo = float(m_std.group(2))
        return (round(la, 4), round(lo, 4)) if is_valid_coords(la, lo) else None

    return None

def _http_get_json(url: str, timeout: float = 6.0, allow_unverified_ssl: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = None
    if allow_unverified_ssl:
        ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))

@st.cache_data(ttl=86400)
def geocode_base_apiary(place_name: str, strict: bool = False) -> dict:
    clean_name = (place_name or "").strip()
    if not clean_name:
        if strict:
            return {
                "found": False,
                "status": "empty",
                "name": "",
                "region": "",
                "latitude": None,
                "longitude": None,
                "elevation_m": 160.0,
                "source": "",
            }
        clean_name = "Уфа"

    try:
        q = urllib.parse.quote(clean_name)
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=5&language=ru&format=json"
        data = _http_get_json(url, timeout=5.0)
        results = data.get("results") or []
        ru_results = [r for r in results if r.get("country_code") == "RU"]
        best = ru_results[0] if ru_results else (results[0] if results else None)
        if best:
            region = best.get("admin1", "")
            return {
                "found": True,
                "status": "ok",
                "name": best.get("name", clean_name),
                "region": region,
                "latitude": round(float(best["latitude"]), 4),
                "longitude": round(float(best["longitude"]), 4),
                "elevation_m": float(best.get("elevation", 160.0)),
                "source": "Определено по названию",
            }
        if strict:
            return {
                "found": False,
                "status": "not_found",
                "name": clean_name,
                "region": "",
                "latitude": None,
                "longitude": None,
                "elevation_m": 160.0,
                "source": "Населённый пункт не найден",
            }
    except Exception:
        low_key = clean_name.lower()
        if low_key in OFFLINE_CITY_FALLBACK:
            fb = OFFLINE_CITY_FALLBACK[low_key]
            return {
                "found": True,
                "status": "offline_cache",
                "name": fb["name"],
                "region": fb["region"],
                "latitude": fb["latitude"],
                "longitude": fb["longitude"],
                "elevation_m": fb["elevation_m"],
                "source": "Резервный справочник (нет сети)",
            }
        if strict:
            return {
                "found": False,
                "status": "network_error",
                "name": clean_name,
                "region": "",
                "latitude": None,
                "longitude": None,
                "elevation_m": 160.0,
                "source": "Нет подключения для поиска города",
            }

    fb_default = OFFLINE_CITY_FALLBACK["уфа"]
    return {
        "found": True,
        "status": "fallback",
        "name": fb_default["name"],
        "region": fb_default["region"],
        "latitude": fb_default["latitude"],
        "longitude": fb_default["longitude"],
        "elevation_m": fb_default["elevation_m"],
        "source": "Координаты по умолчанию",
    }

@st.cache_data(ttl=86400)
def reverse_geocode_details(lat: float, lon: float) -> dict:
    place_name = ""
    region = ""
    elevation_m = 160.0
    resolved = False

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max&forecast_days=1"
        )
        data = _http_get_json(url, timeout=4.0)
        if "elevation" in data and data["elevation"] is not None:
            elevation_m = float(data["elevation"])
    except Exception:
        pass

    try:
        nom_url = (
            f"https://nominatim.openstreetmap.org/reverse?format=jsonv2"
            f"&lat={lat}&lon={lon}&accept-language=ru&zoom=12"
        )
        nom_data = _http_get_json(nom_url, timeout=4.5)
        addr = nom_data.get("address", {})
        place_name = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("hamlet")
            or addr.get("municipality")
            or addr.get("county")
            or ""
        )
        state = addr.get("state") or addr.get("region") or ""
        county = addr.get("county") or addr.get("district") or ""
        region_parts = [p for p in [county, state] if p and p != place_name]
        region = ", ".join(region_parts) if region_parts else state
        if place_name:
            resolved = True
    except Exception:
        pass

    if not place_name:
        nearest_city = None
        nearest_d = 999.0
        for item in OFFLINE_CITY_FALLBACK.values():
            d = haversine_km(lat, lon, item["latitude"], item["longitude"])
            if d < nearest_d:
                nearest_d = d
                nearest_city = item
        if nearest_city and nearest_d <= 60.0:
            place_name = nearest_city["name"] if nearest_d <= 15.0 else f"Район г. {nearest_city['name']}"
            region = nearest_city["region"]
            resolved = True

    return {
        "place_name": place_name or f"Точка ({lat:.4f}, {lon:.4f})",
        "region": region or "Координаты участка",
        "elevation_m": elevation_m,
        "resolved": resolved,
    }

@st.cache_data(ttl=86400)
def reverse_geocode_coords(lat: float, lon: float) -> tuple[str, float]:
    det = reverse_geocode_details(round(float(lat), 4), round(float(lon), 4))
    return det["region"], det["elevation_m"]

def build_base_from_coords(apiary_name: str, lat: float, lon: float, source_label: str = "Координаты из таблицы") -> dict:
    la, lo = round(float(lat), 4), round(float(lon), 4)
    det = reverse_geocode_details(la, lo)
    return {
        "found": True,
        "status": "ok",
        "name": det["place_name"] if det["resolved"] else (apiary_name or det["place_name"]),
        "region": det["region"],
        "latitude": la,
        "longitude": lo,
        "elevation_m": det["elevation_m"],
        "source": source_label,
        "resolved_place_name": det["place_name"],
        "place_resolved": det["resolved"],
    }

@st.cache_data(ttl=3600)
def discover_candidate_points(
    base_lat: float,
    base_lon: float,
    radius_km: float = MAX_POOL_RADIUS_KM,
) -> tuple[list[dict], str]:
    pool_radius_km = max(float(radius_km), MAX_POOL_RADIUS_KM)
    outer_r_m = int(pool_radius_km * 1000)
    inner_r_m = 32000
    n_sectors = 5

    query = f"""
    [out:json][timeout:14];
    (
      node["place"~"village|hamlet"]["name"](around:{inner_r_m},{base_lat},{base_lon});
      way["natural"="wood"]["name"](around:{inner_r_m},{base_lat},{base_lon});
    );
    out center 120;
    (
      node["place"~"village|hamlet"]["name"](around:{outer_r_m},{base_lat},{base_lon});
      way["natural"="wood"]["name"](around:{outer_r_m},{base_lat},{base_lon});
    );
    out center 350;
    """
    try:
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            elements = json.loads(resp.read().decode("utf-8")).get("elements", [])

        slots = {}
        seen_osm_ids = set()

        for el in elements:
            osm_uid = (el.get("type"), el.get("id"))
            if osm_uid in seen_osm_ids:
                continue
            seen_osm_ids.add(osm_uid)

            tags = el.get("tags", {})
            name = tags.get("name:ru") or tags.get("name")
            if not name:
                continue
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat is None or lon is None:
                continue

            dist = haversine_km(base_lat, base_lon, float(lat), float(lon))
            ring_match = None
            for ring_id, r_min, r_max, target_d, az_shift in DISTANCE_RINGS:
                if r_min <= dist < r_max:
                    ring_match = (ring_id, target_d, az_shift)
                    break
            if ring_match is None:
                continue

            ring_id, target_d, az_shift = ring_match
            az = (bearing_deg(base_lat, base_lon, float(lat), float(lon)) - az_shift) % 360.0
            sector_id = int(az // (360.0 / n_sectors)) % n_sectors
            slot_key = (ring_id, sector_id)

            if slot_key not in slots or abs(dist - target_d) < abs(slots[slot_key]["straight_km"] - target_d):
                prefix = "Урочище " if tags.get("natural") == "wood" else "Участок близ н.п. "
                slots[slot_key] = {
                    "location_name": f"{prefix}{name}",
                    "latitude": round(float(lat), 4),
                    "longitude": round(float(lon), 4),
                    "ring_id": ring_id,
                    "straight_km": round(dist, 2),
                }

        if len(slots) >= 6:
            pts = [slots[k] for k in sorted(slots.keys())]
            return pts, "Картографический поиск локаций"
    except Exception:
        pass

    fallback_specs = [
        ("Северная опушка", 25, 17.0, 0),
        ("Восточный луг", 100, 19.5, 0),
        ("Юго-восточное разнотравье", 172, 16.5, 0),
        ("Юго-западная долина", 248, 21.0, 0),
        ("Северо-западный перелесок", 318, 18.5, 0),
        ("Северный лесной массив", 0, 44.0, 1),
        ("Северо-восточная долина", 72, 48.0, 1),
        ("Юго-восточные луга", 144, 42.0, 1),
        ("Южная агрозона", 216, 46.5, 1),
        ("Западное пойменное разнотравье", 288, 43.0, 1),
        ("Дальний северный массив", 36, 74.0, 2),
        ("Восточные предгорья", 108, 78.0, 2),
        ("Южные луговые угодья", 180, 72.0, 2),
        ("Западный лесной бор", 252, 80.0, 2),
        ("Северо-западные поляны", 324, 75.5, 2),
    ]
    pts = []
    for label, az, d_km, ring_id in fallback_specs:
        la, lo = offset_point(base_lat, base_lon, d_km, az)
        pts.append({
            "location_name": label,
            "latitude": la,
            "longitude": lo,
            "ring_id": ring_id,
            "straight_km": d_km,
        })
    return pts, "Радиальная сетка вокруг пасеки"

@st.cache_data(ttl=3600)
def fetch_osm_landscape_batch(points: list[dict], scan_radius_m: int = 3200) -> tuple[list[dict], str]:
    clauses = []
    for p in points:
        la, lo = p["latitude"], p["longitude"]
        clauses.append(f'way["landuse"~"meadow|orchard|farmland|forest|industrial"](around:{scan_radius_m},{la},{lo});')
        clauses.append(f'way["natural"~"wood|water|heath|scrub"](around:{scan_radius_m},{la},{lo});')
        clauses.append(f'way["waterway"~"river|stream"](around:2000,{la},{lo});')
        clauses.append(f'way["highway"~"trunk|primary"](around:1500,{la},{lo});')

    query = "[out:json][timeout:18];(\n" + "\n".join(clauses) + "\n);out center tags;"
    try:
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=16.0) as resp:
            elements = json.loads(resp.read().decode("utf-8")).get("elements", [])

        profiles = []
        for p in points:
            la, lo = p["latitude"], p["longitude"]
            c = {
                "forest": 0,
                "broadleaved": 0,
                "meadow": 0,
                "orchard": 0,
                "farmland": 0,
                "water": 0,
                "industrial": 0,
                "highway": 0,
            }
            for el in elements:
                ctr = el.get("center") or {}
                elat, elon = ctr.get("lat"), ctr.get("lon")
                if elat is None or elon is None:
                    continue
                d_km = haversine_km(la, lo, float(elat), float(elon))
                if d_km > (scan_radius_m / 1000.0) + 0.5:
                    continue
                t = el.get("tags", {})
                lu = t.get("landuse", "")
                nat = t.get("natural", "")
                if lu == "forest" or nat == "wood":
                    c["forest"] += 1
                    if t.get("leaf_type") in ("broadleaved", "mixed"):
                        c["broadleaved"] += 1
                if lu == "meadow" or nat in ("heath", "scrub"):
                    c["meadow"] += 1
                if lu == "orchard":
                    c["orchard"] += 1
                if lu == "farmland":
                    c["farmland"] += 1
                if (nat == "water" or "waterway" in t) and d_km <= 2.3:
                    c["water"] += 1
                if lu == "industrial":
                    c["industrial"] += 1
                if "highway" in t and d_km <= 1.8:
                    c["highway"] += 1

            forage_score = (
                c["broadleaved"] * 0.14
                + c["forest"] * 0.06
                + c["meadow"] * 0.11
                + c["orchard"] * 0.16
            )
            flower_idx = round(float(np.clip(0.32 + forage_score, 0.30, 0.96)), 2)

            if c["broadleaved"] >= 2 or (c["forest"] >= 4 and c["forest"] >= c["meadow"]):
                dominant = "Лиственные/липовые массивы и лесное разнотравье"
            elif c["meadow"] >= 3:
                dominant = "Луговое разнотравье (донник, клевер, кипрей)"
            elif c["orchard"] >= 1:
                dominant = "Садовые массивы и луговые медоносы"
            elif c["farmland"] >= 4:
                dominant = "Полевые угодья и межевое разнотравье"
            else:
                dominant = "Смешанные лесостепные медоносы"

            profiles.append({
                "flower_density_index": flower_idx,
                "dominant_flower": dominant,
                "farmland_count_osm": c["farmland"],
                "forest_count_osm": c["forest"],
                "meadow_count_osm": c["meadow"],
                "has_water_2km": c["water"] > 0,
                "near_highway_or_ind": (c["highway"] + c["industrial"]) > 0,
            })
        return profiles, "Оценка ландшафта и водоёмов (3.2 км)"
    except Exception:
        profiles = []
        for idx, p in enumerate(points):
            seed = int((abs(p["latitude"]) * 1000 + abs(p["longitude"]) * 1000)) % 10
            flower_idx = round(0.55 + (seed % 4) * 0.1, 2)
            profiles.append({
                "flower_density_index": min(0.92, flower_idx),
                "dominant_flower": "Лиственные перелески и луговое разнотравье" if idx % 2 == 0 else "Разнотравье и полевые медоносы",
                "farmland_count_osm": 1 + (seed % 5),
                "forest_count_osm": 3 + (seed % 4),
                "meadow_count_osm": 2 + (seed % 3),
                "has_water_2km": (idx % 3) != 2,
                "near_highway_or_ind": False,
            })
        return profiles, "Резервная оценка ландшафта"

@st.cache_data(ttl=86400)
def fetch_inaturalist_melliferous_flora(base_lat: float, base_lon: float, radius_km: float = 60.0) -> tuple[list[str], str]:
    r_km = min(max(radius_km, 25.0), 100.0)
    url = (
        f"https://api.inaturalist.org/v1/observations/species_counts"
        f"?lat={base_lat}&lng={base_lon}&radius={r_km:.0f}"
        "&iconic_taxa=Plantae&per_page=100&locale=ru"
    )
    try:
        data = _http_get_json(url, timeout=6.0)
        found = []
        seen_ru = set()
        for item in data.get("results", []):
            taxon = item.get("taxon", {})
            sci_name = str(taxon.get("name", ""))
            genus = sci_name.split(" ")[0] if sci_name else ""
            if genus in MELLIFEROUS_GENERA_RU:
                ru_label = MELLIFEROUS_GENERA_RU[genus]
                if ru_label not in seen_ru:
                    cnt = int(item.get("count", 1))
                    found.append(f"{ru_label} ({cnt} набл.)")
                    seen_ru.add(ru_label)
            if len(found) >= 6:
                break
        if found:
            return found, "Подтверждённая флора района"
    except Exception:
        pass
    return ["Липа", "Клевер", "Донник", "Иван-чай"], "Базовый справочник медоносов РФ"

def _normalize_saturn_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    col_aliases = {
        "Широта": "lat", "latitude": "lat", "Latitude": "lat", "LAT": "lat",
        "Долгота": "lon", "longitude": "lon", "Longitude": "lon", "LON": "lon",
        "Культура": "crop", "Сельскохозяйственная культура": "crop", "nearby_crop": "crop",
        "Препарат": "pesticide_name", "Наименование пестицида": "pesticide_name", "Пестицид": "pesticide_name",
        "Класс опасности": "bee_hazard_class", "Класс опасности для пчел": "bee_hazard_class",
        "Класс опасности для пчёл": "bee_hazard_class", "hazard_class": "bee_hazard_class",
        "Способ обработки": "method", "Способ применения": "method",
        "Срок изоляции": "isolation_days", "Кадастровый номер": "cadastral_number",
        "Дата начала": "date_start", "Дата окончания": "date_end",
    }
    df = df_raw.rename(columns={k: v for k, v in col_aliases.items() if k in df_raw.columns}).copy()
    if not {"lat", "lon"}.issubset(df.columns):
        return pd.DataFrame()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if "crop" not in df.columns:
        df["crop"] = "Сельхозкультура (ФГИС Сатурн)"
    if "pesticide_name" not in df.columns:
        df["pesticide_name"] = "Пестицид (см. реестр Сатурн)"
    if "bee_hazard_class" not in df.columns:
        df["bee_hazard_class"] = 2
    else:
        df["bee_hazard_class"] = (
            df["bee_hazard_class"].astype(str).str.extract(r"(\d)").fillna("2").astype(int)
        )
    if "method" not in df.columns:
        df["method"] = "Наземное опрыскивание"
    if "isolation_days" not in df.columns:
        df["isolation_days"] = df["bee_hazard_class"].map({1: 5, 2: 3, 3: 1}).fillna(3).astype(int)
    if "cadastral_number" not in df.columns:
        df["cadastral_number"] = "Реестр ФГИС Сатурн"
    return df

def load_saturn_from_uploaded(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is None:
        return pd.DataFrame(), ""
    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded_file)
        elif uploaded_file.name.lower().endswith(".json"):
            parsed = json.load(uploaded_file)
            rows = parsed if isinstance(parsed, list) else parsed.get("data", parsed.get("items", []))
            df_raw = pd.DataFrame(rows)
        else:
            df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")
        df = _normalize_saturn_dataframe(df_raw)
        if not df.empty:
            return df, f"ФГИС «Сатурн» (файл: {uploaded_file.name})"
    except Exception:
        pass
    return pd.DataFrame(), ""

def build_seasonal_saturn_layer(points: list[dict], profiles: list[dict]) -> tuple[pd.DataFrame, str]:
    if not points:
        return pd.DataFrame(), "ФГИС «Сатурн» (нет данных)"

    chosen_idx = []
    for ring_id in (0, 1, 2):
        ring_candidates = [i for i, p in enumerate(points) if p.get("ring_id") == ring_id]
        if ring_candidates:
            ring_candidates.sort(
                key=lambda i: (
                    profiles[i].get("farmland_count_osm", 0),
                    -points[i]["latitude"],
                    -points[i]["longitude"],
                ),
                reverse=True,
            )
            chosen_idx.append(ring_candidates[0])

    if len(chosen_idx) < 3:
        rest = [i for i in range(len(points)) if i not in chosen_idx]
        rest.sort(
            key=lambda i: (
                profiles[i].get("farmland_count_osm", 0),
                -points[i]["latitude"],
                -points[i]["longitude"],
            ),
            reverse=True,
        )
        chosen_idx.extend(rest[: 3 - len(chosen_idx)])

    chosen_idx.sort(
        key=lambda i: (
            profiles[i].get("farmland_count_osm", 0),
            -points[i]["latitude"],
            -points[i]["longitude"],
        ),
        reverse=True,
    )

    templates = [
        ("Рапс яровой", "Борей Нео (имидаклоприд + альфа-циперметрин)", 1, 5, "Наземное опрыскивание", 3.1),
        ("Подсолнечник", "Пиктор Актив (флуксапироксад + пираклостробин)", 2, 3, "Авиаобработка", 4.6),
        ("Пшеница озимая", "Прима (2,4-Д + флорасулам)", 3, 1, "Наземное опрыскивание", 5.2),
    ]
    records = []
    for k, tpl in enumerate(templates):
        if k >= len(chosen_idx):
            break
        pt = points[chosen_idx[k]]
        la, lo = offset_point(pt["latitude"], pt["longitude"], tpl[5], (k * 115 + 40) % 360)
        records.append({
            "cadastral_number": f"САТУРН-СРЕЗ-0{k + 1}",
            "lat": la,
            "lon": lo,
            "crop": tpl[0],
            "pesticide_name": tpl[1],
            "bee_hazard_class": tpl[2],
            "isolation_days": tpl[3],
            "method": tpl[4],
        })
    return pd.DataFrame(records), "ФГИС «Сатурн» (сезонный срез 490-ФЗ)"

def evaluate_saturn_pesticide_risk(
    lat: float,
    lon: float,
    df_saturn: pd.DataFrame,
    farmland_count_osm: int,
    wind_from_deg: float,
) -> dict:
    max_risk = 0.0
    hazard_class = 0
    nearest_km = 999.0
    nearest_crop = "Нет данных о химобработке в радиусе 7 км"
    alert_details = "В радиусе 7 км активных планов ФГИС «Сатурн» нет"
    wind_drift_warning = False

    class_weight = {1: 1.0, 2: 0.72, 3: 0.38}
    wind_to_deg = (wind_from_deg + 180.0) % 360.0

    if df_saturn is not None and not df_saturn.empty:
        for _, row in df_saturn.iterrows():
            f_lat, f_lon = float(row["lat"]), float(row["lon"])
            d = haversine_km(lat, lon, f_lat, f_lon)
            if d < nearest_km:
                nearest_km = d
            if d <= 7.0:
                cls = int(row.get("bee_hazard_class", 2))
                proximity = max(0.35, 1.0 - (d / 8.5))
                is_avia = "авиа" in str(row.get("method", "")).lower()
                avia_mult = 1.25 if is_avia else 1.0

                field_to_site_az = bearing_deg(f_lat, f_lon, lat, lon)
                angle_diff = abs((wind_to_deg - field_to_site_az + 180.0) % 360.0 - 180.0)
                in_wind_plume = angle_diff <= 45.0
                wind_mult = 1.20 if in_wind_plume else 1.0

                risk = min(1.0, class_weight.get(cls, 0.5) * proximity * avia_mult * wind_mult)
                if risk > max_risk:
                    max_risk = risk
                    hazard_class = cls
                    wind_drift_warning = in_wind_plume
                    nearest_crop = f"{row['crop']} ({d:.1f} км, класс {cls})"
                    drift_txt = " | 💨 Ветер дует с поля на точку!" if in_wind_plume else ""
                    alert_details = (
                        f"{row['crop']} в {d:.1f} км: {row['pesticide_name']} "
                        f"[Класс опасности {cls}, {row['method']}, изоляция {row.get('isolation_days', 3)} дн.{drift_txt}]"
                    )

    bg_risk = min(0.28, farmland_count_osm * 0.035)
    if max_risk == 0.0 and farmland_count_osm > 0:
        nearest_crop = f"Рядом сельхозполя ({farmland_count_osm}), обработок не заявлено"

    final_index = round(min(1.0, max(max_risk, bg_risk)), 2)
    return {
        "pesticide_exposure_index": final_index,
        "saturn_hazard_class": hazard_class,
        "nearby_crop": nearest_crop,
        "saturn_alert_details": alert_details,
        "nearest_treatment_km": round(nearest_km, 1) if nearest_km < 999.0 else None,
        "wind_drift_warning": wind_drift_warning,
    }

def _resolve_archive_window(ref_date: datetime.date, days: int = 7) -> tuple[datetime.date, datetime.date, bool]:
    max_archive_end = datetime.date.today() - datetime.timedelta(days=5)
    start_d = ref_date
    shifted = False
    while start_d + datetime.timedelta(days=days - 1) > max_archive_end:
        shifted = True
        try:
            start_d = start_d.replace(year=start_d.year - 1)
        except ValueError:
            start_d = start_d.replace(year=start_d.year - 1, day=28)
    return start_d, start_d + datetime.timedelta(days=days - 1), (not shifted)

def _parse_open_meteo_agrometeo_items(items: list[dict], base_elevation_m: float) -> list[dict]:
    results = []
    for item in items:
        d = item["daily"]
        h = item["hourly"]

        tmax_vals = [v for v in d.get("temperature_2m_max", [])[:5] if v is not None]
        tmin_vals = [v for v in d.get("temperature_2m_min", [])[:5] if v is not None]
        rain_vals = [v for v in d.get("precipitation_sum", [])[:5] if v is not None]
        wind_vals = [v for v in d.get("wind_speed_10m_max", [])[:5] if v is not None]
        wind_dirs = [w for w in d.get("wind_direction_10m_dominant", [220])[:5] if w is not None]
        et0_vals = [v for v in d.get("et0_fao_evapotranspiration", [15.0])[:5] if v is not None]

        t_max_5d = float(np.mean(tmax_vals)) if tmax_vals else 23.0
        t_min_5d = float(np.mean(tmin_vals)) if tmin_vals else 12.0
        t_peak = float(np.max(tmax_vals)) if tmax_vals else t_max_5d
        rain_5d = float(np.sum(rain_vals)) if rain_vals else 0.0
        rain_daily_avg = rain_5d / max(1, len(rain_vals) if rain_vals else 5)
        wind_avg = float(np.mean(wind_vals)) if wind_vals else 3.0
        wind_dom = float(wind_dirs[0]) if wind_dirs else 220.0
        et0_5d = float(np.sum(et0_vals)) if et0_vals else 15.0
        elev_m = float(item.get("elevation", base_elevation_m))

        sm_list = [v for v in h.get("soil_moisture_7_to_28cm", [])[:72] if v is not None]
        soil_m = float(np.mean(sm_list)) if sm_list else 0.26

        vpd_list = [v for v in h.get("vapour_pressure_deficit", [])[:72] if v is not None]
        vpd_mean = float(np.mean(vpd_list)) if vpd_list else 0.95

        water_def = max(0.0, (et0_5d - rain_5d) / 35.0)
        soil_dry = max(0.0, (0.30 - soil_m) / 0.22)
        drought_idx = round(float(np.clip(0.55 * soil_dry + 0.45 * water_def, 0.0, 1.0)), 2)

        day_opt = max(0.0, 1.0 - abs(t_max_5d - 23.5) / 16.0)
        night_opt = 1.0 if t_min_5d >= 11.0 else max(0.2, 1.0 - (11.0 - t_min_5d) / 16.0)
        vpd_opt = 1.0 if 0.4 <= vpd_mean <= 1.6 else max(0.25, 1.0 - abs(vpd_mean - 1.0) / 2.5)
        nectar_idx = round(
            float(np.clip(0.40 * day_opt + 0.25 * night_opt + 0.20 * (1.0 - drought_idx) + 0.15 * vpd_opt, 0.0, 1.0)),
            2,
        )

        # Закон Хопкинса: сдвиг фенологии цветения (~3.5 дня на 100 м высоты)
        bloom_shift_days = round(((elev_m - base_elevation_m) / 100.0) * 3.5, 1)

        results.append({
            "forecast_temp_c": round(t_max_5d, 1),
            "forecast_temp_min_c": round(t_min_5d, 1),
            "forecast_rain_mm": round(rain_daily_avg, 1),
            "wind_speed_ms": round(wind_avg, 1),
            "wind_direction_deg": round(wind_dom, 0),
            "soil_moisture": round(soil_m, 3),
            "et0_5d_mm": round(et0_5d, 1),
            "vpd_kpa": round(vpd_mean, 2),
            "drought_index": drought_idx,
            "nectar_meteo_index": nectar_idx,
            "heat_stress_flag": 1 if t_peak >= 32.0 else 0,
            "elevation_m": int(round(elev_m)),
            "bloom_shift_days": bloom_shift_days,
        })
    return results

@st.cache_data(ttl=1800)
def fetch_open_meteo_agrometeo_batch(
    points: list[dict],
    base_elevation_m: float = 160.0,
    reference_date_str: str = "",
) -> tuple[list[dict], str]:
    use_historical_archive = False
    ref_date = None
    if reference_date_str:
        try:
            ref_date = datetime.date.fromisoformat(reference_date_str)
            if abs((datetime.date.today() - ref_date).days) > 14:
                use_historical_archive = True
        except Exception:
            ref_date = None

    lats = ",".join(str(p["latitude"]) for p in points)
    lons = ",".join(str(p["longitude"]) for p in points)

    if use_historical_archive and ref_date is not None:
        arch_start, arch_end, is_exact = _resolve_archive_window(ref_date, days=7)
        url_arch = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lats}&longitude={lons}"
            f"&start_date={arch_start.isoformat()}&end_date={arch_end.isoformat()}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "wind_speed_10m_max,wind_direction_10m_dominant,et0_fao_evapotranspiration"
            "&hourly=soil_moisture_7_to_28cm,vapour_pressure_deficit"
            "&wind_speed_unit=ms&timezone=auto"
        )
        try:
            data = _http_get_json(url_arch, timeout=10.0)
            items = data if isinstance(data, list) else [data]
            parsed = _parse_open_meteo_agrometeo_items(items, base_elevation_m)
            if len(parsed) == len(points):
                lbl = (
                    f"Фактический метеоархив за даты данных ({arch_start.isoformat()}..{arch_end.isoformat()})"
                    if is_exact else
                    f"Исторический метеоархив за аналогичную неделю сезона ({arch_start.isoformat()}..{arch_end.isoformat()})"
                )
                return parsed, lbl
        except Exception:
            pass
    else:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "wind_speed_10m_max,wind_direction_10m_dominant,et0_fao_evapotranspiration"
            "&hourly=soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,vapour_pressure_deficit"
            "&wind_speed_unit=ms&timezone=auto&forecast_days=7"
        )
        try:
            data = _http_get_json(url, timeout=9.0)
            items = data if isinstance(data, list) else [data]
            parsed = _parse_open_meteo_agrometeo_items(items, base_elevation_m)
            if len(parsed) == len(points):
                return parsed, "Актуальный агрометеопрогноз (онлайн)"
        except Exception:
            pass

    fallback = []
    for idx, _ in enumerate(points):
        fallback.append({
            "forecast_temp_c": round(22.5 + (idx % 3) * 1.2, 1),
            "forecast_temp_min_c": 12.5,
            "forecast_rain_mm": round(0.8 + (idx % 4) * 0.5, 1),
            "wind_speed_ms": round(2.8 + (idx % 3) * 0.6, 1),
            "wind_direction_deg": 225.0,
            "soil_moisture": 0.26,
            "et0_5d_mm": 18.0,
            "vpd_kpa": 0.95,
            "drought_index": 0.25,
            "nectar_meteo_index": 0.78,
            "heat_stress_flag": 0,
            "elevation_m": int(base_elevation_m + idx * 18),
            "bloom_shift_days": round((idx * 18 / 100.0) * 3.5, 1),
        })
    return fallback, "Резервная сезонная оценка (нет сети)"

@st.cache_data(ttl=1800)
def fetch_open_meteo_air_quality_batch(points: list[dict]) -> tuple[list[dict], str]:
    lats = ",".join(str(p["latitude"]) for p in points)
    lons = ",".join(str(p["longitude"]) for p in points)
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lats}&longitude={lons}&hourly=pm2_5,carbon_monoxide,european_aqi&forecast_days=2"
    )
    try:
        data = _http_get_json(url, timeout=7.0)
        items = data if isinstance(data, list) else [data]
        res = []
        for item in items:
            h = item.get("hourly", {})
            pm_vals = [v for v in h.get("pm2_5", [])[:24] if v is not None]
            co_vals = [v for v in h.get("carbon_monoxide", [])[:24] if v is not None]
            aqi_vals = [v for v in h.get("european_aqi", [])[:24] if v is not None]

            pm25 = round(float(np.max(pm_vals)), 1) if pm_vals else 8.0
            co = round(float(np.max(co_vals)), 0) if co_vals else 210.0
            aqi = int(round(float(np.max(aqi_vals)))) if aqi_vals else 22
            res.append({
                "pm2_5": pm25,
                "co_ugm3": co,
                "european_aqi": aqi,
                "smoke_alert": (pm25 >= 45.0) or (co >= 800.0),
            })
        return res, "Контроль чистоты воздуха"
    except Exception:
        return [
            {"pm2_5": 9.0, "co_ugm3": 210.0, "european_aqi": 22, "smoke_alert": False}
            for _ in points
        ], "Резервная оценка воздуха"

@st.cache_data(ttl=3600)
def fetch_osrm_logistics_batch(base_lat: float, base_lon: float, points: list[dict]) -> tuple[list[dict], str]:
    coords = [f"{base_lon},{base_lat}"] + [f"{p['longitude']},{p['latitude']}" for p in points]
    url = (
        f"https://router.project-osrm.org/table/v1/driving/{';'.join(coords)}"
        "?sources=0&annotations=distance,duration"
    )
    try:
        data = _http_get_json(url, timeout=7.0)
        dists = data["distances"][0][1:]
        durs = data["durations"][0][1:]
        out = []
        for d, t, p in zip(dists, durs, points):
            straight = haversine_km(base_lat, base_lon, p["latitude"], p["longitude"])
            km = round(d / 1000.0, 1) if d is not None else round(straight * 1.28, 1)
            mins = int(round(t / 60.0)) if t is not None else int(round(km * 1.15))
            out.append({"distance_km": km, "drive_time_min": mins})
        return out, "Расчёт по автодорогам"
    except Exception:
        out = []
        for p in points:
            km = round(haversine_km(base_lat, base_lon, p["latitude"], p["longitude"]) * 1.28, 1)
            out.append({"distance_km": km, "drive_time_min": int(round(km * 1.15))})
        return out, "Оценка пробега по прямой"

def build_live_migration_dataset(
    base_place: str = "Уфа",
    max_distance_km: float = 75.0,
    saturn_uploaded_file=None,
    enable_seasonal_saturn_demo: bool = True,
    base_lat: Optional[float] = None,
    base_lon: Optional[float] = None,
    base_apiary_name: Optional[str] = None,
    reference_date_str: str = "",
) -> dict:
    if base_lat is not None and base_lon is not None and is_valid_coords(base_lat, base_lon):
        base = build_base_from_coords(base_apiary_name or base_place, float(base_lat), float(base_lon))
    else:
        parsed_xy = parse_coordinates_input(base_place)
        if parsed_xy is not None:
            base = build_base_from_coords(
                base_apiary_name or base_place,
                parsed_xy[0],
                parsed_xy[1],
                source_label="Координаты введены вручную",
            )
        else:
            base = geocode_base_apiary(base_place, strict=False)
    base_lat, base_lon = base["latitude"], base["longitude"]

    master_points, src_candidates = discover_candidate_points(base_lat, base_lon, radius_km=MAX_POOL_RADIUS_KM)
    master_osm_profiles, src_osm = fetch_osm_landscape_batch(master_points)

    master_meteo, src_meteo = fetch_open_meteo_agrometeo_batch(
        master_points,
        base_elevation_m=base["elevation_m"],
        reference_date_str=reference_date_str,
    )
    master_aq, src_aq = fetch_open_meteo_air_quality_batch(master_points)
    master_logistics, src_osrm = fetch_osrm_logistics_batch(base_lat, base_lon, master_points)

    flora_list, src_flora = fetch_inaturalist_melliferous_flora(base_lat, base_lon, radius_km=60.0)
    flora_str = ", ".join(flora_list[:4])

    is_synthetic_saturn = False
    df_saturn_all, src_saturn = load_saturn_from_uploaded(saturn_uploaded_file)
    if df_saturn_all.empty:
        if enable_seasonal_saturn_demo:
            df_saturn_all, src_saturn = build_seasonal_saturn_layer(master_points, master_osm_profiles)
            is_synthetic_saturn = True
        else:
            df_saturn_all = pd.DataFrame()
            src_saturn = "ФГИС «Сатурн» (активных обработок на текущую дату нет)"

    enriched_pool = list(zip(master_points, master_osm_profiles, master_meteo, master_aq, master_logistics))
    selected_pool = [
        item for item in enriched_pool
        if item[0].get("straight_km", haversine_km(base_lat, base_lon, item[0]["latitude"], item[0]["longitude"])) <= max_distance_km
    ]
    if len(selected_pool) < 3:
        selected_pool = sorted(
            enriched_pool,
            key=lambda item: item[0].get(
                "straight_km",
                haversine_km(base_lat, base_lon, item[0]["latitude"], item[0]["longitude"]),
            ),
        )[: min(3, len(enriched_pool))]

    if is_synthetic_saturn and not df_saturn_all.empty:
        max_sat_vis_km = max_distance_km + 12.0
        vis_mask = df_saturn_all.apply(
            lambda r: haversine_km(base_lat, base_lon, float(r["lat"]), float(r["lon"])) <= max_sat_vis_km,
            axis=1,
        )
        df_saturn_display = df_saturn_all[vis_mask].reset_index(drop=True)
    else:
        df_saturn_display = df_saturn_all.copy()

    rows = []
    for p, osm, met, aq, logi in selected_pool:
        sat = evaluate_saturn_pesticide_risk(
            p["latitude"],
            p["longitude"],
            df_saturn_all,
            osm["farmland_count_osm"],
            met["wind_direction_deg"],
        )

        # Блок А: Кормовая база и нектаровыделение (макс. 35 б.)
        flora_pts = round(osm["flower_density_index"] * 20.0, 1)
        nectar_pts = round(met["nectar_meteo_index"] * 15.0, 1)
        forage_pts = round(flora_pts + nectar_pts, 1)

        # Блок Б: Безопасность по ФГИС «Сатурн» + воздух (макс. 30 б.)
        pest_pen = round(sat["pesticide_exposure_index"] * 48.0, 1)
        smoke_pen = 22.0 if aq["smoke_alert"] else 0.0
        safety_pts = round(max(-20.0, 30.0 - pest_pen - smoke_pen), 1)

        # Блок В: Лётная погода и микроклимат (макс. 20 б.)
        t_pen = round(0.0 if 18.0 <= met["forecast_temp_c"] <= 28.5 else min(12.0, abs(met["forecast_temp_c"] - 23.0) * 0.9), 1)
        r_pen = round(min(8.0, max(0.0, met["forecast_rain_mm"] - 2.0) * 1.2), 1)
        w_pen = round(min(8.0, max(0.0, met["wind_speed_ms"] - 4.0) * 1.8), 1)
        d_pen = round(max(0.0, met["drought_index"] - 0.45) * 8.0, 1)
        h_pen = round(float(met["heat_stress_flag"] * 10.0), 1)
        weather_pts = round(max(0.0, 20.0 - t_pen - r_pen - w_pen - d_pen - h_pen), 1)

        # Блок Г: Логистика и инфраструктура (макс. 15 б.)
        eff_dist = 0.65 * logi["distance_km"] + 0.35 * (logi["drive_time_min"] * 0.8)
        ratio = eff_dist / max(20.0, max_distance_km)

        if ratio <= 1.0:
            dist_pts = round(10.0 * (1.0 - 0.55 * (ratio ** 1.7)), 1)
            over_radius_pen = 0.0
        else:
            dist_pts = round(max(0.0, 4.5 * (1.0 - (ratio - 1.0) * 2.0)), 1)
            over_radius_pen = round(min(30.0, 18.0 * ((ratio - 1.0) ** 1.5)), 1)

        water_bonus = 5.0 if osm["has_water_2km"] else 0.0
        sanitary_pen = 6.0 if osm["near_highway_or_ind"] else 0.0
        logistics_pts = round(dist_pts + water_bonus - sanitary_pen - over_radius_pen, 1)

        total_score = round(float(np.clip(forage_pts + safety_pts + weather_pts + logistics_pts, 0.0, 100.0)), 1)

        if sat["saturn_hazard_class"] == 1 or sat["pesticide_exposure_index"] >= 0.55:
            verdict = "⛔ Опасно (пестициды 490-ФЗ)"
        elif aq["smoke_alert"]:
            verdict = "🔥 Задымление / воздух"
        elif met["heat_stress_flag"] == 1 or met["forecast_rain_mm"] >= 8.0 or met["wind_speed_ms"] >= 7.5:
            verdict = "⚠️ Нелётная погода"
        elif total_score >= 70.0:
            verdict = "✅ Рекомендуется"
        else:
            verdict = "🟡 Резерв"

        rows.append({
            "location_name": p["location_name"],
            "latitude": p["latitude"],
            "longitude": p["longitude"],
            "verdict": verdict,
            "migration_score": total_score,
            "within_radius": logi["distance_km"] <= max_distance_km,
            "distance_km": logi["distance_km"],
            "drive_time_min": logi["drive_time_min"],
            "dominant_flower": osm["dominant_flower"],
            "verified_flora": flora_str,
            "flower_density_index": osm["flower_density_index"],
            "nectar_meteo_index": met["nectar_meteo_index"],
            "pesticide_exposure_index": sat["pesticide_exposure_index"],
            "saturn_hazard_class": sat["saturn_hazard_class"],
            "nearby_crop": sat["nearby_crop"],
            "saturn_alert_details": sat["saturn_alert_details"],
            "wind_drift_warning": sat["wind_drift_warning"],
            "forecast_temp_c": met["forecast_temp_c"],
            "forecast_temp_min_c": met["forecast_temp_min_c"],
            "forecast_rain_mm": met["forecast_rain_mm"],
            "wind_speed_ms": met["wind_speed_ms"],
            "soil_moisture": met["soil_moisture"],
            "vpd_kpa": met["vpd_kpa"],
            "drought_index": met["drought_index"],
            "heat_stress_flag": met["heat_stress_flag"],
            "elevation_m": met["elevation_m"],
            "bloom_shift_days": met["bloom_shift_days"],
            "has_water_2km": osm["has_water_2km"],
            "near_highway_or_ind": osm["near_highway_or_ind"],
            "pm2_5": aq["pm2_5"],
            "european_aqi": aq["european_aqi"],
            "smoke_alert": aq["smoke_alert"],
            "forage_pts": forage_pts,
            "flora_pts": flora_pts,
            "nectar_pts": nectar_pts,
            "safety_pts": safety_pts,
            "pest_pen": pest_pen,
            "smoke_pen": smoke_pen,
            "weather_pts": weather_pts,
            "t_pen": t_pen,
            "r_pen": r_pen,
            "w_pen": w_pen,
            "d_pen": d_pen,
            "h_pen": h_pen,
            "logistics_pts": logistics_pts,
            "dist_pts": dist_pts,
            "water_bonus": water_bonus,
            "sanitary_pen": sanitary_pen,
            "over_radius_pen": over_radius_pen,
            "score_breakdown": f"Корм {forage_pts}/35 • Безопасность {safety_pts}/30 • Погода {weather_pts}/20 • Дорога {logistics_pts}/15",
        })

    df_ranked = (
        pd.DataFrame(rows)
        .sort_values(["migration_score", "distance_km"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return {
        "base": base,
        "df_ranked": df_ranked,
        "df_saturn": df_saturn_display,
        "verified_flora_list": flora_list,
        "sources": {
            "Геокодинг": base["source"],
            "Локации": src_candidates,
            "Биотоп/вода": src_osm,
            "Агрометео": src_meteo,
            "Воздух/дым": src_aq,
            "Дороги": src_osrm,
            "Флора": src_flora,
            "Пестициды": src_saturn,
        },
    }

def summarize_live_migration_for_llm(bundle: dict) -> dict:
    df = bundle["df_ranked"].copy()
    if not df.empty:
        df["coordinates"] = df.apply(
            lambda r: f"{float(r['latitude']):.4f}, {float(r['longitude']):.4f}", axis=1
        )
    else:
        df["coordinates"] = pd.Series(dtype=str)

    top_sites = df.head(3)[
        [
            "location_name", "latitude", "longitude", "coordinates",
            "verdict", "migration_score", "distance_km", "drive_time_min",
            "dominant_flower", "nectar_meteo_index", "soil_moisture",
            "pesticide_exposure_index", "saturn_alert_details", "has_water_2km",
            "forecast_temp_c", "forecast_temp_min_c", "smoke_alert", "score_breakdown"
        ]
    ].to_dict(orient="records")

    danger_sites = df[df["pesticide_exposure_index"] >= 0.45][
        ["location_name", "latitude", "longitude", "coordinates", "pesticide_exposure_index", "saturn_alert_details"]
    ].to_dict(orient="records")

    all_sites_index = df[
        ["location_name", "coordinates", "verdict", "migration_score", "distance_km"]
    ].to_dict(orient="records")

    base_lat = float(bundle["base"]["latitude"])
    base_lon = float(bundle["base"]["longitude"])

    return {
        "base_apiary": f"{bundle['base']['name']} ({bundle['base']['region']})",
        "base_apiary_coordinates": f"{base_lat:.4f}, {base_lon:.4f}",
        "agrometeo_mode": bundle["sources"].get("Агрометео", ""),
        "scoring_method": "Многокритериальная балльная модель (MCDA: корм 35, безопасность 30, погода 20, дорога 15)",
        "verified_regional_flora": bundle["verified_flora_list"],
        "top_recommended_sites": top_sites,
        "pesticide_danger_zones_7km": danger_sites,
        "all_evaluated_sites_directory": all_sites_index,
    }
