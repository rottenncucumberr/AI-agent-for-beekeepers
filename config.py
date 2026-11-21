import os
import streamlit as st


def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


LLM_API_KEY = _get_secret("LLM_API_KEY", "")
LLM_BASE_URL = _get_secret("LLM_BASE_URL", "https://api.aitunnel.ru/v1/")

LLM_MODEL = _get_secret("LLM_MODEL", "deepseek-v4.1-flash-1017")
LLM_TEMPERATURE = 0.35
LLM_MAX_TOKENS = 2500
LLM_TIMEOUT = 45.0

_raw_kb_ids = "kb_tsifrovoy-uley-reglamenty-i-pchelovodstv-6caykf"
AITUNNEL_KB_IDS = [x.strip() for x in _raw_kb_ids.replace(";", ",").split(",") if x.strip()]

# "api" — клиентский поиск (/search), "server_tool" — серверный aitunnel:file_search
AITUNNEL_KB_MODE = os.getenv("AITUNNEL_KB_MODE", "api")
AITUNNEL_KB_MAX_RESULTS = int(os.getenv("AITUNNEL_KB_MAX_RESULTS", "4"))
AITUNNEL_KB_RERANK = None
AITUNNEL_KB_MIN_SCORE = 0.15

# Ориентиры микроклимата расплодного гнезда
TARGET_TEMP_C = 36.0
OVERHEAT_TEMP_C = 36.5
CRITICAL_TEMP_C = 38.0

TARGET_HUMIDITY_P = 55.0
HIGH_HUMIDITY_P = 70.0

TARGET_CO2_PPM = 600
HIGH_CO2_PPM = 1000

# Физические пределы валидности датчиков
SENSOR_LIMITS = {
    "temp_min": 0.0,
    "temp_max": 60.0,
    "hum_min": 5.0,
    "hum_max": 100.0,
    "weight_min": 5.0,
    "weight_max": 200.0,
    "co2_min": 250,
    "co2_max": 40000,
}
