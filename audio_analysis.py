import io
import math
import wave
from typing import Optional

import numpy as np

try:
    from scipy.io import wavfile as scipy_wavfile
    SCIPY_WAV_AVAILABLE = True
except ImportError:
    SCIPY_WAV_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

MAX_ANALYSIS_HZ = 1400.0
MIN_USEFUL_HZ = 80.0
MAX_ANALYZED_DURATION_S = 30.0
MIN_RECOMMENDED_DURATION_S = 1.5

FREQ_BANDS_HZ = {
    "rumble": (10.0, 80.0),      # внешний низкочастотный шум (ветер, техника, транспорт)
    "calm": (100.0, 200.0),      # спокойный фон / покой / зимовка
    "active": (200.0, 300.0),    # рабочий гул, активный медосбор и вентиляция
    "swarm": (300.0, 550.0),     # роевое возбуждение и полоса пения матки (350..520 Гц)
    "stress": (550.0, 1400.0),   # резкий стресс, шипение, беспокойство, вредители
}

QUEEN_PIPING_BAND_HZ = (350.0, 520.0)

def _decode_pcm_bytes(raw_bytes: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64)
        return (arr - 128.0) / 128.0
    if sample_width == 2:
        arr = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float64)
        return arr / 32768.0
    if sample_width == 3:
        raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
        n_samples = len(raw_u8) // 3
        if n_samples == 0:
            return np.array([], dtype=np.float64)
        triplets = raw_u8[: n_samples * 3].reshape(-1, 3)
        vals = (
            triplets[:, 0].astype(np.int32)
            | (triplets[:, 1].astype(np.int32) << 8)
            | (triplets[:, 2].astype(np.int32) << 16)
        )
        vals = np.where(vals >= 0x800000, vals - 0x1000000, vals)
        return vals.astype(np.float64) / 8388608.0
    if sample_width == 4:
        arr = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float64)
        return arr / 2147483648.0
    raise ValueError(f"Неподдерживаемая разрядность WAV ({sample_width * 8} бит).")

def load_audio_bytes(file_bytes: bytes, file_name: str = "recording.wav") -> tuple[np.ndarray, int]:
    if not file_bytes:
        raise ValueError("Аудиофайл пуст.")

    name_low = (file_name or "").lower()

    if name_low.endswith(".wav") or file_bytes[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(file_bytes), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
            sig = _decode_pcm_bytes(raw, sample_width)
            if n_channels > 1 and len(sig) >= n_channels:
                sig = sig[: (len(sig) // n_channels) * n_channels].reshape(-1, n_channels).mean(axis=1)
            return sig.astype(np.float64), int(sample_rate)
        except Exception:
            if SCIPY_WAV_AVAILABLE:
                try:
                    sr, data = scipy_wavfile.read(io.BytesIO(file_bytes))
                    arr = np.asarray(data)
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)
                    if np.issubdtype(arr.dtype, np.integer):
                        info = np.iinfo(arr.dtype)
                        scale = float(max(abs(info.min), abs(info.max)))
                        sig = arr.astype(np.float64) / scale
                    else:
                        sig = arr.astype(np.float64)
                    return np.clip(sig, -1.0, 1.0), int(sr)
                except Exception:
                    pass

    if SOUNDFILE_AVAILABLE:
        try:
            data, sr = sf.read(io.BytesIO(file_bytes), dtype="float64", always_2d=False)
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            return np.clip(arr, -1.0, 1.0), int(sr)
        except Exception:
            pass

    raise ValueError(
        "Не удалось декодировать аудиофайл. Загрузите стандартную запись в формате WAV (PCM 16-bit)."
    )

def _compute_welch_psd(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    target_win = min(n, max(512, int(2 ** round(math.log2(max(sample_rate, 1024))))))
    win_len = min(n, target_win)
    hop = max(1, win_len // 2)

    window = np.hanning(win_len)
    win_norm = float(np.sum(window ** 2)) or 1.0

    starts = list(range(0, max(1, n - win_len + 1), hop))
    if not starts:
        starts = [0]

    acc_power = None
    for st_idx in starts:
        frame = signal[st_idx : st_idx + win_len]
        if len(frame) < win_len:
            padded = np.zeros(win_len, dtype=np.float64)
            padded[: len(frame)] = frame
            frame = padded
        frame = frame - np.mean(frame)
        spec = np.fft.rfft(frame * window)
        power = (np.abs(spec) ** 2) / (sample_rate * win_norm)
        acc_power = power if acc_power is None else (acc_power + power)

    psd = acc_power / len(starts)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / sample_rate)
    return freqs, psd

def _detect_queen_piping_stft(
    signal: np.ndarray,
    sample_rate: int,
    freqs: np.ndarray,
    psd: np.ndarray,
) -> dict:
    result = {
        "detected": False,
        "peak_hz": None,
        "pulse_count": 0,
        "continuous_tone_interference": False,
        "interference_hz": None,
    }

    q_mask = (freqs >= QUEEN_PIPING_BAND_HZ[0]) & (freqs <= QUEEN_PIPING_BAND_HZ[1])
    if not np.any(q_mask):
        return result

    q_freqs = freqs[q_mask]
    q_psd = psd[q_mask]
    bg_median = float(np.median(q_psd))
    if bg_median <= 0.0:
        return result

    peak_idx = int(np.argmax(q_psd))
    peak_val = float(q_psd[peak_idx])
    peak_hz = float(q_freqs[peak_idx])

    prominence_ratio = peak_val / (bg_median + 1e-18)
    narrow_mask = np.abs(q_freqs - peak_hz) <= 12.0
    concentration = float(np.sum(q_psd[narrow_mask])) / (float(np.sum(q_psd)) + 1e-18)

    if prominence_ratio < 10.0 or concentration < 0.28:
        return result

    win_len = max(256, int(2 ** round(math.log2(max(sample_rate * 0.25, 256)))))
    if len(signal) < win_len * 3:
        return result

    hop = max(1, win_len // 2)
    window = np.hanning(win_len)
    win_norm = float(np.sum(window ** 2)) or 1.0
    stft_freqs = np.fft.rfftfreq(win_len, d=1.0 / sample_rate)

    band_m = (stft_freqs >= QUEEN_PIPING_BAND_HZ[0]) & (stft_freqs <= QUEEN_PIPING_BAND_HZ[1])
    tone_m = np.abs(stft_freqs - peak_hz) <= 15.0
    if not np.any(band_m) or not np.any(tone_m):
        return result

    tone_series = []
    prom_series = []
    for st_idx in range(0, len(signal) - win_len + 1, hop):
        frame = signal[st_idx : st_idx + win_len]
        frame = frame - float(np.mean(frame))
        p_frame = (np.abs(np.fft.rfft(frame * window)) ** 2) / (sample_rate * win_norm)
        t_val = float(np.max(p_frame[tone_m]))
        b_med = float(np.median(p_frame[band_m])) + 1e-18
        tone_series.append(t_val)
        prom_series.append(t_val / b_med)

    tone_arr = np.asarray(tone_series, dtype=np.float64)
    prom_arr = np.asarray(prom_series, dtype=np.float64)
    if len(tone_arr) < 4:
        return result

    max_tone = float(np.max(tone_arr))
    if max_tone <= 0.0:
        return result

    active_frames = (tone_arr >= 0.25 * max_tone) & (prom_arr >= 10.0)
    active_ratio = float(np.mean(active_frames))

    padded_act = np.concatenate(([False], active_frames, [False]))
    starts = np.where(padded_act[1:] & ~padded_act[:-1])[0]
    ends = np.where(~padded_act[1:] & padded_act[:-1])[0]
    valid_pulses = [int(e - s) for s, e in zip(starts, ends) if (e - s) >= 1]

    cv_tone = float(np.std(tone_arr) / (np.mean(tone_arr) + 1e-18))

    # Непрерывная стационарная помеха (например, 400 Гц гул мотора/вентилятора): тон активен >=85% времени и ровный
    if active_ratio >= 0.85 and cv_tone < 0.35:
        result["continuous_tone_interference"] = True
        result["interference_hz"] = round(peak_hz, 1)
        return result

    # Биологический импульсный сигнал пения матки: >=2 отдельных импульса или прерывистая модуляция с паузами
    if (len(valid_pulses) >= 2 and 0.06 <= active_ratio <= 0.80) or (
        len(valid_pulses) >= 1 and active_ratio <= 0.65 and cv_tone >= 0.55
    ):
        result["detected"] = True
        result["peak_hz"] = round(peak_hz, 1)
        result["pulse_count"] = len(valid_pulses)
        return result

    return result

def _downsample_spectrum_for_chart(
    freqs: np.ndarray,
    psd: np.ndarray,
    max_hz: float = MAX_ANALYSIS_HZ,
    n_bins: int = 180,
) -> tuple[list[float], list[float]]:
    mask = (freqs >= 0.0) & (freqs <= max_hz)
    f_sub = freqs[mask]
    p_sub = psd[mask]
    if len(f_sub) == 0:
        return [], []

    edges = np.linspace(0.0, min(max_hz, float(f_sub[-1])), n_bins + 1)
    out_f = []
    out_db = []
    p_max = float(np.max(p_sub)) + 1e-18

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (f_sub >= lo) & (f_sub < hi if i < n_bins - 1 else f_sub <= hi)
        if np.any(in_bin):
            val = float(np.max(p_sub[in_bin]))
            out_f.append(round((lo + hi) * 0.5, 1))
            db_rel = 10.0 * math.log10((val + 1e-18) / p_max)
            out_db.append(round(max(-60.0, db_rel), 1))
    return out_f, out_db

def analyze_hive_audio(file_bytes: bytes, file_name: str = "recording.wav") -> dict:
    signal, sample_rate = load_audio_bytes(file_bytes, file_name)
    if sample_rate < 3000 or len(signal) < 256:
        return {
            "status": "signal_fault",
            "file_name": file_name,
            "duration_s": round(len(signal) / max(sample_rate, 1), 2),
            "sample_rate": sample_rate,
            "acoustic_state": "signal_fault",
            "state_label": "Некорректный аудиосигнал",
            "summary": "Слишком низкая частота дискретизации или повреждённая запись — анализ невозможен.",
            "quality_warnings": ["Аудиофайл повреждён или имеет частоту дискретизации ниже 3 кГц."],
            "metrics": {},
            "swarm_risk_delta": 0,
            "stress_flag": False,
            "spectrum_freqs_hz": [],
            "spectrum_rel_db": [],
        }

    total_duration_s = len(signal) / float(sample_rate)
    quality_warnings = []

    if total_duration_s < MIN_RECOMMENDED_DURATION_S:
        quality_warnings.append(
            f"Запись короткая ({total_duration_s:.1f} с; рекомендуется ~10 с) — точность оценки снижена."
        )

    if total_duration_s > MAX_ANALYZED_DURATION_S:
        max_samples = int(MAX_ANALYZED_DURATION_S * sample_rate)
        start = (len(signal) - max_samples) // 2
        signal = signal[start : start + max_samples]

    sig_centered = signal - float(np.mean(signal))
    rms = float(np.sqrt(np.mean(sig_centered ** 2)))
    clipping_ratio = float(np.mean(np.abs(signal) >= 0.99))

    sound_dbfs = round(float(max(-90.0, 20.0 * math.log10(rms + 1e-9))), 1)

    if clipping_ratio > 0.03:
        quality_warnings.append(
            f"Обнаружена перегрузка микрофона ({clipping_ratio * 100:.1f}% сэмплов в клиппинге) — спектр может быть искажён."
        )

    if rms < 1.5e-4:
        return {
            "status": "signal_fault",
            "file_name": file_name,
            "duration_s": round(total_duration_s, 1),
            "sample_rate": sample_rate,
            "acoustic_state": "signal_fault",
            "state_label": "Тишина / возможен сбой микрофона",
            "summary": (
                f"Уровень цифрового сигнала крайне низкий ({sound_dbfs} dBFS) — акустический гул пчёл не регистрируется. "
                "Проверьте подключение микрофона и состояние семьи очно."
            ),
            "quality_warnings": quality_warnings + ["Амплитуда сигнала близка к нулю (тишина или обрыв микрофона)."],
            "metrics": {
                "sound_dbfs": sound_dbfs,
                "sound_peak_hz": 0.0,
                "spectral_centroid_hz": 0.0,
                "rumble_ratio": 0.0,
                "calm_band_ratio": 0.0,
                "active_band_ratio": 0.0,
                "sound_swarm_band_ratio": 0.0,
                "stress_band_ratio": 0.0,
                "queen_piping_detected": False,
                "queen_piping_hz": None,
                "queen_piping_pulses": 0,
                "continuous_tone_interference": False,
            },
            "swarm_risk_delta": 0,
            "stress_flag": False,
            "spectrum_freqs_hz": [],
            "spectrum_rel_db": [],
        }

    freqs, psd = _compute_welch_psd(sig_centered, sample_rate)

    def _band_energy(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sum(psd[m])) if np.any(m) else 0.0

    e_rumble = _band_energy(*FREQ_BANDS_HZ["rumble"])
    e_calm = _band_energy(*FREQ_BANDS_HZ["calm"])
    e_active = _band_energy(*FREQ_BANDS_HZ["active"])
    e_swarm = _band_energy(*FREQ_BANDS_HZ["swarm"])
    e_stress = _band_energy(*FREQ_BANDS_HZ["stress"])

    e_total_1400 = _band_energy(10.0, MAX_ANALYSIS_HZ) + 1e-18
    e_bee_total = (e_calm + e_active + e_swarm + e_stress) + 1e-18

    rumble_ratio = round(e_rumble / e_total_1400, 3)
    calm_ratio = round(e_calm / e_bee_total, 3)
    active_ratio = round(e_active / e_bee_total, 3)
    swarm_ratio = round(e_swarm / e_bee_total, 3)
    stress_ratio = round(e_stress / e_bee_total, 3)

    bee_mask = (freqs >= MIN_USEFUL_HZ) & (freqs <= MAX_ANALYSIS_HZ)
    if np.any(bee_mask):
        bee_f = freqs[bee_mask]
        bee_p = psd[bee_mask]
        peak_freq_hz = round(float(bee_f[int(np.argmax(bee_p))]), 1)
        centroid_hz = round(float(np.sum(bee_f * bee_p) / (np.sum(bee_p) + 1e-18)), 1)
    else:
        peak_freq_hz = 0.0
        centroid_hz = 0.0

    e_queen_band = _band_energy(*QUEEN_PIPING_BAND_HZ)
    queen_band_share = e_queen_band / e_bee_total

    if queen_band_share >= 0.12:
        piping_info = _detect_queen_piping_stft(sig_centered, sample_rate, freqs, psd)
    else:
        piping_info = {
            "detected": False,
            "peak_hz": None,
            "pulse_count": 0,
            "continuous_tone_interference": False,
            "interference_hz": None,
        }

    # Непрерывный тон считаем мешающей помехой только если он действительно доминирует (>=35% энергии пчелиной полосы),
    dominant_continuous_interference = (
        piping_info["continuous_tone_interference"] and queen_band_share >= 0.35
    )
    piping_info["continuous_tone_interference"] = dominant_continuous_interference

    queen_piping = piping_info["detected"]
    piping_hz = piping_info["peak_hz"]
    if dominant_continuous_interference:
        quality_warnings.append(
            f"Обнаружен доминирующий непрерывный тон на {piping_info['interference_hz']} Гц "
            "(вероятна техногенная помеха, а не импульсное пение матки) — исключён из признаков роения."
        )

    chart_freqs, chart_db = _downsample_spectrum_for_chart(freqs, psd)

    metrics = {
        "sound_dbfs": sound_dbfs,
        "sound_peak_hz": peak_freq_hz,
        "spectral_centroid_hz": centroid_hz,
        "rumble_ratio": rumble_ratio,
        "calm_band_ratio": calm_ratio,
        "active_band_ratio": active_ratio,
        "sound_swarm_band_ratio": swarm_ratio,
        "stress_band_ratio": stress_ratio,
        "queen_piping_detected": queen_piping,
        "queen_piping_hz": piping_hz,
        "queen_piping_pulses": piping_info["pulse_count"],
        "continuous_tone_interference": piping_info["continuous_tone_interference"],
    }

    if rumble_ratio >= 0.55:
        quality_warnings.append(
            f"Высокая доля низкочастотного шума ниже 80 Гц ({int(rumble_ratio * 100)}%) — возможен ветер или работа техники."
        )
        return {
            "status": "noisy_signal",
            "file_name": file_name,
            "duration_s": round(total_duration_s, 1),
            "sample_rate": sample_rate,
            "acoustic_state": "external_noise",
            "state_label": "Сильный внешний шум (ветер / техника)",
            "summary": (
                f"Более половины энергии записи ({int(rumble_ratio * 100)}%) приходится на низкочастотный гул ниже 80 Гц. "
                "По такому фрагменту ставить акустический вывод нельзя — опирайтесь на вес, температуру и CO₂."
            ),
            "quality_warnings": quality_warnings,
            "metrics": metrics,
            "swarm_risk_delta": 0,
            "stress_flag": False,
            "spectrum_freqs_hz": chart_freqs,
            "spectrum_rel_db": chart_db,
        }

    if piping_info["continuous_tone_interference"]:
        return {
            "status": "noisy_signal",
            "file_name": file_name,
            "duration_s": round(total_duration_s, 1),
            "sample_rate": sample_rate,
            "acoustic_state": "external_noise",
            "state_label": f"Подозрение на стационарную тональную помеху ({piping_info['interference_hz']} Гц)",
            "summary": (
                f"На частоте {piping_info['interference_hz']} Гц присутствует ровный непрерывный тон без биологической импульсности "
                "(уровень {sound_dbfs} dBFS). Сигнал не засчитан как роение или пение матки — проверьте отсутствие внешних источников гула."
            ).format(sound_dbfs=sound_dbfs),
            "quality_warnings": quality_warnings,
            "metrics": metrics,
            "swarm_risk_delta": 0,
            "stress_flag": False,
            "spectrum_freqs_hz": chart_freqs,
            "spectrum_rel_db": chart_db,
        }

    swarm_risk_delta = 0
    stress_flag = False

    if (swarm_ratio >= 0.42 and peak_freq_hz >= 295.0) or (queen_piping and swarm_ratio >= 0.30):
        acoustic_state = "swarm_excitation"
        if queen_piping:
            state_label = f"Роевое возбуждение + импульсное пение матки ({piping_hz} Гц)"
            swarm_risk_delta = 22
        else:
            state_label = "Роевое возбуждение (сдвиг спектра в 300–550 Гц)"
            swarm_risk_delta = 18
        summary = (
            f"Доминирующая частота смещена в роевую полосу ({peak_freq_hz} Гц, доля диапазона 300–550 Гц — "
            f"{int(swarm_ratio * 100)}%, уровень сигнала {sound_dbfs} dBFS). "
            + (
                f"Выявлены прерывистые тональные импульсы около {piping_hz} Гц ({piping_info['pulse_count']} имп.), "
                "характерные для пения матки (признак активного роевого периода, в том числе риска выхода повторного роя-вторака). "
                if queen_piping else ""
            )
            + "Это подтверждающий фактор в связке с температурой, CO₂ и весом."
        )
    elif stress_ratio >= 0.35 or peak_freq_hz >= 550.0:
        acoustic_state = "stress_alarm"
        state_label = "Повышенный стресс / тревожный шум (550–1400 Гц)"
        stress_flag = True
        swarm_risk_delta = 5
        summary = (
            f"Выражена высокочастотная составляющая ({int(stress_ratio * 100)}% энергии в 550–1400 Гц, "
            f"пик {peak_freq_hz} Гц, уровень {sound_dbfs} dBFS) — признак беспокойства, шипения, "
            "возможной потери матки или внешнего раздражителя."
        )
    elif active_ratio >= 0.38 or (200.0 <= peak_freq_hz < 300.0):
        acoustic_state = "active_ventilation"
        state_label = "Активный рабочий гул / вентиляция гнезда"
        summary = (
            f"Основная энергия сосредоточена в рабочем диапазоне 200–300 Гц ({int(active_ratio * 100)}%, "
            f"пик {peak_freq_hz} Гц, уровень {sound_dbfs} dBFS) — типичная картина активного медосбора и выпаривания нектара."
        )
    else:
        acoustic_state = "calm_normal"
        state_label = "Спокойный ровный фон"
        summary = (
            f"Ровный низкочастотный гул (пик {peak_freq_hz} Гц, доля 100–200 Гц — {int(calm_ratio * 100)}%, "
            f"уровень {sound_dbfs} dBFS). Акустических признаков роения или стресса нет."
        )

    return {
        "status": "ok",
        "file_name": file_name,
        "duration_s": round(total_duration_s, 1),
        "sample_rate": sample_rate,
        "acoustic_state": acoustic_state,
        "state_label": state_label,
        "summary": summary,
        "quality_warnings": quality_warnings,
        "metrics": metrics,
        "swarm_risk_delta": swarm_risk_delta,
        "stress_flag": stress_flag,
        "spectrum_freqs_hz": chart_freqs,
        "spectrum_rel_db": chart_db,
    }

def evaluate_table_acoustics(
    sound_db: Optional[float],
    sound_peak_hz: Optional[float],
    sound_swarm_band_ratio: Optional[float],
) -> Optional[dict]:
    if sound_db is None and sound_peak_hz is None and sound_swarm_band_ratio is None:
        return None

    anomalies = []
    if sound_db is not None and not (0.0 <= sound_db <= 130.0):
        anomalies.append(f"Уровень шума ({sound_db} дБ) выходит за физические пределы (0..130 дБ).")
    if sound_peak_hz is not None and not (0.0 <= sound_peak_hz <= 2000.0):
        anomalies.append(f"Пиковая частота звука ({sound_peak_hz} Гц) выходит за допустимый диапазон (0..2000 Гц).")
    if sound_swarm_band_ratio is not None and not (0.0 <= sound_swarm_band_ratio <= 1.0):
        anomalies.append(f"Доля роевого спектра ({sound_swarm_band_ratio}) должна быть в пределах 0..1.")

    if anomalies:
        return {
            "status": "signal_fault",
            "source": "table",
            "acoustic_state": "signal_fault",
            "state_label": "Сбой акустического датчика",
            "summary": " ".join(anomalies),
            "quality_warnings": anomalies,
            "metrics": {
                "sound_db": sound_db,
                "sound_peak_hz": sound_peak_hz,
                "sound_swarm_band_ratio": sound_swarm_band_ratio,
                "queen_piping_detected": False,
            },
            "swarm_risk_delta": 0,
            "stress_flag": False,
        }

    swarm_risk_delta = 0
    stress_flag = False

    if (sound_peak_hz is not None and 300.0 <= sound_peak_hz <= 550.0) and (
        sound_swarm_band_ratio is None or sound_swarm_band_ratio >= 0.35
    ):
        acoustic_state = "swarm_excitation"
        state_label = "Роевое возбуждение по акустике"
        swarm_risk_delta = 18
        summary = (
            f"Пиковая частота {sound_peak_hz} Гц в роевой полосе (300–550 Гц)"
            + (f", доля роевого спектра {int(sound_swarm_band_ratio * 100)}%" if sound_swarm_band_ratio is not None else "")
            + (f", уровень {sound_db} дБ" if sound_db is not None else "")
            + " — подтверждающий признак подготовки к роению."
        )
    elif sound_swarm_band_ratio is not None and sound_swarm_band_ratio >= 0.45:
        acoustic_state = "swarm_excitation"
        state_label = "Повышенная энергия в роевой полосе"
        swarm_risk_delta = 15
        summary = f"Доля энергии в роевом диапазоне (300–550 Гц) достигла {int(sound_swarm_band_ratio * 100)}%."
    elif sound_peak_hz is not None and sound_peak_hz > 550.0:
        acoustic_state = "stress_alarm"
        state_label = "Тревожный высокочастотный шум"
        stress_flag = True
        summary = (
            f"Пик спектра смещён в зону тревоги ({sound_peak_hz} Гц"
            + (f", {sound_db} дБ" if sound_db is not None else "")
            + ") — возможное беспокойство или стресс семьи."
        )
    elif sound_peak_hz is not None and 200.0 <= sound_peak_hz < 300.0:
        acoustic_state = "active_ventilation"
        state_label = "Активный рабочий гул / вентиляция"
        summary = (
            f"Рабочая частота {sound_peak_hz} Гц"
            + (f", уровень {sound_db} дБ" if sound_db is not None else "")
            + " — нормальный режим медосбора и вентиляции."
        )
    else:
        acoustic_state = "calm_normal"
        state_label = "Спокойный фон"
        summary = (
            "Акустические показатели в спокойном диапазоне"
            + (f" ({sound_peak_hz} Гц" if sound_peak_hz is not None else "")
            + (f", {sound_db} дБ)" if sound_peak_hz is not None and sound_db is not None else (")" if sound_peak_hz is not None else ""))
            + "."
        )

    return {
        "status": "ok",
        "source": "table",
        "acoustic_state": acoustic_state,
        "state_label": state_label,
        "summary": summary,
        "quality_warnings": [],
        "metrics": {
            "sound_db": sound_db,
            "sound_peak_hz": sound_peak_hz,
            "sound_swarm_band_ratio": sound_swarm_band_ratio,
            "queen_piping_detected": False,
        },
        "swarm_risk_delta": swarm_risk_delta,
        "stress_flag": stress_flag,
    }
