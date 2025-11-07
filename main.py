from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import tempfile
import noisereduce as nr
import soundfile as sf
import subprocess
import os
import base64

app = FastAPI()

# ✅ CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 ["http://localhost:5173"]로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Audio Analyzer + Denoiser Backend is running 🚀"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    tmp_path, wav_path, clean_path = None, None, None

    try:
        # 1️⃣ 업로드 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2️⃣ 모든 형식을 WAV로 변환 (ffmpeg 필요)
        wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # 3️⃣ 오디오 로드
        y, sr = librosa.load(wav_path, sr=None)

        # 4️⃣ 잡음 제거 (앞 0.5초를 노이즈로 가정)
        noise_sample = y[:int(0.5 * sr)] if len(y) > sr // 2 else y
        reduced = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr)

        # 5️⃣ 깨끗한 오디오 저장
        clean_path = wav_path.replace(".wav", "_clean.wav")
        sf.write(clean_path, reduced, sr)

        # 6️⃣ 분석
        duration = librosa.get_duration(y=reduced, sr=sr)
        rms = float(np.mean(librosa.feature.rms(y=reduced)))
        pitches, _ = librosa.piptrack(y=reduced, sr=sr)
        pitch_values = pitches[pitches > 0]
        mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0

        # 7️⃣ 파형 데이터 (단순화)
        step = max(1, len(reduced) // 200)
        waveform = [float(np.mean(np.abs(reduced[i:i + step]))) for i in range(0, len(reduced), step)]

        # 8️⃣ Base64로 인코딩 (React에서 다운로드 가능)
        with open(clean_path, "rb") as f:
            clean_b64 = base64.b64encode(f.read()).decode("utf-8")

        print("✅ 잡음 제거 + 분석 완료")

        # 9️⃣ JSON 응답
        return {
            "duration": round(duration, 2),
            "rms": round(rms, 4),
            "mean_pitch": round(mean_pitch, 2),
            "waveform": waveform,
            "summary": "✅ 잡음 제거 및 분석 완료!",
            "clean_audio_b64": clean_b64,
            "sample_rate": sr,
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

    finally:
        # 🔹 임시파일 정리
        for f in [tmp_path, wav_path, clean_path]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
