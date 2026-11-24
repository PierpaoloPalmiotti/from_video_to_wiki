import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import os
import sys
import re
import gc
import math
import time
import json
import warnings
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime

# Sopprimi i warnings
warnings.filterwarnings("ignore")

import llm_engine as llm
import perf


# ---------------------------------------------------------------------------
# CustomTkinter: tema e aspetto
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------------------------------------------------------------------
# Registra i percorsi DLL NVIDIA installati via pip (cublas, cudnn, ecc.)
# ---------------------------------------------------------------------------
_DLL_HANDLES = []   # riferimenti alle cartelle DLL aggiunte (Windows)


def _registra_nvidia_dll():
    """Aggiunge al PATH di sistema le cartelle lib/bin dei pacchetti nvidia-*
    installati via pip, così CTranslate2 può trovare cublas64_12.dll ecc.
    Serve solo su Windows (su macOS non c'è CUDA, su Linux vale LD_LIBRARY_PATH)."""
    if os.name != "nt":
        return

    def _aggiungi(cartella):
        if cartella not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cartella + os.pathsep + os.environ.get("PATH", "")
        try:  # Python 3.8+: il PATH non basta per le DLL dipendenti
            _DLL_HANDLES.append(os.add_dll_directory(cartella))
        except Exception:
            pass

    try:
        import importlib.metadata
        nvidia_packages = [
            d.metadata["Name"] for d in importlib.metadata.distributions()
            if (d.metadata["Name"] or "").startswith("nvidia-")
        ]
        for pkg in nvidia_packages:
            try:
                files = importlib.metadata.files(pkg)
                if not files:
                    continue
                pkg_dir = str(files[0].locate().parent)
                # Risali alla radice del pacchetto
                for subdir in ("lib", "bin"):
                    candidate = os.path.join(pkg_dir, subdir)
                    if os.path.isdir(candidate):
                        _aggiungi(candidate)
                # Prova anche le sottocartelle che contengono DLL
                for root, dirs, fnames in os.walk(pkg_dir):
                    if any(f.lower().endswith(".dll") for f in fnames):
                        _aggiungi(root)
            except Exception:
                continue
    except Exception:
        pass

_registra_nvidia_dll()

# ---------------------------------------------------------------------------
# Check opzionale faster-whisper
# ---------------------------------------------------------------------------
WHISPER_DISPONIBILE = False
try:
    from faster_whisper import WhisperModel
    WHISPER_DISPONIBILE = True
except ImportError:
    pass

MODELLI_WHISPER = [
    "tiny", "base", "small", "medium",
    "large-v2", "large-v3", "large-v3-turbo",
]
ESTENSIONI_AUDIO = ('.wav', '.mp3', '.ogg', '.opus', '.m4a', '.flac', '.wma', '.aac')
ESTENSIONI_VIDEO = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
                    '.m4v', '.mpg', '.mpeg', '.ts', '.3gp')

# Colori custom
COLORE_ACCENT = "#1f6aa5"
COLORE_VERDE = "#2d8659"
COLORE_ROSSO = "#c0392b"
COLORE_GRIGIO = "#4a4a4a"
COLORE_SFONDO_LOG = "#1a1a2e"
COLORE_TESTO_LOG = "#e0e0e0"
COLORE_TESTO_DIM = "#888888"
COLORE_ARANCIO = "#d4781f"
COLORE_GIALLO = "#b8860b"
COLORE_VIOLA = "#7d3c98"

PROGRESSO_FILE = "_progresso.json"
BATCH_PROGRESSO_FILE = "_batch_progresso.json"


# ===========================================================================
# Utility cleanup CUDA — evita crash a fine trascrizione
# ===========================================================================
def _libera_risorse_cuda(model_ref, dev, callback=None):
    """Libera esplicitamente il modello Whisper e svuota la cache CUDA.

    Senza questo passo, il GC di Python può liberare il modello in modo
    non deterministico (anche dopo che la GUI ha mostrato il messagebox),
    causando un crash silenzioso del processo su alcune configurazioni CUDA.
    """
    try:
        if model_ref is not None:
            del model_ref
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    if dev == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    if callback:
        callback("🧹 Risorse modello liberate")


# ===========================================================================
# FUNZIONI BACKEND (estrazione + splitting + trascrizione)
# ===========================================================================

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


_FFMPEG_CACHE = None


def find_ffmpeg():
    """Cerca ffmpeg (Windows: ffmpeg.exe nella cartella dell'app o nel PATH;
    macOS/Linux: PATH, Homebrew, /usr/bin). Il risultato viene messo in cache."""
    global _FFMPEG_CACHE
    if _FFMPEG_CACHE:
        return _FFMPEG_CACHE
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    possible_paths = [
        get_resource_path(exe),
        os.path.join(os.path.dirname(sys.executable), exe),
        shutil.which("ffmpeg"),
    ]
    if os.name != "nt":
        possible_paths += ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
    for path in possible_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            result = subprocess.run([path, '-version'], capture_output=True,
                                    timeout=5, **perf.SUBPROCESS_KW)
            if result.returncode == 0:
                _FFMPEG_CACHE = path
                return path
        except Exception:
            continue
    return None


def setup_moviepy():
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        return True
    return False


def get_file_size_mb(file_path):
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0


def ordine_naturale(nome):
    """Ordinamento naturale basato sull'ultimo numero presente nel nome."""
    numeri = re.findall(r'\d+', nome)
    return int(numeri[-1]) if numeri else 0


def split_audio_with_ffmpeg(input_path, output_dir, target_size_mb, base_name,
                            callback=None, stop_event=None):
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise Exception("FFmpeg non trovato")

    info_cmd = [ffmpeg_path, '-i', input_path, '-hide_banner']
    try:
        result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30,
                                **perf.SUBPROCESS_KW)
        output = result.stderr
        duration_line = None
        for line in output.split('\n'):
            if 'Duration:' in line:
                duration_line = line
                break
        if not duration_line:
            raise Exception("Impossibile determinare la durata del file audio")
        duration_part = duration_line.split('Duration:')[1].split(',')[0].strip()
        time_parts = duration_part.split(':')
        total_seconds = (float(time_parts[0]) * 3600 +
                         float(time_parts[1]) * 60 +
                         float(time_parts[2]))
    except Exception as e:
        raise Exception(f"Errore info file: {e}")

    file_size_mb = get_file_size_mb(input_path)
    mb_per_second = file_size_mb / total_seconds if total_seconds > 0 else 0
    segment_duration = (target_size_mb / mb_per_second) if mb_per_second > 0 else 600
    num_segments = math.ceil(total_seconds / segment_duration)

    segments_created = []
    for i in range(num_segments):
        if stop_event and stop_event.is_set():
            if callback:
                callback(f"\n⏹  Splitting interrotto dall'utente al segmento {i+1}/{num_segments}")
            break

        start_time = i * segment_duration
        if start_time >= total_seconds:
            break
        segment_filename = f"{base_name}_segmento_{i + 1}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        cmd = [
            ffmpeg_path, '-i', input_path,
            '-ss', str(start_time),
            '-t', str(min(segment_duration, total_seconds - start_time)),
            '-acodec', 'pcm_s16le', '-y', segment_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300,
                                    **perf.SUBPROCESS_KW)
            if result.returncode != 0:
                if callback:
                    callback(f"  ✗ Errore FFmpeg segmento {i+1}")
                continue
            segments_created.append(segment_path)
            if callback:
                callback(f"  ✓ Segmento {len(segments_created)}/{num_segments}: "
                         f"{segment_filename}")
        except subprocess.TimeoutExpired:
            if callback:
                callback(f"  ✗ Timeout segmento {i+1}")
        except Exception as e:
            if callback:
                callback(f"  ✗ Errore segmento {i+1}: {e}")

    return segments_created, segment_duration / 60, total_seconds


def estrai_e_splitta_video(video_path, output_dir, target_size_mb,
                           callback=None, stop_event=None):
    """Estrae l'audio da un video e lo segmenta in `output_dir`.

    Backend GUI-agnostico: usa solo `callback` per il log e `stop_event`
    per l'interruzione. È condiviso dal flusso singolo video e dal batch.

    Ritorna: (lista_segmenti, durata_media_min, tempo_splitting_secondi).
    Solleva Exception se il video non contiene audio.
    """
    from moviepy.video.io.VideoFileClip import VideoFileClip
    import contextlib, io

    t0 = time.time()
    if callback:
        callback("⏳ Caricamento video...")
    video = VideoFileClip(video_path)
    if video.audio is None:
        video.close()
        raise Exception("Il video non contiene audio.")

    if callback:
        callback("⏳ Estrazione audio...")
    audio = video.audio
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        temp_audio_path = tmp.name

    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
    except Exception:
        audio.write_audiofile(temp_audio_path)

    if stop_event and stop_event.is_set():
        audio.close()
        video.close()
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        if callback:
            callback("⏹  Interrotto dopo estrazione audio")
        return [], 0.0, time.time() - t0

    if callback:
        callback("⏳ Segmentazione audio...\n")

    segments, seg_dur_min, durata_audio_s = split_audio_with_ffmpeg(
        temp_audio_path, output_dir, target_size_mb, base_name,
        callback=callback, stop_event=stop_event)

    audio.close()
    video.close()
    try:
        os.unlink(temp_audio_path)
    except:
        pass

    dt = time.time() - t0
    video_mb = get_file_size_mb(video_path)
    if callback and dt > 0:
        callback(f"   ⏱  Estrazione + segmentazione: {formatta_tempo(dt)} · "
                 f"{video_mb / dt:.1f} MB/s · audio {durata_audio_s / 60:.1f} min "
                 f"({durata_audio_s / dt:.0f}x tempo reale)")
    perf.registra_performance(dict(
        fase="estrazione_audio", sorgente=os.path.basename(video_path),
        backend="ffmpeg/moviepy", device="cpu", audio_min=durata_audio_s / 60,
        input_mb=video_mb, tempo_s=dt,
        x_realtime=(durata_audio_s / dt) if dt > 0 else None,
        note="interrotto" if (stop_event and stop_event.is_set()) else ""))

    return segments, seg_dur_min, dt


def trova_file_audio(cartella):
    file_audio = [
        f for f in os.listdir(cartella)
        if os.path.isfile(os.path.join(cartella, f))
        and os.path.splitext(f)[1].lower() in ESTENSIONI_AUDIO
    ]
    file_audio.sort(key=ordine_naturale)
    return file_audio


def trova_file_video(cartella):
    """Ritorna i soli file video presenti direttamente nella cartella
    (non ricorsivo), ordinati in modo naturale."""
    file_video = [
        f for f in os.listdir(cartella)
        if os.path.isfile(os.path.join(cartella, f))
        and os.path.splitext(f)[1].lower() in ESTENSIONI_VIDEO
    ]
    file_video.sort(key=ordine_naturale)
    return file_video


# ---------------------------------------------------------------------------
# Checkpoint: salvataggio / caricamento progresso (per cartella di segmenti)
# ---------------------------------------------------------------------------

def _carica_progresso(cartella):
    path = os.path.join(cartella, PROGRESSO_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            pass
    return {"completati": [], "risultati": [], "risultati_dettaglio": [], "durata_totale": 0.0}


def _salva_progresso(cartella, progresso):
    path = os.path.join(cartella, PROGRESSO_FILE)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(progresso, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _rimuovi_progresso(cartella):
    path = os.path.join(cartella, PROGRESSO_FILE)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _nome_output(cartella):
    nome_cartella = os.path.basename(os.path.normpath(cartella))
    return f"{nome_cartella}_trascrizione.txt"


def _scrivi_file_trascrizione(cartella, risultati, risultati_dettaglio,
                               durata_totale, formato, tempo_trascrizione=0):
    """Scrive/sovrascrive il file di trascrizione con tutti i risultati disponibili."""
    if formato == "dettagliato":
        output = _formatta_dettagliato(risultati_dettaglio, durata_totale, tempo_trascrizione)
    else:
        output = "\n\n".join(r for r in risultati if r)

    nome_file_out = _nome_output(cartella)
    percorso_output = os.path.join(cartella, nome_file_out)
    with open(percorso_output, "w", encoding="utf-8") as f:
        f.write(output)
    return percorso_output


# ---------------------------------------------------------------------------
# Checkpoint di livello BATCH (per progetto Wiki)
# ---------------------------------------------------------------------------

def _carica_batch_progresso(cartella_progetto):
    path = os.path.join(cartella_progetto, BATCH_PROGRESSO_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completati": []}


def _salva_batch_progresso(cartella_progetto, dati):
    path = os.path.join(cartella_progetto, BATCH_PROGRESSO_FILE)
    try:
        os.makedirs(cartella_progetto, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dati, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Trascrizione con checkpoint + stop/pausa + file parziale
# ---------------------------------------------------------------------------

def _risolvi_device(device):
    """'auto' → cuda se disponibile (solo NVIDIA), altrimenti cpu.
    Su macOS faster-whisper non supporta GPU (Metal): resta CPU."""
    if device != "auto":
        return device
    if sys.platform == "darwin":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        try:
            import ctranslate2
            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"


def _descrivi_esecuzione_whisper(dev, compute, cpu_threads):
    if dev == "cuda":
        return f"GPU CUDA · {compute}"
    return f"CPU · {compute} · {perf.thread_whisper_effettivi(cpu_threads)} thread"


def _carica_whisper(modello_nome, dev, compute, cpu_threads, callback=None):
    """Carica WhisperModel con i thread richiesti; fallback CUDA → CPU."""
    t0 = time.time()
    if callback:
        callback(f"⏳ Caricamento modello '{modello_nome}' "
                 f"({_descrivi_esecuzione_whisper(dev, compute, cpu_threads)})...")
    try:
        model = WhisperModel(modello_nome, device=dev, compute_type=compute,
                             cpu_threads=int(cpu_threads or 0))
    except Exception as e:
        if dev != "cuda":
            raise
        if callback:
            callback(f"⚠ CUDA non disponibile ({e})")
            callback("⏳ Fallback automatico su CPU (int8)...")
        dev, compute = "cpu", "int8"
        model = WhisperModel(modello_nome, device=dev, compute_type=compute,
                             cpu_threads=int(cpu_threads or 0))
    if callback:
        callback(f"✓ Modello caricato in {time.time() - t0:.1f}s "
                 f"({_descrivi_esecuzione_whisper(dev, compute, cpu_threads)})\n")
    return model, dev, compute


def trascrivi_segmenti(cartella, modello_nome="large-v3-turbo", lingua=None,
                       device="auto", formato="txt", callback=None,
                       stop_event=None, pause_event=None,
                       video_size_mb=0, tempo_splitting=0,
                       model_precaricato=None, cpu_threads=0):
    """Trascrive i file audio di `cartella` con checkpoint, stop/pausa e file
    parziale. `cpu_threads`: thread CPU per CTranslate2 (0 = default, 4)."""
    if not WHISPER_DISPONIBILE:
        raise ImportError("faster-whisper non installato.\nInstalla con: pip install faster-whisper")

    dev = _risolvi_device(device)
    compute = "float16" if dev == "cuda" else "int8"

    # Se ci viene passato un modello già caricato (batch), lo riusiamo e NON
    # lo liberiamo qui: la gestione del ciclo di vita è del chiamante.
    model = model_precaricato
    model_di_proprieta = model is None

    try:
        if model is None:
            model, dev, compute = _carica_whisper(modello_nome, dev, compute,
                                                  cpu_threads, callback)
        esecuzione = _descrivi_esecuzione_whisper(dev, compute, cpu_threads)

        file_audio = trova_file_audio(cartella)
        if not file_audio:
            if callback:
                callback("⚠ Nessun file audio trovato.")
            return "", 0.0

        # Carica progresso precedente
        progresso = _carica_progresso(cartella)
        completati = set(progresso.get("completati", []))
        risultati = list(progresso.get("risultati", []))
        risultati_dettaglio = list(progresso.get("risultati_dettaglio", []))
        durata_totale = progresso.get("durata_totale", 0.0)

        file_da_fare = [f for f in file_audio if f not in completati]
        n_gia_fatti = len(file_audio) - len(file_da_fare)

        if n_gia_fatti > 0 and callback:
            callback(f"🔄 Ripresa: {n_gia_fatti}/{len(file_audio)} segmenti gia' trascritti, "
                     f"riprendo da segmento {n_gia_fatti + 1}\n")

        if not file_da_fare:
            if callback:
                callback("✅ Tutti i segmenti sono gia' stati trascritti.")
            percorso_output = _finalizza_trascrizione(
                cartella, risultati, risultati_dettaglio,
                durata_totale, file_audio, formato, 0, callback,
                video_size_mb=video_size_mb, tempo_splitting=tempo_splitting,
                esecuzione=esecuzione)
            return percorso_output, 0.0

        if callback:
            callback(f"📂 Trovati {len(file_audio)} file audio totali, "
                     f"{len(file_da_fare)} da trascrivere · esecuzione: {esecuzione}\n")

        t_inizio = time.time()
        interrotto = False
        durata_sessione = 0.0      # audio trascritto in QUESTA esecuzione (per le metriche)

        for i, nome_file in enumerate(file_da_fare, 1):
            # --- Check STOP ---
            if stop_event and stop_event.is_set():
                interrotto = True
                if callback:
                    callback(f"\n⏹  Trascrizione interrotta dall'utente dopo "
                             f"{n_gia_fatti + i - 1}/{len(file_audio)} segmenti")
                break

            # --- Check PAUSA ---
            if pause_event and pause_event.is_set():
                if callback:
                    callback(f"⏸  In pausa... (clicca ▶ Riprendi per continuare)")
                t_pausa = time.time()
                while pause_event.is_set():
                    if stop_event and stop_event.is_set():
                        interrotto = True
                        break
                    time.sleep(0.3)
                t_inizio += time.time() - t_pausa     # la pausa non conta nelle metriche
                if interrotto:
                    if callback:
                        callback(f"\n⏹  Trascrizione interrotta durante la pausa")
                    break
                if callback:
                    callback(f"▶  Ripresa trascrizione...")

            percorso = os.path.join(cartella, nome_file)
            indice_globale = n_gia_fatti + i
            if callback:
                callback(f"[{indice_globale}/{len(file_audio)}] {nome_file}...")

            try:
                t_seg = time.time()
                segmenti_gen, info = model.transcribe(
                    percorso, language=lingua, beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                    condition_on_previous_text=True, no_speech_threshold=0.6,
                )
                segmenti_lista = []
                testi = []
                for seg in segmenti_gen:   # la trascrizione avviene durante l'iterazione
                    t = seg.text.strip()
                    if t:
                        segmenti_lista.append({"inizio": seg.start, "fine": seg.end, "testo": t})
                        testi.append(t)
                dt_seg = time.time() - t_seg

                testo_unito = " ".join(testi)
                durata_totale += info.duration
                durata_sessione += info.duration
                risultati.append(testo_unito)
                risultati_dettaglio.append({
                    "file": nome_file, "lingua": info.language,
                    "durata": info.duration, "testo": testo_unito,
                    "segmenti": segmenti_lista,
                })
                completati.add(nome_file)

                # Salva checkpoint dopo ogni segmento
                _salva_progresso(cartella, {
                    "completati": list(completati),
                    "risultati": risultati,
                    "risultati_dettaglio": risultati_dettaglio,
                    "durata_totale": durata_totale,
                })

                stato = "✓" if testo_unito else "— silenzio"
                if callback:
                    vel = f"{info.duration / dt_seg:.1f}x" if dt_seg > 0 else "-"
                    callback(f"  {stato} | {info.language} | {info.duration:.1f}s audio "
                             f"in {dt_seg:.1f}s → {vel} tempo reale")
            except Exception as e:
                risultati.append("")
                if callback:
                    callback(f"  ✗ ERRORE: {e}")

        tempo_trascrizione = time.time() - t_inizio

        if durata_sessione > 0 and tempo_trascrizione > 0:
            perf.registra_performance(dict(
                fase="trascrizione", sorgente=os.path.basename(os.path.normpath(cartella)),
                modello=modello_nome, backend="faster-whisper", device=dev, compute=compute,
                thread=perf.thread_whisper_effettivi(cpu_threads, dev) or "",
                gpu_offload="cuda" if dev == "cuda" else "no",
                audio_min=durata_sessione / 60, input_mb=video_size_mb or None,
                tempo_s=tempo_trascrizione, x_realtime=durata_sessione / tempo_trascrizione,
                chiamate=len(file_da_fare), note="interrotto" if interrotto else ""))

        if interrotto:
            # Genera file di trascrizione parziale
            percorso_parziale = _scrivi_file_trascrizione(
                cartella, risultati, risultati_dettaglio,
                durata_totale, formato, tempo_trascrizione)

            n_completati = len(completati)
            if callback:
                callback(f"\n{'─'*50}")
                callback(f"💾 PROGRESSO SALVATO + FILE PARZIALE")
                callback(f"{'─'*50}")
                callback(f"   Segmenti completati:  {n_completati}/{len(file_audio)}")
                callback(f"   Rimanenti:            {len(file_audio) - n_completati}")
                callback(f"   File parziale:        {percorso_parziale}")
                callback(f"   Riprendi con 'Trascrivi segmenti' o 'Da cartella audio…'")
                callback(f"   (alla ripresa il file verra' aggiornato con i nuovi segmenti)")
            return "", tempo_trascrizione

        # Trascrizione completata al 100%
        percorso_output = _finalizza_trascrizione(
            cartella, risultati, risultati_dettaglio,
            durata_totale, file_audio, formato, tempo_trascrizione, callback,
            video_size_mb=video_size_mb, tempo_splitting=tempo_splitting,
            esecuzione=esecuzione, durata_sessione=durata_sessione)

        return percorso_output, tempo_trascrizione

    finally:
        # Libera il modello solo se l'abbiamo creato noi in questa chiamata.
        # In modalità batch il modello è condiviso e viene liberato dal
        # chiamante a fine ciclo.
        if model_di_proprieta:
            _libera_risorse_cuda(model, dev, callback=callback)


def _finalizza_trascrizione(cartella, risultati, risultati_dettaglio,
                            durata_totale, file_audio, formato,
                            tempo_trascrizione, callback,
                            video_size_mb=0, tempo_splitting=0,
                            esecuzione="", durata_sessione=None):
    """Scrive il file finale, pulisce audio e rimuove il checkpoint."""

    percorso_output = _scrivi_file_trascrizione(
        cartella, risultati, risultati_dettaglio,
        durata_totale, formato, tempo_trascrizione)

    n_ok = sum(1 for r in risultati if r)
    audio_misurato = durata_sessione if durata_sessione is not None else durata_totale
    if callback:
        callback(f"\n{'─'*50}")
        callback(f"📊 REPORT TRASCRIZIONE")
        callback(f"{'─'*50}")
        callback(f"   Segmenti trascritti:  {n_ok}/{len(file_audio)}")
        callback(f"   Audio totale:         {durata_totale/60:.1f} min")
        if esecuzione:
            callback(f"   Esecuzione:           {esecuzione}")
        callback(f"   Tempo trascrizione:   {formatta_tempo(tempo_trascrizione)}")
        if tempo_trascrizione > 0 and audio_misurato > 0:
            callback(f"   Velocita':            {audio_misurato/tempo_trascrizione:.1f}x tempo reale"
                     f"  ({tempo_trascrizione / (audio_misurato / 60):.1f} s per minuto di audio)")
        if video_size_mb > 0 and (tempo_splitting + tempo_trascrizione) > 0:
            sec_per_mb = (tempo_splitting + tempo_trascrizione) / video_size_mb
            callback(f"   Performance E2E:      {sec_per_mb:.2f} sec/MB")
        callback(f"   Salvato in:           {percorso_output}")
        if tempo_trascrizione > 0:
            callback(f"   📈 Metriche aggiunte a: {perf.PERF_CSV}")

    # Pulizia segmenti audio (solo se completato al 100%)
    if n_ok > 0:
        file_rimossi = 0
        for nome_file in file_audio:
            try:
                percorso_file = os.path.join(cartella, nome_file)
                if os.path.exists(percorso_file):
                    os.remove(percorso_file)
                    file_rimossi += 1
            except Exception as e:
                if callback:
                    callback(f"  ⚠ Impossibile eliminare {nome_file}: {e}")
        if callback:
            callback(f"\n🗑  Pulizia: {file_rimossi}/{len(file_audio)} file audio eliminati")

    # Rimuovi checkpoint
    _rimuovi_progresso(cartella)

    return percorso_output


def _formatta_dettagliato(risultati, durata_totale, tempo_elab):
    righe = []
    righe.append("=" * 70)
    righe.append("TRASCRIZIONE AUDIO")
    righe.append(f"Durata audio totale: {durata_totale/60:.1f} min")
    righe.append(f"Tempo elaborazione: {tempo_elab:.1f}s")
    if tempo_elab > 0:
        righe.append(f"Velocita': {durata_totale/tempo_elab:.1f}x tempo reale")
    righe.append("=" * 70)
    righe.append("")
    for r in risultati:
        righe.append(f"--- {r['file']} ({r['lingua']}, {r['durata']:.1f}s) ---")
        if not r["testo"]:
            righe.append("  [silenzio]")
        else:
            for seg in r.get("segmenti", []):
                m_i, s_i = divmod(int(seg["inizio"]), 60)
                m_f, s_f = divmod(int(seg["fine"]), 60)
                righe.append(f"  [{m_i:02d}:{s_i:02d} -> {m_f:02d}:{s_f:02d}] {seg['testo']}")
        righe.append("")
    righe.append("=" * 70)
    righe.append("TESTO COMPLETO")
    righe.append("=" * 70)
    righe.append("")
    righe.append("\n\n".join(r["testo"] for r in risultati if r["testo"]))
    return "\n".join(righe)


def formatta_tempo(secondi):
    if secondi < 60:
        return f"{secondi:.1f}s"
    elif secondi < 3600:
        m, s = int(secondi // 60), int(secondi % 60)
        return f"{m}m {s}s"
    else:
        h = int(secondi // 3600)
        m, s = int((secondi % 3600) // 60), int(secondi % 60)
        return f"{h}h {m}m {s}s"


# ===========================================================================
# LLM WIKI — generazione struttura a partire dalle trascrizioni
# ===========================================================================

def _slug(nome):
    s = re.sub(r'[^\w\-]+', '_', nome, flags=re.UNICODE).strip('_')
    # Limite di lunghezza: su Windows i percorsi oltre 260 caratteri falliscono
    return s[:80].rstrip('_') or "sorgente"


def _scrivi_sorgente_wiki(raw_dir, indice, video_name, testo, n_segmenti=None):
    """Scrive una trascrizione come sorgente markdown immutabile in raw/.
    Ritorna il percorso del file creato."""
    base = os.path.splitext(video_name)[0]
    nome_md = f"{indice:02d}_{_slug(base)}.md"
    path = os.path.join(raw_dir, nome_md)
    oggi = datetime.now().strftime('%Y-%m-%d')

    if not (testo or "").strip():
        corpo_testo = "_(Nessun parlato rilevato nella trascrizione.)_"
    else:
        corpo_testo = testo.strip()

    frontmatter = (
        "---\n"
        f"source: {video_name}\n"
        "type: transcript\n"
        f"date: {oggi}\n"
        f"index: {indice:02d}\n"
        + (f"segments: {n_segmenti}\n" if n_segmenti is not None else "") +
        "tags: [trascrizione, sorgente]\n"
        "---\n\n"
    )
    corpo = f"# Trascrizione — {base}\n\n{corpo_testo}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(frontmatter + corpo)
    return path


CLAUDE_SCHEMA = """# Schema della Wiki — Istruzioni per l'agente LLM

Questa cartella è una **LLM Wiki**: una base di conoscenza in markdown che TU
(l'agente) costruisci e mantieni a partire dalle trascrizioni video presenti in
`raw/`. L'utente cura le fonti e fa le domande; tu fai tutto il lavoro di sintesi,
collegamento e manutenzione. La cartella va aperta come vault in Obsidian.

A differenza di un classico RAG, qui la conoscenza non viene riscoperta a ogni
domanda: viene **compilata una volta e tenuta aggiornata**. La wiki è un artefatto
persistente che si arricchisce con ogni nuova sorgente e con ogni domanda.

## Struttura
- `raw/` — fonti **immutabili**: una trascrizione per video. Non modificarle mai.
- `wiki/` — pagine generate da te: sintesi, pagine entità, pagine concetto, confronti.
- `wiki/index.md` — catalogo di tutte le pagine. Aggiornalo a ogni ingest.
- `wiki/log.md` — registro cronologico append-only (ingest, query, lint).
- `CLAUDE.md` / `AGENTS.md` — questo schema. Co-evolvilo con l'utente nel tempo.

## Convenzioni
- Tutto in markdown, in italiano.
- Collegamenti con i wikilink di Obsidian: `[[Nome Pagina]]`.
- Nomi file: `entita_<nome>.md`, `concetto_<nome>.md`, `sintesi_<tema>.md`,
  `sorgente_<slug>.md`, `query_<data>_<slug>.md` (risposte archiviate).
- Le pagine con `generato_da: <modello>` nel frontmatter sono una **prima bozza**
  prodotta da un LLM locale dall'app "From Video to Wiki": verificale sulle
  fonti `raw/`, arricchiscile e correggile liberamente.
- Ogni pagina inizia con frontmatter YAML (vedi sotto).
- Quando un dato nuovo contraddice uno vecchio, segnalalo con un blocco
  `> [!warning]` e cita entrambe le fonti.
- Cita sempre le fonti `raw/` da cui proviene un'affermazione.

## Operazioni

### Ingest (una fonte alla volta, consigliato)
1. Leggi la trascrizione in `raw/`.
2. Discuti con l'utente i punti chiave.
3. Crea una pagina di sintesi della fonte: `wiki/sorgente_<slug>.md`.
4. Crea/aggiorna le pagine entità e concetto collegate (persone, progetti,
   decisioni, strumenti, temi ricorrenti).
5. Aggiorna i cross-reference esistenti.
6. Aggiorna `wiki/index.md`.
7. Aggiungi una riga a `wiki/log.md`:
   `## [AAAA-MM-GG] ingest | <titolo sorgente>`.
   Una singola fonte può toccare 10-15 pagine.

### Query
1. Leggi prima `wiki/index.md` per individuare le pagine rilevanti.
2. Apri le pagine, sintetizza la risposta con citazioni alle fonti `raw/`.
3. Se la risposta è preziosa, archiviala come nuova pagina in `wiki/` e
   aggiorna l'index: le esplorazioni devono accumularsi, non perdersi in chat.

### Lint (manutenzione periodica)
Cerca: contraddizioni tra pagine, affermazioni superate da fonti più recenti,
pagine orfane (senza link in entrata), concetti citati ma senza pagina propria,
cross-reference mancanti, lacune colmabili con una ricerca. Proponi nuove domande
da investigare e nuove fonti da cercare.

## Formati

### Frontmatter pagina wiki
```yaml
---
title: <titolo>
type: entita | concetto | sintesi | sorgente
fonti: [01_video1, 02_video2]
aggiornato: AAAA-MM-GG
tags: [...]
---
```

### index.md
Organizzato per categoria: Sorgenti, Entità, Concetti, Sintesi.
Ogni voce: `- [[pagina]] — riassunto in una riga`.

### log.md
Append-only. Ogni voce inizia con `## [AAAA-MM-GG] <tipo> | <titolo>`,
così è interrogabile con `grep "^## \\[" wiki/log.md | tail -5`.

## Flusso tipico per l'utente
1. Apri questa cartella come vault in Obsidian.
2. Apri un agente (es. Claude Code) nella stessa cartella.
3. "Leggi CLAUDE.md e fai l'ingest delle sorgenti in raw/, una alla volta."
4. Naviga il risultato in Obsidian (graph view, link, index).
"""

README_WIKI = """# LLM Wiki — Guida rapida

Questa cartella è un progetto **LLM Wiki**: una base di conoscenza in markdown
costruita a partire dalle trascrizioni dei tuoi video, mantenuta da un agente LLM
(es. Claude Code) e navigata in Obsidian.

## Come usarla
1. Apri **questa cartella** come vault in Obsidian.
2. Apri un agente LLM (es. Claude Code) nella stessa cartella.
3. Chiedi all'agente di leggere `CLAUDE.md` e di fare l'**ingest** delle sorgenti
   in `raw/`, una alla volta.
4. L'agente costruisce le pagine in `wiki/` e mantiene `wiki/index.md` e
   `wiki/log.md`.
5. Fai domande: l'agente risponde dalla wiki e archivia le risposte utili.

## Struttura
- `raw/` — trascrizioni dei video (fonti immutabili).
- `wiki/` — pagine generate dall'agente (sintesi, entità, concetti).
- `CLAUDE.md` / `AGENTS.md` — istruzioni per l'agente (lo schema).
- `_segmenti/` — cartelle di lavoro per video (puoi ignorarle/eliminarle).
"""


def genera_struttura_wiki(cartella_progetto, cartella_video_origine, callback=None,
                          voce_log=True):
    """Crea/aggiorna lo scaffold della LLM Wiki nel progetto.

    Non sovrascrive CLAUDE.md, README, index.md se già presenti (la wiki è
    di proprietà dell'agente): aggiunge solo ciò che manca e, se `voce_log`,
    logga il batch.
    """
    raw_dir = os.path.join(cartella_progetto, "raw")
    wiki_dir = os.path.join(cartella_progetto, "wiki")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wiki_dir, exist_ok=True)
    oggi = datetime.now().strftime('%Y-%m-%d')

    sorgenti_md = sorted(
        [f for f in os.listdir(raw_dir) if f.lower().endswith(".md")],
        key=ordine_naturale)

    # CLAUDE.md + AGENTS.md (solo se mancanti)
    claude_path = os.path.join(cartella_progetto, "CLAUDE.md")
    if not os.path.exists(claude_path):
        with open(claude_path, "w", encoding="utf-8") as f:
            f.write(CLAUDE_SCHEMA)
        try:
            with open(os.path.join(cartella_progetto, "AGENTS.md"),
                      "w", encoding="utf-8") as f:
                f.write(CLAUDE_SCHEMA)
        except Exception:
            pass

    # README_WIKI.md (solo se mancante)
    readme_path = os.path.join(cartella_progetto, "README_WIKI.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(README_WIKI)

    # wiki/index.md starter (solo se mancante)
    index_path = os.path.join(wiki_dir, "index.md")
    if not os.path.exists(index_path):
        righe = [
            "# Indice della Wiki", "",
            f"_Generato il {oggi}. Da qui in avanti lo mantiene l'agente LLM._", "",
            "## Sorgenti (raw)", "",
        ]
        for s in sorgenti_md:
            nome = os.path.splitext(s)[0]
            righe.append(f"- [[{nome}]] — _(da sintetizzare)_")
        righe += ["", "## Entità", "", "_(vuoto)_", "",
                  "## Concetti", "", "_(vuoto)_", "",
                  "## Sintesi", "", "_(vuoto)_", ""]
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(righe))

    # wiki/log.md: crea se manca, poi append della voce di batch
    log_path = os.path.join(wiki_dir, "log.md")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Log della Wiki\n\nRegistro cronologico append-only.\n\n")
    if voce_log:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"## [{oggi}] batch | preparate {len(sorgenti_md)} sorgenti "
                    f"da '{os.path.basename(os.path.normpath(cartella_video_origine))}'\n")
            f.write("- Sorgenti aggiornate in `raw/` e pronte per l'ingest.\n")
            f.write("- Prossimo passo: l'agente fa l'ingest una fonte alla volta.\n\n")

    if callback:
        callback(f"📚 Struttura Wiki pronta/aggiornata in: {cartella_progetto}")


# ===========================================================================
# BATCH — processa tutti i video di una cartella e prepara la Wiki
# ===========================================================================

def elabora_cartella_video(cartella_video, target_size_mb, modello, lingua,
                           device, formato, genera_wiki=True,
                           callback=None, stop_event=None, pause_event=None,
                           cpu_threads=0):
    """Processa TUTTI i video di una cartella (estrai → splitta → trascrivi)
    e prepara la struttura LLM Wiki con tutte le trascrizioni come sorgenti.

    È resumable a livello di video (checkpoint di batch) e a livello di
    segmento (checkpoint per cartella). Carica il modello Whisper UNA sola
    volta e lo riusa per tutti i video.

    Ritorna un dict di riepilogo (o None se non ci sono video).
    """
    file_video = trova_file_video(cartella_video)
    if not file_video:
        if callback:
            callback("⚠ Nessun file video trovato nella cartella.")
        return None

    nome_progetto = os.path.basename(os.path.normpath(cartella_video))
    cartella_progetto = os.path.join(cartella_video, f"_Wiki_{nome_progetto}")
    raw_dir = os.path.join(cartella_progetto, "raw")
    seg_root = os.path.join(cartella_progetto, "_segmenti")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(seg_root, exist_ok=True)

    batch_prog = _carica_batch_progresso(cartella_progetto)
    completati = set(batch_prog.get("completati", []))

    if callback:
        callback(f"📁 Cartella: {cartella_video}")
        callback(f"🎞  Video trovati: {len(file_video)}")
        if completati:
            callback(f"🔄 Già completati in un run precedente: {len(completati)}")
        callback(f"📦 Progetto Wiki: {cartella_progetto}\n")

    # --- Risoluzione device + caricamento UNICO del modello ---
    dev = _risolvi_device(device)
    compute = "float16" if dev == "cuda" else "int8"
    if callback:
        callback("(il modello Whisper viene caricato una sola volta per tutto il batch)")
    model, dev, compute = _carica_whisper(modello, dev, compute, cpu_threads, callback)

    t_batch0 = time.time()
    n_nuovi = 0

    try:
        for idx, video_name in enumerate(file_video, 1):
            if stop_event and stop_event.is_set():
                if callback:
                    callback(f"\n⏹  Batch interrotto prima del video {idx}/{len(file_video)}.")
                break

            if video_name in completati:
                if callback:
                    callback(f"⏭  [{idx}/{len(file_video)}] {video_name} già fatto — salto")
                continue

            if callback:
                callback(f"\n{'━'*50}")
                callback(f"🎬  [{idx}/{len(file_video)}] {video_name}")
                callback(f"{'━'*50}")

            video_path = os.path.join(cartella_video, video_name)
            base = os.path.splitext(video_name)[0]
            seg_dir = os.path.join(seg_root, f"{idx:02d}_{_slug(base)}")

            # --- Estrazione + splitting ---
            try:
                segments, seg_dur_min, t_split = estrai_e_splitta_video(
                    video_path, seg_dir, target_size_mb,
                    callback=callback, stop_event=stop_event)
            except Exception as e:
                if callback:
                    callback(f"  ✗ Errore su '{video_name}': {e} — passo al prossimo")
                continue

            if stop_event and stop_event.is_set():
                if callback:
                    callback("⏹  Batch interrotto durante lo splitting (progresso salvato).")
                break
            if not segments:
                if callback:
                    callback(f"  ⚠ Nessun segmento per '{video_name}' — salto")
                continue

            # --- Trascrizione (modello condiviso) ---
            video_mb = get_file_size_mb(video_path)
            percorso_txt, _ = trascrivi_segmenti(
                cartella=seg_dir, modello_nome=modello, lingua=lingua,
                device=dev, formato=formato, callback=callback,
                stop_event=stop_event, pause_event=pause_event,
                video_size_mb=video_mb, tempo_splitting=t_split,
                model_precaricato=model, cpu_threads=cpu_threads)

            if not percorso_txt:
                # Interrotto durante la trascrizione (progresso già salvato),
                # oppure nessun file: in entrambi i casi non marchiamo completato.
                if stop_event and stop_event.is_set():
                    if callback:
                        callback("⏹  Batch interrotto durante la trascrizione (progresso salvato).")
                    break
                continue

            # --- Scrivi la trascrizione come sorgente della Wiki ---
            try:
                with open(percorso_txt, "r", encoding="utf-8") as f:
                    testo = f.read()
            except Exception:
                testo = ""

            if genera_wiki:
                raw_md = _scrivi_sorgente_wiki(raw_dir, idx, video_name, testo, len(segments))
                if callback:
                    callback(f"  📝 Sorgente Wiki: {os.path.basename(raw_md)}")

            completati.add(video_name)
            n_nuovi += 1
            _salva_batch_progresso(cartella_progetto, {"completati": list(completati)})
            if callback:
                callback(f"  ✅ '{video_name}' completato")

    finally:
        # Libera il modello (condiviso) UNA volta sola, a fine batch.
        _libera_risorse_cuda(model, dev, callback=callback)

    # --- Genera/aggiorna struttura Wiki ---
    wiki_pronta = False
    if genera_wiki:
        raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".md")] \
            if os.path.isdir(raw_dir) else []
        if raw_files:
            genera_struttura_wiki(cartella_progetto, cartella_video, callback=callback)
            wiki_pronta = True

    return {
        "cartella_progetto": cartella_progetto,
        "n_video": len(file_video),
        "n_completati": len(completati),
        "n_nuovi": n_nuovi,
        "tempo": time.time() - t_batch0,
        "wiki_pronta": wiki_pronta,
        "interrotto": bool(stop_event and stop_event.is_set()),
    }


# ===========================================================================
# GUI — CustomTkinter
# ===========================================================================
SETTINGS_PATH = os.path.join(llm.APP_DIR, "settings.json")

CTX_SCELTE = {"Auto": "auto", "4K": 4096, "8K": 8192, "16K": 16384,
              "32K": 32768, "64K": 65536, "128K": 131072}
MAX_OUTPUT_SCELTE = ["1024", "2048", "4096", "8192"]
LINGUE_WHISPER = ["Auto-detect", "it (Italiano)", "en (English)", "es (Español)",
                  "fr (Français)", "de (Deutsch)", "pt (Português)"]
BRAIN_UGUALE = "(come Impostazioni)"
COLORE_TEAL = "#16808a"

# Differenze tra piattaforme
SU_MAC = sys.platform == "darwin"
SU_WINDOWS = os.name == "nt"
FONT_LOG = "Menlo" if SU_MAC else ("Consolas" if SU_WINDOWS else "DejaVu Sans Mono")
# faster-whisper non ha backend GPU su macOS (niente Metal): solo CPU
DEVICE_SCELTE = ["auto", "cpu"] if SU_MAC else ["auto", "cpu", "cuda"]
N_CPU = os.cpu_count() or 4
THREAD_WHISPER_AUTO = f"Auto ({perf.WHISPER_THREAD_DEFAULT})"
THREAD_WHISPER_SCELTE = [THREAD_WHISPER_AUTO] + [str(i) for i in range(1, N_CPU + 1)]
THREAD_LLM_SCELTE = ["Auto"] + [str(i) for i in range(1, N_CPU + 1)]
ACCEL_SCELTE = {"Auto (GPU se disponibile)": "auto", "Solo CPU": "cpu"}
COLORE_SOTTOFRAME = ("gray86", "gray20")


INFO_PARAMETRI = {
    # Trascrizione
    "Modello:": "Modello Whisper per la trascrizione. Più grande = più preciso ma più lento. "
                "large-v3-turbo è il miglior compromesso per italiano e inglese.",
    "Lingua:": "Lingua parlata nella registrazione. Auto-detect la riconosce da sola; "
               "forzarla evita errori su audio misti o rumorosi.",
    "Formato:": "txt = solo testo. dettagliato = testo con i timestamp di ogni frase.",
    "Device:": "Dove gira Whisper. auto = GPU NVIDIA se presente, altrimenti CPU. "
               "Su Mac Whisper usa sempre la CPU.",
    "MB/segmento:": "L'audio viene diviso in pezzi di questa dimensione. Pezzi piccoli = "
                    "checkpoint più frequenti (ripresa dopo uno stop), nessun impatto sulla qualità.",
    "Thread CPU:": "Core della CPU usati da Whisper. Il default della libreria è 4: alzarlo "
                   "accelera la trascrizione (es. 12 thread ≈ 1.7x più veloce su M5 Pro). "
                   f"Questa macchina ha {os.cpu_count() or '?'} core logici.",
    # LLM
    "Modello LLM:": "Il modello che scrive la nota. Più parametri (8B, 14B, 32B…) = note migliori "
                    "ma più RAM e più lentezza. Ollama lo scarica al primo uso.",
    "Tipo contenuto:": "Sceglie la struttura della nota: standup, riunione, lezione da studiare, "
                       "talk, intervista… Auto = il modello riconosce da solo il tipo.",
    "Lingua nota:": "Lingua in cui viene scritta la nota, indipendentemente da quella della registrazione.",
    "Contesto:": "Quanto testo il modello legge in una volta (istruzioni + trascrizione + risposta), "
                 "in token (~¾ di parola). Auto lo adatta al testo; se non basta, la trascrizione "
                 "viene divisa in blocchi. Più contesto = più RAM.",
    "Max output:": "Lunghezza massima della nota in token (4096 ≈ 3000 parole). "
                   "Se il log segnala 'risposta troncata', alzalo.",
    "Temperatura:": "Creatività del modello (0–2). Bassa (0.1–0.3) = fedele alla trascrizione; "
                    "alta = più libera ma con rischio di inventare.",
    "Thread LLM:": "Core CPU usati dall'LLM. Conta solo quando il modello gira su CPU. "
                   "Auto = sceglie Ollama.",
    "Accelerazione:": "GPU se disponibile = più veloce (Metal su Mac, NVIDIA su Windows). "
                      "Solo CPU = per test di performance sui thread o se il modello non sta in GPU.",
    # Singolo contenuto / Wiki
    "Video:": "Il file video (o audio dentro un video) da elaborare.",
    "Knowledge base:": "Facoltativa. Cartella Wiki in cui archiviare ogni nota: sorgente in raw/, "
                       "sintesi, pagine concetto collegate, index e log. Cresce a ogni contenuto.",
    "Cartella video:": "Cartella con più video da elaborare in sequenza (batch). "
                       "Il risultato va in '_Wiki_<nome>' dentro la cartella.",
    "Cervello Wiki:": "Il modello LLM che legge le trascrizioni e costruisce la Wiki "
                      "(sintesi, concetti, risposte). Può essere diverso da quello delle note.",
    "nota": "Dopo la trascrizione, l'LLM scrive una nota .md strutturata accanto alla "
            "trascrizione (e nella knowledge base, se impostata).",
    "struttura": "Crea la struttura Wiki per Obsidian: raw/ (trascrizioni), wiki/ (pagine), "
                 "CLAUDE.md/AGENTS.md (istruzioni per un agente).",
    "ingest": "Dopo il batch, il cervello sintetizza ogni trascrizione, crea pagine concetto/entità "
              "collegate e aggiorna index e log.",
    "scarica": "Scarica subito il modello selezionato (altrimenti viene scaricato al primo uso).",
    "refresh": "Ricarica l'elenco dei modelli già installati in Ollama.",
}


class Tooltip:
    """Fumetto esplicativo: compare al passaggio del mouse (o al clic) sul widget."""

    def __init__(self, widget, testo, ritardo=350, al_clic=True):
        self.widget, self.testo, self.ritardo = widget, testo, ritardo
        self._tip = self._job = None
        widget.bind("<Enter>", self._pianifica, add="+")
        widget.bind("<Leave>", self._nascondi, add="+")
        if al_clic:      # sulle icone ⓘ; sui pulsanti il clic esegue l'azione
            widget.bind("<Button-1>", self._alterna, add="+")
        else:
            widget.bind("<Button-1>", self._nascondi, add="+")

    def _pianifica(self, _=None):
        self._annulla()
        self._job = self.widget.after(self.ritardo, self._mostra)

    def _annulla(self):
        if self._job:
            try:
                self.widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _mostra(self):
        self._job = None
        if self._tip or not self.widget.winfo_exists():
            return
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass
        tk.Label(tw, text=self.testo, justify="left", wraplength=360,
                 bg="#333740", fg="#f2f2f2", relief="solid", bd=1,
                 padx=9, pady=6, font=("TkDefaultFont", 11)).pack()
        tw.update_idletasks()
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        sw, sh = tw.winfo_screenwidth(), tw.winfo_screenheight()
        x = min(x, sw - tw.winfo_reqwidth() - 8)
        if y + tw.winfo_reqheight() > sh - 8:
            y = self.widget.winfo_rooty() - tw.winfo_reqheight() - 4
        tw.wm_geometry(f"+{max(0, x)}+{max(0, y)}")
        tw.lift()
        self._tip = tw

    def _nascondi(self, _=None):
        self._annulla()
        if self._tip:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    def _alterna(self, _=None):
        if self._tip:
            self._nascondi()
        else:
            self._annulla()
            self._mostra()


def _icona_info(parent, testo):
    """Piccola icona ⓘ con tooltip."""
    ico = ctk.CTkLabel(parent, text="ⓘ", width=14, cursor="hand2",
                       text_color=("#1f6aa5", "#5fa8e8"), font=ctk.CTkFont(size=13))
    Tooltip(ico, testo)
    return ico


def _carica_impostazioni():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            dati = json.load(f)
        return dati if isinstance(dati, dict) else {}
    except Exception:
        return {}


def _prossimo_indice_raw(raw_dir):
    numeri = []
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            m = re.match(r"^(\d+)_", f)
            if m:
                numeri.append(int(m.group(1)))
    return max(numeri, default=0) + 1


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("From Video to Wiki")
        self._configura_finestra()

        self.cartella_segmenti = None
        self.in_esecuzione = False
        self.video_size_mb = 0.0
        self.tempo_splitting = 0.0
        self.tempo_trascrizione = 0.0
        self._video_path = None
        self._cartella_video_batch = None
        self._kb_path = None

        # Eventi per stop e pausa
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # Stato modelli LLM
        self._imp = _carica_impostazioni()
        self._modelli_custom = [t for t in self._imp.get("llm_custom", [])
                                if isinstance(t, str) and t.strip()]
        self._llm_specs = {}
        self._ollama_ok = None           # None = verifica in corso
        self._ollama_installati = None   # set di nomi normalizzati
        self._llm_label_prec = llm.DEFAULT_LLM_LABEL

        # Grid principale: header fisso, pagina scorrevole (impostazioni,
        # sezioni operative e log), status bar fissa.
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._crea_header()
        # Tutta l'interfaccia scorre con la barra laterale: le sezioni in alto
        # restano nella loro forma, il log sotto è alto quanto la finestra.
        self.pagina = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        self.pagina.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.pagina.grid_columnconfigure(0, weight=1)
        self._crea_sezione_impostazioni(self.pagina, 0)
        self._crea_sezioni_operative(self.pagina, 1)
        self._crea_sezione_log(self.pagina, 2)
        self._crea_status_bar()
        self._altezza_log_corrente = None
        self.bind("<Configure>", self._pianifica_adatta_log, add="+")

        self._applica_impostazioni()
        self._aggiorna_menu_modelli()
        self._aggiorna_stato_bottoni(False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._aggiorna_modelli_locali)
        self.after(100, self._carica_info_sistema)

    def _configura_finestra(self):
        """Dimensiona la finestra sullo schermo reale. Su Windows con scaling
        DPI (125%/150%) Tk riporta pixel fisici: si ragiona in pixel logici."""
        try:
            scala = float(ctk.ScalingTracker.get_window_scaling(self)) or 1.0
        except Exception:
            scala = 1.0
        sw = int(self.winfo_screenwidth() / scala)
        sh = int(self.winfo_screenheight() / scala)
        w = max(900, min(1280, sw - 40))
        h = max(560, min(1000, sh - 90))
        self.geometry(f"{w}x{h}")
        # La pagina scorre: la finestra può essere anche bassa
        self.minsize(min(1100, w), min(480, h))

    # -------------------------------------------------------------------
    # Navigazione nella pagina scorrevole
    # -------------------------------------------------------------------
    def _scorri_a(self, frazione):
        try:
            self.pagina._parent_canvas.yview_moveto(frazione)
        except Exception:
            pass

    def _vai_al_log(self):
        self.update_idletasks()
        self._scorri_a(1.0)

    def _vai_in_alto(self):
        self._scorri_a(0.0)

    def _pianifica_adatta_log(self, event=None):
        if event is not None and event.widget is not self:
            return
        if getattr(self, "_job_adatta_log", None):
            self.after_cancel(self._job_adatta_log)
        self._job_adatta_log = self.after(120, self._adatta_altezza_log)

    def _adatta_altezza_log(self):
        """Il log occupa tutta l'altezza visibile della pagina (meno la sua
        intestazione): scorrendo in basso lo si vede per intero."""
        self._job_adatta_log = None
        try:
            scala = float(ctk.ScalingTracker.get_widget_scaling(self)) or 1.0
        except Exception:
            scala = 1.0
        # Altezza dell'area visibile (il canvas), non del contenuto che scorre
        try:
            visibile = self.pagina._parent_canvas.winfo_height() / scala
        except Exception:
            visibile = (self.winfo_height() - 80) / scala
        nuova = int(max(260, visibile - 95))
        if nuova != self._altezza_log_corrente:
            self._altezza_log_corrente = nuova
            self.log_text.configure(height=nuova)

    def _rotella_log(self, event):
        """Rotella sul log: scorre il log; ai suoi estremi lascia scorrere la pagina."""
        tb = self.log_text._textbox
        if getattr(event, "num", None) in (4, 5):        # Linux
            passi = -3 if event.num == 4 else 3
        elif SU_MAC:
            passi = -event.delta
        else:
            passi = -int(event.delta / 120) * 3
        if passi == 0:
            return "break"
        inizio, fine = tb.yview()
        if (passi < 0 and inizio <= 0.0) or (passi > 0 and fine >= 1.0):
            return None                  # log al limite: scorre la pagina
        tb.yview_scroll(passi, "units")
        return "break"

    def _carica_info_sistema(self):
        def lavoro():
            testo = "🖥  " + perf.descrizione_breve()
            self.after(0, lambda: self.lbl_sistema.configure(text=testo))
        threading.Thread(target=lavoro, daemon=True).start()

    # -------------------------------------------------------------------
    # Helper widget
    # -------------------------------------------------------------------
    @staticmethod
    def _label(parent, testo, row, col, padx=(12, 5), pady=4, info=None):
        """Etichetta di un parametro + icona ⓘ con la spiegazione (se prevista)."""
        info = info or INFO_PARAMETRI.get(testo)
        if not info:
            lbl = ctk.CTkLabel(parent, text=testo, anchor="w")
            lbl.grid(row=row, column=col, padx=padx, pady=pady, sticky="w")
            return lbl
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row, column=col, padx=padx, pady=pady, sticky="w")
        lbl = ctk.CTkLabel(box, text=testo, anchor="w")
        lbl.pack(side="left")
        _icona_info(box, info).pack(side="left", padx=(3, 0))
        return lbl

    @staticmethod
    def _titolo(parent, testo, columnspan, size=14):
        ctk.CTkLabel(parent, text=testo, anchor="w",
                     font=ctk.CTkFont(size=size, weight="bold"),
                     ).grid(row=0, column=0, columnspan=columnspan,
                            padx=15 if size == 14 else 12,
                            pady=(10, 6) if size == 14 else (8, 4), sticky="w")

    @staticmethod
    def _nota_dim(parent, testo, row, columnspan, wraplength=500):
        ctk.CTkLabel(parent, text=testo, text_color=COLORE_TESTO_DIM,
                     font=ctk.CTkFont(size=11), anchor="w", justify="left",
                     wraplength=wraplength,
                     ).grid(row=row, column=0, columnspan=columnspan,
                            padx=15, pady=(2, 10), sticky="w")

    @staticmethod
    def _set_entry(entry, testo):
        entry.configure(state="normal")
        entry.delete(0, "end")
        if testo:
            entry.insert(0, testo)
        entry.configure(state="readonly")

    def _bottone_grigio(self, parent, testo, comando, width=40):
        return ctk.CTkButton(parent, text=testo, width=width, command=comando,
                             fg_color=COLORE_GRIGIO, hover_color="#5a5a5a")

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    def _crea_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(12, 4), sticky="ew")

        ctk.CTkLabel(
            header, text="🎬  From Video to Wiki",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left")
        ctk.CTkLabel(
            header, text="  estrai · trascrivi · sintetizza · costruisci la tua knowledge base",
            text_color=COLORE_TESTO_DIM, font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=(8, 0), pady=(6, 0))

        ctk.CTkSegmentedButton(
            header, values=["☀️ Light", "🌙 Dark"],
            command=self._cambia_tema,
            font=ctk.CTkFont(size=12),
            selected_color=COLORE_ACCENT,
        ).pack(side="right")
        self._bottone_grigio(header, "⤓ Log", self._vai_al_log, 70).pack(side="right", padx=(0, 10))
        self.lbl_sistema = ctk.CTkLabel(header, text="🖥  …", text_color=COLORE_TESTO_DIM,
                                        font=ctk.CTkFont(size=11))
        self.lbl_sistema.pack(side="right", padx=(0, 14), pady=(6, 0))

    def _cambia_tema(self, scelta):
        modo = "light" if "Light" in scelta else "dark"
        ctk.set_appearance_mode(modo)

    # -------------------------------------------------------------------
    # Sezione 1: Impostazioni (trascrizione + LLM)
    # -------------------------------------------------------------------
    def _crea_sezione_impostazioni(self, parent, row):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=0, padx=20, pady=(4, 5), sticky="ew")
        frame.grid_columnconfigure(0, weight=0)   # trascrizione: larghezza naturale
        frame.grid_columnconfigure(1, weight=1)   # LLM: prende lo spazio restante
        self._titolo(frame, "1 │ Impostazioni", columnspan=2)

        # ---------------- Trascrizione (faster-whisper) ----------------
        wh = ctk.CTkFrame(frame, fg_color=COLORE_SOTTOFRAME, corner_radius=8)
        wh.grid(row=1, column=0, padx=(15, 6), pady=(0, 12), sticky="nsew")
        self._titolo(wh, "🎙  Trascrizione · faster-whisper", columnspan=4, size=13)

        self._label(wh, "Modello:", 1, 0)
        self.modello_var = ctk.StringVar(value="large-v3-turbo")
        ctk.CTkOptionMenu(wh, variable=self.modello_var, values=MODELLI_WHISPER,
                          width=130).grid(row=1, column=1, padx=5, pady=4, sticky="w")

        self._label(wh, "Lingua:", 1, 2)
        self.lingua_var = ctk.StringVar(value="Auto-detect")
        ctk.CTkOptionMenu(wh, variable=self.lingua_var, values=LINGUE_WHISPER,
                          width=130).grid(row=1, column=3, padx=(5, 12), pady=4, sticky="w")

        self._label(wh, "Formato:", 2, 0)
        self.formato_var = ctk.StringVar(value="txt")
        ctk.CTkOptionMenu(wh, variable=self.formato_var, values=["txt", "dettagliato"],
                          width=130).grid(row=2, column=1, padx=5, pady=4, sticky="w")

        self._label(wh, "Device:", 2, 2)
        self.device_var = ctk.StringVar(value="auto")
        ctk.CTkOptionMenu(wh, variable=self.device_var, values=DEVICE_SCELTE,
                          width=130).grid(row=2, column=3, padx=(5, 12), pady=4, sticky="w")

        self._label(wh, "MB/segmento:", 3, 0)
        self.size_entry = ctk.CTkEntry(wh, width=70, justify="center")
        self.size_entry.grid(row=3, column=1, padx=5, pady=4, sticky="w")
        self.size_entry.insert(0, "5")

        self._label(wh, "Thread CPU:", 3, 2)
        self.whisper_thread_var = ctk.StringVar(value=THREAD_WHISPER_AUTO)
        ctk.CTkOptionMenu(wh, variable=self.whisper_thread_var, values=THREAD_WHISPER_SCELTE,
                          width=130).grid(row=3, column=3, padx=(5, 12), pady=4, sticky="w")

        if not WHISPER_DISPONIBILE:
            ctk.CTkLabel(
                wh, text="⚠  faster-whisper non installato — pip install faster-whisper",
                text_color=COLORE_ROSSO, font=ctk.CTkFont(size=11),
            ).grid(row=5, column=0, columnspan=4, padx=12, pady=(0, 8), sticky="w")

        # ---------------- Sintesi (LLM open source) ----------------
        lf = ctk.CTkFrame(frame, fg_color=COLORE_SOTTOFRAME, corner_radius=8)
        lf.grid(row=1, column=1, padx=(6, 15), pady=(0, 12), sticky="nsew")
        lf.grid_columnconfigure(5, weight=1)
        self._titolo(lf, "🧠  Sintesi · LLM open source in locale", columnspan=6, size=13)

        self._label(lf, "Modello:", 1, 0, info=INFO_PARAMETRI["Modello LLM:"])
        rm = ctk.CTkFrame(lf, fg_color="transparent")
        rm.grid(row=1, column=1, columnspan=5, padx=5, pady=4, sticky="w")
        self.llm_modello_var = ctk.StringVar(value=llm.DEFAULT_LLM_LABEL)
        self.llm_menu = ctk.CTkOptionMenu(
            rm, variable=self.llm_modello_var, values=[llm.DEFAULT_LLM_LABEL],
            width=340, dynamic_resizing=False, command=self._on_llm_scelto)
        self.llm_menu.pack(side="left")
        self.btn_llm_scarica = self._bottone_grigio(rm, "⬇ Scarica", self._scarica_modello_llm, 90)
        self.btn_llm_scarica.pack(side="left", padx=(8, 0))
        self.btn_llm_refresh = self._bottone_grigio(rm, "↻", self._aggiorna_modelli_locali, 34)
        self.btn_llm_refresh.pack(side="left", padx=(6, 0))
        Tooltip(self.btn_llm_scarica, INFO_PARAMETRI["scarica"], al_clic=False)
        Tooltip(self.btn_llm_refresh, INFO_PARAMETRI["refresh"], al_clic=False)

        self.llm_info = ctk.CTkLabel(lf, text="", text_color=COLORE_TESTO_DIM,
                                     font=ctk.CTkFont(size=11), anchor="w",
                                     justify="left", wraplength=560)
        self.llm_info.grid(row=2, column=0, columnspan=6, padx=12, pady=(0, 2), sticky="w")

        self._label(lf, "Tipo contenuto:", 3, 0)
        self.llm_tipo_var = ctk.StringVar(value=llm.DEFAULT_TIPO)
        ctk.CTkOptionMenu(lf, variable=self.llm_tipo_var, values=list(llm.TIPI_NOTA),
                          width=220, dynamic_resizing=False,
                          ).grid(row=3, column=1, columnspan=2, padx=5, pady=4, sticky="w")

        self._label(lf, "Lingua nota:", 3, 3)
        self.llm_lingua_var = ctk.StringVar(value="Italiano")
        ctk.CTkOptionMenu(lf, variable=self.llm_lingua_var, values=list(llm.LINGUE_NOTA),
                          width=160).grid(row=3, column=4, columnspan=2, padx=(5, 12),
                                          pady=4, sticky="w")

        self._label(lf, "Contesto:", 4, 0, pady=4)
        self.llm_ctx_var = ctk.StringVar(value="Auto")
        ctk.CTkOptionMenu(lf, variable=self.llm_ctx_var, values=list(CTX_SCELTE), width=90,
                          command=lambda _: self._aggiorna_info_llm(),
                          ).grid(row=4, column=1, padx=5, pady=4, sticky="w")

        self._label(lf, "Max output:", 4, 2, padx=(10, 5))
        self.llm_maxout_var = ctk.StringVar(value="4096")
        ctk.CTkOptionMenu(lf, variable=self.llm_maxout_var, values=MAX_OUTPUT_SCELTE,
                          width=90).grid(row=4, column=3, padx=5, pady=4, sticky="w")

        self._label(lf, "Temperatura:", 4, 4, padx=(10, 5))
        self.llm_temp_entry = ctk.CTkEntry(lf, width=60, justify="center")
        self.llm_temp_entry.grid(row=4, column=5, padx=(5, 12), pady=4, sticky="w")
        self.llm_temp_entry.insert(0, "0.3")

        self._label(lf, "Thread LLM:", 5, 0, pady=(0, 10))
        self.llm_thread_var = ctk.StringVar(value="Auto")
        ctk.CTkOptionMenu(lf, variable=self.llm_thread_var, values=THREAD_LLM_SCELTE, width=90,
                          ).grid(row=5, column=1, padx=5, pady=(0, 10), sticky="w")
        self._label(lf, "Accelerazione:", 5, 2, padx=(10, 5), pady=(0, 10))
        self.llm_accel_var = ctk.StringVar(value=list(ACCEL_SCELTE)[0])
        ctk.CTkOptionMenu(lf, variable=self.llm_accel_var, values=list(ACCEL_SCELTE), width=200,
                          command=lambda _: self._aggiorna_info_llm(),
                          ).grid(row=5, column=3, columnspan=3, padx=5, pady=(0, 10), sticky="w")

    # -------------------------------------------------------------------
    # Sezioni 2 e 3 affiancate
    # -------------------------------------------------------------------
    def _crea_sezioni_operative(self, parent, row):
        riga = ctk.CTkFrame(parent, fg_color="transparent")
        riga.grid(row=row, column=0, padx=20, pady=5, sticky="ew")
        riga.grid_columnconfigure((0, 1), weight=1, uniform="op")
        self._crea_sezione_singolo(riga)
        self._crea_sezione_wiki(riga)

    # Sezione 2: singolo contenuto ---------------------------------------
    def _crea_sezione_singolo(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, padx=(0, 5), sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)
        self._titolo(frame, "2 │ Singolo contenuto  ·  estrai → trascrivi → nota", columnspan=3)

        self._label(frame, "Video:", 1, 0, padx=(15, 5))
        self.entry_video = ctk.CTkEntry(frame, placeholder_text="Seleziona un file video...",
                                        state="readonly")
        self.entry_video.grid(row=1, column=1, padx=5, pady=4, sticky="ew")
        self._bottone_grigio(frame, "📁", self._browse_file).grid(
            row=1, column=2, padx=(5, 15), pady=4, sticky="w")

        self._label(frame, "Knowledge base:", 2, 0, padx=(15, 5))
        self.entry_kb = ctk.CTkEntry(
            frame, placeholder_text="(opzionale) cartella Wiki in cui archiviare le note",
            state="readonly")
        self.entry_kb.grid(row=2, column=1, padx=5, pady=4, sticky="ew")
        kb_btns = ctk.CTkFrame(frame, fg_color="transparent")
        kb_btns.grid(row=2, column=2, padx=(5, 15), pady=4, sticky="w")
        self._bottone_grigio(kb_btns, "📁", self._browse_kb).pack(side="left")
        self._bottone_grigio(kb_btns, "✕", self._rimuovi_kb, 30).pack(side="left", padx=(4, 0))

        self.nota_var = ctk.BooleanVar(value=True)
        riga_nota = ctk.CTkFrame(frame, fg_color="transparent")
        riga_nota.grid(row=3, column=0, columnspan=3, padx=15, pady=(6, 4), sticky="w")
        ctk.CTkCheckBox(
            riga_nota, text="🧠 Genera nota .md con LLM (modello e tipo dalle Impostazioni)",
            variable=self.nota_var, command=self._aggiorna_testo_bottoni,
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        _icona_info(riga_nota, INFO_PARAMETRI["nota"]).pack(side="left", padx=(4, 0))

        r1 = ctk.CTkFrame(frame, fg_color="transparent")
        r1.grid(row=4, column=0, columnspan=3, padx=15, pady=4, sticky="ew")
        self.btn_elabora = ctk.CTkButton(
            r1, text="▶  Solo estrai audio", command=self._process_video,
            fg_color=COLORE_ACCENT, hover_color="#174f7a",
            font=ctk.CTkFont(size=13, weight="bold"), height=36)
        self.btn_elabora.pack(side="left", padx=(0, 6))
        self.btn_elabora_e_trascrivi = ctk.CTkButton(
            r1, text="⚡ Elabora + Trascrivi + Nota", command=self._process_video_e_trascrivi,
            fg_color=COLORE_ARANCIO, hover_color="#b5651d",
            font=ctk.CTkFont(size=13, weight="bold"), height=36)
        self.btn_elabora_e_trascrivi.pack(side="left", expand=True, fill="x")

        r2 = ctk.CTkFrame(frame, fg_color="transparent")
        r2.grid(row=5, column=0, columnspan=3, padx=15, pady=4, sticky="ew")
        self.btn_trascrivi = ctk.CTkButton(
            r2, text="🎙 Trascrivi segmenti", command=self._trascrivi_segmenti,
            fg_color=COLORE_VERDE, hover_color="#1e6b43", height=32)
        self.btn_trascrivi.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.btn_trascrivi_cartella = ctk.CTkButton(
            r2, text="📂 Da cartella audio…", command=self._trascrivi_da_cartella,
            fg_color=COLORE_VERDE, hover_color="#1e6b43", height=32)
        self.btn_trascrivi_cartella.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.btn_nota_txt = ctk.CTkButton(
            r2, text="📝 Nota da trascrizione…", command=self._nota_da_file,
            fg_color=COLORE_TEAL, hover_color="#0f5f66", height=32)
        self.btn_nota_txt.pack(side="left", expand=True, fill="x")

        self._nota_dim(
            frame,
            "Senza knowledge base la nota è salvata accanto alla trascrizione. Con una "
            "knowledge base viene anche archiviata nella Wiki (raw/ + wiki/ + concetti + "
            "index), che cresce a ogni nuovo contenuto.",
            row=6, columnspan=3)

    # Sezione 3: Knowledge base / Wiki ------------------------------------
    def _crea_sezione_wiki(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=1, padx=(5, 0), sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)
        self._titolo(frame, "3 │ Knowledge Base  ·  cartella → LLM Wiki", columnspan=3)

        self._label(frame, "Cartella video:", 1, 0, padx=(15, 5))
        self.entry_cartella_video = ctk.CTkEntry(
            frame, placeholder_text="Seleziona una cartella con più video...", state="readonly")
        self.entry_cartella_video.grid(row=1, column=1, padx=5, pady=4, sticky="ew")
        self._bottone_grigio(frame, "📁", self._browse_cartella_video).grid(
            row=1, column=2, padx=(5, 15), pady=4, sticky="w")

        self._label(frame, "Cervello Wiki:", 2, 0, padx=(15, 5))
        self.wiki_brain_var = ctk.StringVar(value=BRAIN_UGUALE)
        self.wiki_brain_menu = ctk.CTkOptionMenu(
            frame, variable=self.wiki_brain_var, values=[BRAIN_UGUALE],
            dynamic_resizing=False, fg_color=COLORE_VIOLA, button_color="#633974",
            button_hover_color="#4f2d5c")
        self.wiki_brain_menu.grid(row=2, column=1, columnspan=2, padx=(5, 15), pady=4, sticky="ew")

        checks = ctk.CTkFrame(frame, fg_color="transparent")
        checks.grid(row=3, column=0, columnspan=3, padx=15, pady=(6, 4), sticky="w")
        self.wiki_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(checks, text="Genera struttura Wiki (raw/ + wiki/)",
                        variable=self.wiki_var, font=ctk.CTkFont(size=12),
                        ).pack(side="left")
        _icona_info(checks, INFO_PARAMETRI["struttura"]).pack(side="left", padx=(4, 16))
        self.wiki_ingest_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(checks, text="🧠 Ingest automatico con LLM",
                        variable=self.wiki_ingest_var, font=ctk.CTkFont(size=12),
                        ).pack(side="left")
        _icona_info(checks, INFO_PARAMETRI["ingest"]).pack(side="left", padx=(4, 0))

        self.btn_batch = ctk.CTkButton(
            frame, text="🚀 Elabora Cartella + Wiki", command=self._elabora_cartella_batch,
            fg_color=COLORE_VIOLA, hover_color="#633974",
            font=ctk.CTkFont(size=13, weight="bold"), height=36)
        self.btn_batch.grid(row=4, column=0, columnspan=3, padx=15, pady=4, sticky="ew")

        r2 = ctk.CTkFrame(frame, fg_color="transparent")
        r2.grid(row=5, column=0, columnspan=3, padx=15, pady=4, sticky="ew")
        self.btn_ingest_wiki = ctk.CTkButton(
            r2, text="🧠 Ingest Wiki esistente…", command=self._ingest_wiki_esistente,
            fg_color=COLORE_TEAL, hover_color="#0f5f66", height=32)
        self.btn_ingest_wiki.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.btn_chiedi_wiki = ctk.CTkButton(
            r2, text="💬 Chiedi alla Wiki…", command=self._chiedi_alla_wiki,
            fg_color=COLORE_TEAL, hover_color="#0f5f66", height=32)
        self.btn_chiedi_wiki.pack(side="left", expand=True, fill="x")

        self._nota_dim(
            frame,
            "Output in '<cartella>/_Wiki_<nome>'. Il cervello legge le trascrizioni in raw/, "
            "scrive una sintesi per fonte, crea pagine concetto/entità collegate e aggiorna "
            "index e log. Anche le risposte alle domande vengono archiviate nella Wiki.",
            row=6, columnspan=3)

    # -------------------------------------------------------------------
    # Log + Progress + Controlli Pausa/Stop
    # -------------------------------------------------------------------
    def _crea_sezione_log(self, parent, row):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=0, padx=20, pady=(5, 10), sticky="nsew")
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        top_row = ctk.CTkFrame(frame, fg_color="transparent")
        top_row.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")
        top_row.grid_columnconfigure(1, weight=1)

        sinistra = ctk.CTkFrame(top_row, fg_color="transparent", height=30)
        sinistra.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            sinistra, text="Log",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")
        self._bottone_grigio(sinistra, "⤒ Torna su", self._vai_in_alto, 90).pack(side="left", padx=(12, 0))
        self._bottone_grigio(sinistra, "🧹 Pulisci", lambda: self.log_text.delete("1.0", "end"),
                             90).pack(side="left", padx=(6, 0))

        # height esplicita: un CTkFrame vuoto (Pausa/Stop nascosti) sarebbe alto 200 px
        self.ctrl_frame = ctk.CTkFrame(top_row, fg_color="transparent", height=30)
        self.ctrl_frame.grid(row=0, column=1, sticky="e")

        self.btn_pausa = ctk.CTkButton(
            self.ctrl_frame, text="⏸ Pausa", width=90,
            command=self._toggle_pausa,
            fg_color=COLORE_GIALLO, hover_color="#9a7209",
            font=ctk.CTkFont(size=12, weight="bold"), height=30,
        )
        self.btn_stop = ctk.CTkButton(
            self.ctrl_frame, text="⏹ Stop", width=90,
            command=self._richiedi_stop,
            fg_color=COLORE_ROSSO, hover_color="#a93226",
            font=ctk.CTkFont(size=12, weight="bold"), height=30,
        )

        self.progress = ctk.CTkProgressBar(frame, height=6)
        self.progress.grid(row=1, column=0, padx=15, pady=(0, 5), sticky="ew")
        self.progress.set(0)

        self.log_text = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(family=FONT_LOG, size=12),
            activate_scrollbars=True, wrap="word",
            fg_color=COLORE_SFONDO_LOG, text_color=COLORE_TESTO_LOG,
            corner_radius=8, height=420,
        )
        self.log_text.grid(row=2, column=0, padx=15, pady=(0, 12), sticky="nsew")
        for evento in (("<Button-4>", "<Button-5>") if sys.platform.startswith("linux")
                       else ("<MouseWheel>",)):
            self.log_text._textbox.bind(evento, self._rotella_log, add="+")
        try:  # il tk.Text interno chiede 24 righe: lasciamo che sia la griglia a espanderlo
            self.log_text._textbox.configure(height=4)
        except Exception:
            pass

    def _crea_status_bar(self):
        self.status_label = ctk.CTkLabel(
            self, text="  Pronto",
            font=ctk.CTkFont(size=11),
            text_color=COLORE_TESTO_DIM, anchor="w",
        )
        self.status_label.grid(row=2, column=0, padx=20, pady=(2, 6), sticky="ew")

    # ===================================================================
    # Impostazioni persistenti
    # ===================================================================
    def _applica_impostazioni(self):
        imp = self._imp

        def _set(var, chiave, valori):
            v = imp.get(chiave)
            if v is not None and v in valori:
                var.set(v)

        _set(self.modello_var, "whisper_modello", MODELLI_WHISPER)
        _set(self.lingua_var, "whisper_lingua", LINGUE_WHISPER)
        _set(self.formato_var, "whisper_formato", ["txt", "dettagliato"])
        _set(self.device_var, "whisper_device", DEVICE_SCELTE)
        _set(self.whisper_thread_var, "whisper_thread", THREAD_WHISPER_SCELTE)
        _set(self.llm_thread_var, "llm_thread", THREAD_LLM_SCELTE)
        _set(self.llm_accel_var, "llm_accelerazione", list(ACCEL_SCELTE))
        _set(self.llm_tipo_var, "llm_tipo", list(llm.TIPI_NOTA))
        _set(self.llm_lingua_var, "llm_lingua", list(llm.LINGUE_NOTA))
        _set(self.llm_ctx_var, "llm_ctx", list(CTX_SCELTE))
        _set(self.llm_maxout_var, "llm_max_output", MAX_OUTPUT_SCELTE)

        for entry, chiave in ((self.size_entry, "mb_segmento"),
                              (self.llm_temp_entry, "llm_temperatura")):
            if imp.get(chiave):
                entry.delete(0, "end")
                entry.insert(0, str(imp[chiave]))
        for var, chiave in ((self.nota_var, "nota"), (self.wiki_var, "wiki_struttura"),
                            (self.wiki_ingest_var, "wiki_ingest")):
            if isinstance(imp.get(chiave), bool):
                var.set(imp[chiave])

        if imp.get("llm_modello"):
            self.llm_modello_var.set(imp["llm_modello"])
            self._llm_label_prec = imp["llm_modello"]
        if imp.get("wiki_cervello"):
            self.wiki_brain_var.set(imp["wiki_cervello"])

        kb = imp.get("kb_path")
        if kb and os.path.isdir(kb):
            self._kb_path = kb
            self._set_entry(self.entry_kb, kb)
        cv = imp.get("cartella_video")
        if cv and os.path.isdir(cv):
            self._cartella_video_batch = cv
            self._set_entry(self.entry_cartella_video, cv)
        self._aggiorna_testo_bottoni()

    def _salva_impostazioni(self):
        dati = {
            "whisper_modello": self.modello_var.get(),
            "whisper_lingua": self.lingua_var.get(),
            "whisper_formato": self.formato_var.get(),
            "whisper_device": self.device_var.get(),
            "whisper_thread": self.whisper_thread_var.get(),
            "llm_thread": self.llm_thread_var.get(),
            "llm_accelerazione": self.llm_accel_var.get(),
            "mb_segmento": self.size_entry.get().strip(),
            "llm_modello": self.llm_modello_var.get(),
            "llm_tipo": self.llm_tipo_var.get(),
            "llm_lingua": self.llm_lingua_var.get(),
            "llm_ctx": self.llm_ctx_var.get(),
            "llm_max_output": self.llm_maxout_var.get(),
            "llm_temperatura": self.llm_temp_entry.get().strip(),
            "llm_custom": self._modelli_custom,
            "nota": bool(self.nota_var.get()),
            "wiki_struttura": bool(self.wiki_var.get()),
            "wiki_ingest": bool(self.wiki_ingest_var.get()),
            "wiki_cervello": self.wiki_brain_var.get(),
            "kb_path": self._kb_path,
            "cartella_video": self._cartella_video_batch,
        }
        try:
            os.makedirs(llm.APP_DIR, exist_ok=True)
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(dati, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _on_close(self):
        if self.in_esecuzione and not messagebox.askyesno(
                "Esci", "Un'elaborazione è in corso. Uscire comunque?\n"
                        "(Il progresso salvato potrà essere ripreso.)"):
            return
        self._salva_impostazioni()
        self.destroy()

    # ===================================================================
    # Modelli LLM: elenco, stato, download
    # ===================================================================
    def _etichette_modelli(self):
        self._llm_specs = {}
        labels = []

        def _aggiungi(spec):
            if spec["label"] not in self._llm_specs:
                self._llm_specs[spec["label"]] = spec
                labels.append(spec["label"])

        for spec in llm.CATALOGO_LLM:
            _aggiungi(spec)
        noti = {llm.normalizza_nome_ollama(s["id"])
                for s in llm.CATALOGO_LLM if s["backend"] == "ollama"}
        locali = set()
        for nome in sorted(self._ollama_installati or []):
            if nome in noti:
                continue
            spec = llm.spec_personalizzata(nome)
            spec["label"] = f"{nome} · Ollama (locale)"
            locali.add(nome)
            _aggiungi(spec)
        for testo in self._modelli_custom:
            try:
                spec = llm.spec_personalizzata(testo)
            except ValueError:
                continue
            if spec["backend"] == "ollama" and llm.normalizza_nome_ollama(spec["id"]) in (noti | locali):
                continue
            _aggiungi(spec)
        labels.append(llm.CUSTOM_LABEL)
        return labels

    def _label_per_spec(self, spec):
        for label, s in self._llm_specs.items():
            if s["backend"] == spec["backend"] and (
                    llm.normalizza_nome_ollama(s["id"]) == llm.normalizza_nome_ollama(spec["id"])
                    if s["backend"] == "ollama"
                    else (s["id"], s.get("pattern")) == (spec["id"], spec.get("pattern"))):
                return label
        return None

    def _aggiorna_menu_modelli(self):
        labels = self._etichette_modelli()
        self.llm_menu.configure(values=labels)
        self.wiki_brain_menu.configure(
            values=[BRAIN_UGUALE] + [l for l in labels if l != llm.CUSTOM_LABEL])
        verifica_in_corso = self._ollama_ok is None
        if self.llm_modello_var.get() not in self._llm_specs and not verifica_in_corso:
            self.llm_modello_var.set(llm.DEFAULT_LLM_LABEL)
            self._llm_label_prec = llm.DEFAULT_LLM_LABEL
        brain = self.wiki_brain_var.get()
        if brain != BRAIN_UGUALE and brain not in self._llm_specs and not verifica_in_corso:
            self.wiki_brain_var.set(BRAIN_UGUALE)
        self._aggiorna_info_llm()

    def _aggiorna_modelli_locali(self):
        def lavoro():
            ok = llm.ollama_raggiungibile()
            installati = None
            if ok:
                try:
                    installati = set(llm.ollama_modelli_installati())
                except Exception:
                    installati = None

            def _fine():
                self._ollama_ok = ok
                self._ollama_installati = installati
                self._aggiorna_menu_modelli()
            self.after(0, _fine)
        threading.Thread(target=lavoro, daemon=True).start()

    def _on_llm_scelto(self, label):
        if label != llm.CUSTOM_LABEL:
            self._llm_label_prec = label
            self._aggiorna_info_llm()
            return
        dlg = ctk.CTkInputDialog(
            title="Modello personalizzato",
            text=("Nome di un modello Ollama, es.\n"
                  "   qwen3:30b-a3b   ·   mistral-small3.2   ·   deepseek-r1:8b\n\n"
                  "oppure un GGUF da HuggingFace (richiede llama-cpp-python):\n"
                  "   hf:<owner>/<repo>[:<pattern>]\n"
                  "   es. hf:bartowski/Qwen_Qwen3-14B-GGUF:*Q4_K_M.gguf"))
        testo = (dlg.get_input() or "").strip()
        if not testo:
            self.llm_modello_var.set(self._llm_label_prec)
            return
        try:
            spec = llm.spec_personalizzata(testo)
        except ValueError as e:
            messagebox.showerror("Errore", str(e))
            self.llm_modello_var.set(self._llm_label_prec)
            return
        if testo not in self._modelli_custom:
            self._modelli_custom.append(testo)
        self._aggiorna_menu_modelli()
        label = self._label_per_spec(spec) or self._llm_label_prec
        self.llm_modello_var.set(label)
        self._llm_label_prec = label
        self._aggiorna_info_llm()
        self._salva_impostazioni()

    def _aggiorna_info_llm(self):
        spec = self._llm_specs.get(self.llm_modello_var.get())
        if not spec:
            self.llm_info.configure(text="… verifica dei modelli locali in corso")
            return
        parti = []
        if spec["backend"] == "ollama":
            parti.append("Ollama")
            if self._ollama_ok is None:
                stato = "… verifica di Ollama in corso"
            elif not self._ollama_ok:
                stato = ("⚠ Ollama non raggiungibile: installalo da ollama.com/download "
                         "(l'app prova ad avviarlo da sola)")
            elif llm.normalizza_nome_ollama(spec["id"]) in (self._ollama_installati or set()):
                stato = "✓ installato"
            else:
                stato = "⬇ da scaricare (automatico al primo uso o con '⬇ Scarica')"
        else:
            parti.append("HuggingFace GGUF · llama.cpp")
            if not llm.librerie_hf_disponibili():
                stato = "⚠ richiede: pip install -r requirements-hf.txt"
            elif llm.hf_percorso_locale(spec):
                stato = "✓ scaricato"
            else:
                stato = "⬇ da scaricare (automatico al primo uso o con '⬇ Scarica')"
        if spec.get("size_gb"):
            parti.append(f"~{spec['size_gb']:g} GB")
        if spec.get("ctx_max"):
            parti.append(f"contesto max {spec['ctx_max'] // 1024}K")
        ctx = CTX_SCELTE.get(self.llm_ctx_var.get(), "auto")
        if ctx == "auto":
            parti.append("contesto adattato al testo (≤32K)")
        else:
            parti.append(f"KV cache ≈ {llm.stima_kv_gb(ctx):.1f} GB a {ctx // 1024}K (stima)")
        if spec.get("note"):
            parti.append(spec["note"])
        if ACCEL_SCELTE.get(self.llm_accel_var.get()) == "cpu":
            parti.append("solo CPU")
        avviso = llm.avviso_ram(spec, ctx if ctx != "auto" else 16384)
        if avviso:
            stato += f"   ⚠ {avviso}"
        self.llm_info.configure(text="  ·  ".join(parti) + "\n" + stato)

    def _scarica_modello_llm(self):
        if self.in_esecuzione:
            return
        spec = self._llm_specs.get(self.llm_modello_var.get())
        if not spec:
            messagebox.showerror("Errore", "Seleziona un modello LLM valido.")
            return
        self._set_in_esecuzione(True)
        self._apri_log_file("download_llm")
        self._log(f"\n{'━'*50}")
        self._log("⬇  DOWNLOAD MODELLO LLM")
        self._log(f"{'━'*50}")
        self._log(f"   {spec['label']}\n")
        self._log_sistema()
        self._set_status(f"Download {spec['label']}...")

        def lavoro():
            try:
                llm.scarica_modello(spec, self._log, self._set_progress, self.stop_event)
                self._set_status("Modello LLM pronto")
            except llm.Interrotto:
                self._log("⏹  Download interrotto.")
                self._set_status("Download interrotto")
            except Exception as e:
                err = str(e)
                self._log(f"✗ ERRORE: {err}")
                self._set_status("Errore download")
                self.after(0, lambda m=err: messagebox.showerror("Errore", m))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)
                self._aggiorna_modelli_locali()

        threading.Thread(target=lavoro, daemon=True).start()

    def _leggi_config_llm(self, per_wiki=False):
        """Legge (nel thread GUI) la configurazione LLM. None se non valida."""
        label = self.llm_modello_var.get()
        if per_wiki and self.wiki_brain_var.get() != BRAIN_UGUALE:
            label = self.wiki_brain_var.get()
        spec = self._llm_specs.get(label)
        if not spec:
            messagebox.showerror("Errore", "Seleziona un modello LLM valido nelle Impostazioni.")
            return None
        try:
            temp = float(self.llm_temp_entry.get().strip().replace(",", "."))
            if not 0 <= temp <= 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Errore", "Temperatura non valida: usa un numero tra 0 e 2 (es. 0.3).")
            return None
        if spec["backend"] == "hf" and not llm.librerie_hf_disponibili():
            messagebox.showerror(
                "Errore", "Per i modelli HuggingFace installa:\n\n"
                          "  pip install -r requirements-hf.txt\n\n"
                          "oppure scegli un modello '· Ollama'.")
            return None
        return {
            "spec": dict(spec),
            "opzioni": {
                "ctx": CTX_SCELTE.get(self.llm_ctx_var.get(), "auto"),
                "max_output": int(self.llm_maxout_var.get()),
                "temperatura": temp,
                "lingua": self.llm_lingua_var.get(),
                "num_thread": self._llm_threads(),
                "gpu": ACCEL_SCELTE.get(self.llm_accel_var.get(), "auto"),
            },
            "tipo": self.llm_tipo_var.get(),
            "kb": self._kb_path,
        }

    def _llm_kwargs(self):
        return dict(callback=self._log, stato_cb=self._set_status,
                    progress_cb=self._set_progress, stop_event=self.stop_event,
                    pause_event=self.pause_event)

    def _whisper_threads(self):
        """Thread CPU per faster-whisper (0 = default della libreria)."""
        v = self.whisper_thread_var.get()
        return int(v) if v.isdigit() else 0

    def _llm_threads(self):
        """Thread CPU per l'LLM (0 = scelta automatica del backend)."""
        v = self.llm_thread_var.get()
        return int(v) if v.isdigit() else 0

    def _log_sistema(self, whisper=False, cfg_llm=None):
        """Intestazione hardware/configurazione: rende confrontabili i test di performance."""
        info = perf.info_sistema()
        self._log(f"🖥  Sistema: {info['os']} ({info['arch']}) · Python {info['python']}")
        self._log(f"   CPU: {info['cpu']} · {perf.descrizione_core(info)}"
                  + (f" · RAM {info['ram_gb']:.0f} GB" if info.get("ram_gb") else ""))
        if info.get("gpu"):
            self._log(f"   GPU: {info['gpu']}")
        if whisper:
            dev = _risolvi_device(self.device_var.get())
            n = self._whisper_threads()
            if dev == "cuda":
                esec = "GPU CUDA (float16)"
            else:
                esec = (f"CPU int8 · {n} thread" if n else
                        f"CPU int8 · auto = {perf.thread_whisper_effettivi(0)} thread")
            self._log(f"   Whisper: {self.modello_var.get()} · device {self.device_var.get()} → {esec}")
        if cfg_llm:
            o = cfg_llm["opzioni"]
            ctx = o["ctx"] if o["ctx"] != "auto" else "auto"
            self._log(f"   LLM: {cfg_llm['spec']['label']} · thread {o['num_thread'] or 'auto'} · "
                      f"{'solo CPU' if o['gpu'] == 'cpu' else 'GPU se disponibile'} · "
                      f"contesto {ctx} · max output {o['max_output']} · temp {o['temperatura']}")
        self._log(f"   Log performance: {perf.PERF_CSV}\n")

    # ===================================================================
    # Fase LLM: trascrizione → nota (+ archiviazione nella knowledge base)
    # ===================================================================
    def _fase_nota(self, percorso_txt, nome_sorgente, cfg):
        """Eseguita nel thread di lavoro. Ritorna (path_nota, path_pagina_wiki)."""
        try:
            with open(percorso_txt, "r", encoding="utf-8") as f:
                contenuto = f.read()
        except Exception as e:
            self._log(f"✗ Impossibile leggere la trascrizione: {e}")
            return None, None
        testo = llm.testo_pulito_trascrizione(contenuto)
        if not testo.strip():
            self._log("⚠ Trascrizione vuota: nessuna nota generata.")
            return None, None

        self._log(f"\n{'━'*50}")
        self._log(f"🧠  NOTA LLM — {cfg['tipo']}")
        self._log(f"{'━'*50}")
        self._set_status("Sintesi LLM in corso...")
        self._set_progress(-1)

        sess = llm.SessioneLLM(cfg["spec"], cfg["opzioni"], **self._llm_kwargs(),
                               fase="llm_nota", sorgente=nome_sorgente)
        try:
            sess.avvia([testo])
            nota = llm.genera_nota(sess, testo, cfg["tipo"], fonte=nome_sorgente)

            cartella = os.path.dirname(os.path.abspath(percorso_txt))
            if os.path.basename(cartella) == "raw":   # non sporcare le fonti immutabili
                cartella = os.path.dirname(cartella)
            base = re.sub(r"_trascrizione$", "",
                          os.path.splitext(os.path.basename(percorso_txt))[0])
            path_nota = os.path.join(cartella, f"{base}_nota.md")
            with open(path_nota, "w", encoding="utf-8") as f:
                f.write(nota["md"])
            self._log(f"\n📝 Nota salvata: {path_nota}")

            path_wiki = None
            kb = cfg.get("kb")
            if kb:
                self._log(f"\n📚 Archiviazione nella knowledge base: {kb}")
                genera_struttura_wiki(kb, kb, callback=None, voce_log=False)
                raw_dir = os.path.join(kb, "raw")
                raw_path = _scrivi_sorgente_wiki(
                    raw_dir, _prossimo_indice_raw(raw_dir), nome_sorgente, contenuto)
                self._log(f"  📄 Sorgente: raw/{os.path.basename(raw_path)}")
                path_wiki = llm.ingest_sorgente(sess, kb, raw_path, cfg["tipo"], nota=nota)
            return path_nota, path_wiki
        except llm.Interrotto:
            self._log("⏹  Sintesi LLM interrotta.")
            return None, None
        except Exception as e:
            err = str(e)
            self._log(f"✗ ERRORE LLM: {err}")
            self._log("   La trascrizione è salvata: puoi rigenerare la nota con "
                      "'📝 Nota da trascrizione…'.")
            self.after(0, lambda m=err: messagebox.showwarning(
                "Sintesi LLM non riuscita", f"{m}\n\nLa trascrizione è comunque salvata."))
            return None, None
        finally:
            sess.chiudi()

    # ===================================================================
    # Utility
    # ===================================================================
    def _apri_log_file(self, etichetta="elaborazione"):
        """Apre un nuovo file di log nella cartella logs/ accanto a main.py."""
        try:
            logs_dir = perf.LOGS_DIR
            os.makedirs(logs_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_file = f"{ts}_{etichetta}.log"
            self._log_file = open(os.path.join(logs_dir, nome_file),
                                  "w", encoding="utf-8", buffering=1)
            intestazione = (
                f"{'='*60}\n"
                f"Log elaborazione — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"{'='*60}\n"
            )
            self._log_file.write(intestazione)
        except Exception as e:
            self._log_file = None
            print(f"[WARN] Impossibile aprire log file: {e}")

    def _chiudi_log_file(self):
        """Chiude il file di log corrente (se aperto)."""
        try:
            if getattr(self, "_log_file", None):
                self._log_file.write(
                    f"\n{'='*60}\n"
                    f"Fine — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
                    f"{'='*60}\n"
                )
                self._log_file.close()
        except Exception:
            pass
        finally:
            self._log_file = None

    def _log(self, msg):
        try:
            if getattr(self, "_log_file", None):
                ts = datetime.now().strftime("%H:%M:%S")
                self._log_file.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

        def _append():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.after(0, _append)

    def _set_status(self, msg):
        self.after(0, lambda: self.status_label.configure(text=f"  {msg}"))

    def _set_progress(self, valore):
        def _update():
            if valore < 0:
                self.progress.configure(mode="indeterminate")
                self.progress.start()
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.progress.set(valore)
        self.after(0, _update)

    def _mostra_controlli(self):
        self.btn_pausa.pack(side="left", padx=(0, 5))
        self.btn_stop.pack(side="left")

    def _nascondi_controlli(self):
        self.btn_pausa.pack_forget()
        self.btn_stop.pack_forget()

    def _toggle_pausa(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.btn_pausa.configure(text="⏸ Pausa", fg_color=COLORE_GIALLO,
                                     hover_color="#9a7209")
            self._set_status("Ripresa in corso...")
        else:
            self.pause_event.set()
            self.btn_pausa.configure(text="▶ Riprendi", fg_color=COLORE_VERDE,
                                     hover_color="#1e6b43")
            self._set_status("In pausa (effettiva al termine del passo corrente)")

    def _richiedi_stop(self):
        if messagebox.askyesno("Conferma Stop",
                "Vuoi interrompere l'esecuzione?\n\n"
                "Il progresso e la trascrizione parziale\n"
                "verranno salvati. Potrai riprendere\n"
                "in un secondo momento."):
            self.stop_event.set()
            self.pause_event.clear()
            self._set_status("Interruzione in corso...")
            self._log("⏹  Stop richiesto, attendo il completamento del passo corrente...")

    def _aggiorna_stato_bottoni(self, occupato):
        s = "disabled" if occupato else "normal"
        wh = s if WHISPER_DISPONIBILE else "disabled"
        for b in (self.btn_elabora, self.btn_nota_txt, self.btn_ingest_wiki,
                  self.btn_chiedi_wiki, self.btn_llm_scarica, self.btn_llm_refresh):
            b.configure(state=s)
        for b in (self.btn_elabora_e_trascrivi, self.btn_trascrivi,
                  self.btn_trascrivi_cartella, self.btn_batch):
            b.configure(state=wh)

    def _set_in_esecuzione(self, stato):
        def _update():
            self.in_esecuzione = stato
            self._aggiorna_stato_bottoni(stato)
            if stato:
                self._set_progress(-1)
                self.stop_event.clear()
                self.pause_event.clear()
                self.btn_pausa.configure(text="⏸ Pausa", fg_color=COLORE_GIALLO,
                                         hover_color="#9a7209")
                self._mostra_controlli()
                self.after(150, self._vai_al_log)
            else:
                self._set_progress(0)
                self._nascondi_controlli()
        # Impostiamo subito il flag (evita doppi avvii prima che `after` giri)
        self.in_esecuzione = stato
        self.after(0, _update)

    def _aggiorna_testo_bottoni(self):
        testo = "⚡ Elabora + Trascrivi + Nota" if self.nota_var.get() else "⚡ Elabora + Trascrivi"
        self.btn_elabora_e_trascrivi.configure(text=testo)

    def _stampa_report_e2e(self):
        tempo_e2e = self.tempo_splitting + self.tempo_trascrizione
        self._log(f"\n{'━'*50}")
        self._log(f"📋  REPORT FINALE END-TO-END")
        self._log(f"{'━'*50}")
        self._log(f"   Video sorgente:       {self.video_size_mb:.1f} MB")
        self._log(f"   Tempo splitting:      {formatta_tempo(self.tempo_splitting)}")
        self._log(f"   Tempo trascrizione:   {formatta_tempo(self.tempo_trascrizione)}")
        self._log(f"   Tempo totale E2E:     {formatta_tempo(tempo_e2e)}")
        if self.video_size_mb > 0 and tempo_e2e > 0:
            sec_per_mb = tempo_e2e / self.video_size_mb
            mb_per_min = (self.video_size_mb / tempo_e2e) * 60
            self._log(f"   Performance:          {sec_per_mb:.2f} sec/MB  ({mb_per_min:.1f} MB/min)")
        self._log(f"{'━'*50}")

    # ===================================================================
    # Selezione file / cartelle
    # ===================================================================
    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleziona file video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                ("All files", "*.*"),
            ]
        )
        if file_path:
            self._set_entry(self.entry_video, file_path)
            self._video_path = file_path

    def _browse_kb(self):
        cartella = filedialog.askdirectory(
            title="Seleziona (o crea) la cartella della knowledge base",
            initialdir=self._kb_path or os.path.expanduser("~"))
        if cartella:
            self._kb_path = cartella
            self._set_entry(self.entry_kb, cartella)
            self._salva_impostazioni()

    def _rimuovi_kb(self):
        self._kb_path = None
        self._set_entry(self.entry_kb, "")
        self._salva_impostazioni()

    def _browse_cartella_video(self):
        cartella = filedialog.askdirectory(title="Seleziona cartella con i video")
        if cartella:
            self._set_entry(self.entry_cartella_video, cartella)
            self._cartella_video_batch = cartella

    def _scegli_cartella_wiki(self, titolo):
        iniziale = self._kb_path or os.path.expanduser("~")
        if self._cartella_video_batch and not self._kb_path:
            nome = os.path.basename(os.path.normpath(self._cartella_video_batch))
            cand = os.path.join(self._cartella_video_batch, f"_Wiki_{nome}")
            iniziale = cand if os.path.isdir(cand) else self._cartella_video_batch
        cartella = filedialog.askdirectory(title=titolo, initialdir=iniziale)
        if not cartella:
            return None
        if any(os.path.isdir(os.path.join(cartella, d)) for d in ("raw", "wiki")):
            return cartella
        sotto = [d for d in os.listdir(cartella)
                 if d.startswith("_Wiki_") and os.path.isdir(os.path.join(cartella, d))]
        if len(sotto) == 1:
            return os.path.join(cartella, sotto[0])
        messagebox.showerror("Errore", "La cartella selezionata non sembra una Wiki "
                                       "(mancano raw/ e wiki/).")
        return None

    # ===================================================================
    # Elaborazione video (singolo)
    # ===================================================================
    def _validate_file_size(self):
        try:
            size_mb = float(self.size_entry.get().strip())
            if size_mb <= 0:
                messagebox.showerror("Errore", "La dimensione deve essere positiva.")
                return None
            if size_mb > 500:
                if not messagebox.askyesno("Conferma",
                        f"{size_mb} MB per segmento. Sicuro?"):
                    return None
            return size_mb
        except ValueError:
            messagebox.showerror("Errore", "Inserisci un numero valido.")
            return None

    def _process_video(self):
        self._avvia_elaborazione_video(auto_trascrivi=False)

    def _process_video_e_trascrivi(self):
        if not WHISPER_DISPONIBILE:
            messagebox.showerror("Errore",
                "faster-whisper non installato.\n"
                "Installa con: pip install faster-whisper")
            return
        self._avvia_elaborazione_video(auto_trascrivi=True)

    def _avvia_elaborazione_video(self, auto_trascrivi=False):
        if self.in_esecuzione:
            return
        if not setup_moviepy():
            messagebox.showerror("Errore",
                "FFmpeg non trovato.\n1. Installa FFmpeg nel PATH\n"
                "2. Oppure metti ffmpeg.exe nella cartella dell'eseguibile")
            return

        video_path = self._video_path
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Errore", "Seleziona un file video valido.")
            return

        target_size_mb = self._validate_file_size()
        if target_size_mb is None:
            return

        cfg_llm = None
        if auto_trascrivi and self.nota_var.get():
            cfg_llm = self._leggi_config_llm()
            if cfg_llm is None:
                return

        self._salva_impostazioni()
        self.video_size_mb = get_file_size_mb(video_path)
        self._set_in_esecuzione(True)

        etichetta = "elaborazione_trascrizione" if auto_trascrivi else "elaborazione_video"
        self._apri_log_file(etichetta)

        modalita = "ELABORAZIONE + TRASCRIZIONE" if auto_trascrivi else "ELABORAZIONE VIDEO"
        if cfg_llm:
            modalita += " + NOTA LLM"
        self._log(f"\n{'━'*50}")
        self._log(f"🎬  {modalita}")
        self._log(f"{'━'*50}")
        self._log(f"   File: {os.path.basename(video_path)}")
        self._log(f"   Dimensione: {self.video_size_mb:.1f} MB")
        if cfg_llm:
            self._log(f"   LLM: {cfg_llm['spec']['label']} · {cfg_llm['tipo']}")
            if cfg_llm["kb"]:
                self._log(f"   Knowledge base: {cfg_llm['kb']}")
        self._log("")
        self._log_sistema(whisper=auto_trascrivi, cfg_llm=cfg_llm)

        lingua = self._get_lingua()
        modello = self.modello_var.get()
        formato = self.formato_var.get()
        device = self.device_var.get()
        cpu_threads = self._whisper_threads()

        def lavoro():
            try:
                self._set_status("Estrazione e segmentazione...")
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = os.path.join(os.path.dirname(video_path), base_name)

                segments, seg_dur_min, t_split = estrai_e_splitta_video(
                    video_path, output_dir, target_size_mb,
                    callback=self._log, stop_event=self.stop_event)
                self.tempo_splitting = t_split

                if self.stop_event.is_set() and not segments:
                    self._set_status("Interrotto")
                    return

                if not segments:
                    self._set_status("Errore splitting")
                    self.after(0, lambda: messagebox.showerror("Errore",
                        "Nessun segmento creato."))
                    return

                self.cartella_segmenti = output_dir
                self._log(f"\n{'─'*50}")
                self._log(f"📊  REPORT SPLITTING")
                self._log(f"{'─'*50}")
                self._log(f"   Segmenti creati:   {len(segments)}")
                self._log(f"   Durata media:      ~{seg_dur_min:.1f} min/segmento")
                self._log(f"   Tempo splitting:   {formatta_tempo(self.tempo_splitting)}")
                self._log(f"   Cartella:          {output_dir}")

                if auto_trascrivi and not self.stop_event.is_set():
                    self._log(f"\n{'━'*50}")
                    self._log(f"🎙  TRASCRIZIONE AUTOMATICA")
                    self._log(f"{'━'*50}")
                    self._log(f"   Cartella: {output_dir}\n")
                    self._set_status("Trascrizione in corso...")

                    percorso_out, tempo_trasc = trascrivi_segmenti(
                        cartella=output_dir, modello_nome=modello,
                        lingua=lingua, device=device, formato=formato,
                        callback=self._log,
                        stop_event=self.stop_event,
                        pause_event=self.pause_event,
                        video_size_mb=self.video_size_mb,
                        tempo_splitting=self.tempo_splitting,
                        cpu_threads=cpu_threads)

                    self.tempo_trascrizione = tempo_trasc

                    if percorso_out:
                        self._stampa_report_e2e()
                        nota = wiki = None
                        if cfg_llm and not self.stop_event.is_set():
                            nota, wiki = self._fase_nota(
                                percorso_out, os.path.basename(video_path), cfg_llm)
                        self._set_status("Completato!" if (nota or not cfg_llm)
                                         else "Trascrizione completata (nota non generata)")
                        self._log("\n🟢 App pronta — puoi consultare i log o avviare un'altra elaborazione.")
                        msg = (f"Processo end-to-end completato!\n\n"
                               f"Segmenti: {len(segments)}\n"
                               f"Trascrizione: {percorso_out}\n"
                               f"Tempo trascrizione: {formatta_tempo(self.tempo_splitting + self.tempo_trascrizione)}")
                        if nota:
                            msg += f"\n\nNota: {nota}"
                        if wiki:
                            msg += f"\nWiki: {wiki}"
                        self.after(0, lambda m=msg: messagebox.showinfo("Completato", m))
                    else:
                        if self.stop_event.is_set():
                            self._set_status("Interrotto — progresso e file parziale salvati")
                        else:
                            self._set_status("Splitting OK, nessun file trascritto")
                elif self.stop_event.is_set():
                    self._set_status("Interrotto dopo splitting")
                    self._log(f"\n✅ Splitting completato. Trascrizione non avviata (interrotto).")
                    self._log(f"   Riprendi con 'Trascrivi segmenti' o 'Da cartella audio…'")
                else:
                    self._set_status("Splitting completato!")
                    self._log(f"\n✅ Puoi ora cliccare 'Trascrivi segmenti'")
                    self.after(0, lambda: messagebox.showinfo("Successo",
                        f"{len(segments)} segmenti creati\n"
                        f"Tempo: {formatta_tempo(self.tempo_splitting)}\n"
                        f"Percorso: {output_dir}"))
            except ImportError as e:
                err = str(e)
                self._set_status("Errore trascrizione")
                self._log(f"\n✗ ERRORE: {err}")
                self.after(0, lambda m=err: messagebox.showerror("Errore", m))
            except Exception as e:
                self._set_status("Errore")
                err = str(e)
                if "'NoneType'" in err:
                    err = "FFmpeg non trovato o non configurato."
                self._log(f"✗ ERRORE: {err}")
                self.after(0, lambda m=err: messagebox.showerror("Errore", m))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()

    # ===================================================================
    # Trascrizione (segmenti / cartella audio)
    # ===================================================================
    def _get_lingua(self):
        sel = self.lingua_var.get()
        if sel.startswith("Auto"):
            return None
        return sel.split(" ")[0]

    def _trascrivi_segmenti(self):
        if self.in_esecuzione:
            return
        if not self.cartella_segmenti or not os.path.isdir(self.cartella_segmenti):
            messagebox.showwarning("Attenzione",
                "Nessuna cartella disponibile.\n"
                "Elabora prima un video o usa 'Da cartella audio…'.")
            return
        self._esegui_trascrizione(self.cartella_segmenti)

    def _trascrivi_da_cartella(self):
        if self.in_esecuzione:
            return
        cartella = filedialog.askdirectory(title="Seleziona cartella con file audio")
        if cartella:
            self.cartella_segmenti = cartella
            self._esegui_trascrizione(cartella)

    def _esegui_trascrizione(self, cartella):
        cfg_llm = None
        if self.nota_var.get():
            cfg_llm = self._leggi_config_llm()
            if cfg_llm is None:
                return
        self._salva_impostazioni()
        self._set_in_esecuzione(True)
        if not getattr(self, "_log_file", None):
            self._apri_log_file("trascrizione")
        self._log(f"\n{'━'*50}")
        self._log(f"🎙  TRASCRIZIONE" + (" + NOTA LLM" if cfg_llm else ""))
        self._log(f"{'━'*50}")
        self._log(f"   Cartella: {cartella}\n")
        self._log_sistema(whisper=True, cfg_llm=cfg_llm)
        self._set_status("Trascrizione in corso...")

        lingua = self._get_lingua()
        modello = self.modello_var.get()
        formato = self.formato_var.get()
        device = self.device_var.get()
        cpu_threads = self._whisper_threads()
        nome_sorgente = os.path.basename(os.path.normpath(cartella))

        def lavoro():
            try:
                percorso_out, tempo_trasc = trascrivi_segmenti(
                    cartella=cartella, modello_nome=modello,
                    lingua=lingua, device=device, formato=formato,
                    callback=self._log,
                    stop_event=self.stop_event,
                    pause_event=self.pause_event,
                    cpu_threads=cpu_threads)

                self.tempo_trascrizione = tempo_trasc

                if percorso_out:
                    if self.tempo_splitting > 0 and self.video_size_mb > 0:
                        self._stampa_report_e2e()
                    nota = wiki = None
                    if cfg_llm and not self.stop_event.is_set():
                        nota, wiki = self._fase_nota(percorso_out, nome_sorgente, cfg_llm)
                    self._set_status("Completato!" if (nota or not cfg_llm)
                                     else "Trascrizione completata (nota non generata)")
                    self._log("\n🟢 App pronta — puoi consultare i log o avviare un'altra elaborazione.")
                    msg = f"Trascrizione salvata in:\n{percorso_out}"
                    if nota:
                        msg += f"\n\nNota:\n{nota}"
                    if wiki:
                        msg += f"\n\nWiki:\n{wiki}"
                    self.after(0, lambda m=msg: messagebox.showinfo("Completato", m))
                else:
                    if self.stop_event.is_set():
                        self._set_status("Interrotto — progresso e file parziale salvati")
                    else:
                        self._set_status("Nessun file trascritto")
            except Exception as e:
                err = str(e)
                self._set_status("Errore trascrizione")
                self._log(f"\n✗ ERRORE: {err}")
                self.after(0, lambda m=err: messagebox.showerror("Errore", m))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()

    def _nota_da_file(self):
        if self.in_esecuzione:
            return
        path = filedialog.askopenfilename(
            title="Seleziona una trascrizione",
            filetypes=[("Trascrizioni", "*.txt *.md"), ("Tutti i file", "*.*")])
        if not path:
            return
        cfg = self._leggi_config_llm()
        if cfg is None:
            return
        self._salva_impostazioni()
        self._set_in_esecuzione(True)
        self._apri_log_file("nota_llm")
        self._log(f"\n{'━'*50}")
        self._log("📝  NOTA DA TRASCRIZIONE")
        self._log(f"{'━'*50}")
        self._log(f"   File: {path}")
        self._log_sistema(cfg_llm=cfg)

        def lavoro():
            try:
                nota, wiki = self._fase_nota(path, os.path.basename(path), cfg)
                if nota:
                    self._set_status("Nota generata!")
                    msg = f"Nota salvata in:\n{nota}"
                    if wiki:
                        msg += f"\n\nArchiviata nella Wiki:\n{wiki}"
                    self.after(0, lambda m=msg: messagebox.showinfo("Completato", m))
                else:
                    self._set_status("Interrotto" if self.stop_event.is_set()
                                     else "Nota non generata")
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()

    # ===================================================================
    # Batch cartella → LLM Wiki
    # ===================================================================
    def _elabora_cartella_batch(self):
        if self.in_esecuzione:
            return
        if not WHISPER_DISPONIBILE:
            messagebox.showerror("Errore",
                "faster-whisper non installato.\n"
                "Installa con: pip install faster-whisper")
            return
        if not setup_moviepy():
            messagebox.showerror("Errore",
                "FFmpeg non trovato.\n1. Installa FFmpeg nel PATH\n"
                "2. Oppure metti ffmpeg.exe nella cartella dell'eseguibile")
            return

        cartella = self._cartella_video_batch
        if not cartella or not os.path.isdir(cartella):
            messagebox.showerror("Errore", "Seleziona una cartella valida con i video.")
            return

        video = trova_file_video(cartella)
        if not video:
            messagebox.showwarning("Attenzione",
                "Nessun file video trovato nella cartella selezionata.")
            return

        target_size_mb = self._validate_file_size()
        if target_size_mb is None:
            return

        genera_wiki = bool(self.wiki_var.get())
        cfg_llm = None
        if genera_wiki and self.wiki_ingest_var.get():
            cfg_llm = self._leggi_config_llm(per_wiki=True)
            if cfg_llm is None:
                return

        lingua = self._get_lingua()
        modello = self.modello_var.get()
        formato = self.formato_var.get()
        device = self.device_var.get()
        cpu_threads = self._whisper_threads()

        # Reset metriche E2E (non significative in batch multi-video)
        self.tempo_splitting = 0.0
        self.video_size_mb = 0.0

        self._salva_impostazioni()
        self._set_in_esecuzione(True)
        self._apri_log_file("batch_wiki")
        self._log(f"\n{'━'*50}")
        self._log(f"🚀  BATCH CARTELLA → LLM WIKI")
        self._log(f"{'━'*50}")
        self._log(f"   Video da processare: {len(video)}")
        self._log(f"   Genera Wiki:         {'sì' if genera_wiki else 'no'}")
        self._log(f"   Cervello Wiki:       "
                  f"{cfg_llm['spec']['label'] + ' · ' + cfg_llm['tipo'] if cfg_llm else 'nessuno (solo trascrizioni)'}\n")
        self._log_sistema(whisper=True, cfg_llm=cfg_llm)
        self._set_status("Batch in corso...")

        def lavoro():
            try:
                risultato = elabora_cartella_video(
                    cartella_video=cartella, target_size_mb=target_size_mb,
                    modello=modello, lingua=lingua, device=device,
                    formato=formato, genera_wiki=genera_wiki,
                    callback=self._log,
                    stop_event=self.stop_event, pause_event=self.pause_event,
                    cpu_threads=cpu_threads)

                if not risultato:
                    self._set_status("Nessun video elaborato")
                    return

                cp = risultato["cartella_progetto"]
                ingest = None
                if cfg_llm and risultato["wiki_pronta"] and not risultato["interrotto"]:
                    self._log(f"\n{'━'*50}")
                    self._log("🧠  INGEST WIKI CON LLM")
                    self._log(f"{'━'*50}")
                    self._set_status("Ingest Wiki con LLM...")
                    self._set_progress(-1)
                    try:
                        ingest = llm.ingest_cartella_wiki(
                            cp, cfg_llm["spec"], cfg_llm["opzioni"], cfg_llm["tipo"],
                            **self._llm_kwargs())
                    except Exception as e:
                        err = str(e)
                        self._log(f"✗ ERRORE INGEST LLM: {err}")
                        self._log("   Le trascrizioni sono salvate: riprova con '🧠 Ingest Wiki esistente…'.")
                        self.after(0, lambda m=err: messagebox.showwarning("Ingest LLM non riuscito", m))

                self._log(f"\n{'━'*50}")
                self._log(f"📋  REPORT BATCH")
                self._log(f"{'━'*50}")
                self._log(f"   Video totali:        {risultato['n_video']}")
                self._log(f"   Completati (tot):    {risultato['n_completati']}")
                self._log(f"   Nuovi in questo run: {risultato['n_nuovi']}")
                self._log(f"   Tempo batch:         {formatta_tempo(risultato['tempo'])}")
                if ingest:
                    self._log(f"   Sorgenti in wiki:    +{ingest['n_nuovi']} (totale sorgenti {ingest['n_tot']})")
                self._log(f"   Progetto:            {cp}")
                self._log(f"{'━'*50}")

                interrotto = risultato["interrotto"] or bool(ingest and ingest["interrotto"])
                if interrotto:
                    self._set_status("Batch interrotto — progresso salvato")
                    self._log("\n⏹  Interrotto. Riavvia il batch sulla stessa cartella per riprendere.")
                elif risultato["wiki_pronta"]:
                    self._set_status("Batch + Wiki completati!")
                    self._log(f"\n📚 LLM Wiki pronta.")
                    self._log(f"   1) Apri '{os.path.basename(cp)}' come vault in Obsidian")
                    self._log(f"   2) Usa '💬 Chiedi alla Wiki…' oppure un agente (es. Claude Code)")
                    self._log("\n🟢 App pronta.")
                    msg = (f"Batch completato!\n\n"
                           f"Video completati: {risultato['n_completati']}/{risultato['n_video']}\n")
                    if ingest:
                        msg += f"Sorgenti sintetizzate ora: {ingest['n_nuovi']}\n"
                    msg += f"\nProgetto Wiki:\n{cp}\n\nAprila in Obsidian come vault."
                    self.after(0, lambda m=msg: messagebox.showinfo("Completato", m))
                else:
                    self._set_status("Batch completato (senza Wiki)")
                    self.after(0, lambda: messagebox.showinfo("Completato",
                        f"Batch completato!\n\n"
                        f"Video completati: {risultato['n_completati']}/{risultato['n_video']}\n"
                        f"Trascrizioni nelle sottocartelle di:\n{cp}"))
            except Exception as e:
                err = str(e)
                self._set_status("Errore batch")
                self._log(f"\n✗ ERRORE BATCH: {err}")
                self.after(0, lambda m=err: messagebox.showerror("Errore", m))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()

    # ===================================================================
    # Wiki esistente: ingest e domande
    # ===================================================================
    def _ingest_wiki_esistente(self):
        if self.in_esecuzione:
            return
        cartella = self._scegli_cartella_wiki("Seleziona la cartella della Wiki (contiene raw/)")
        if not cartella:
            return
        cfg = self._leggi_config_llm(per_wiki=True)
        if cfg is None:
            return
        self._salva_impostazioni()
        self._set_in_esecuzione(True)
        self._apri_log_file("ingest_wiki")
        self._log(f"\n{'━'*50}")
        self._log("🧠  INGEST WIKI")
        self._log(f"{'━'*50}")
        self._log(f"   Wiki: {cartella}")
        self._log(f"   Cervello: {cfg['spec']['label']} · {cfg['tipo']}\n")
        self._log_sistema(cfg_llm=cfg)
        self._set_status("Ingest Wiki con LLM...")

        def lavoro():
            try:
                res = llm.ingest_cartella_wiki(cartella, cfg["spec"], cfg["opzioni"],
                                               cfg["tipo"], **self._llm_kwargs())
                if res["interrotto"]:
                    self._set_status("Ingest interrotto — riprendi quando vuoi")
                else:
                    self._set_status("Ingest completato!")
                    msg = (f"Ingest completato.\n\nSorgenti sintetizzate ora: {res['n_nuovi']}\n"
                           f"Sorgenti totali: {res['n_tot']}\n\nWiki: {cartella}")
                    self.after(0, lambda m=msg: messagebox.showinfo("Completato", m))
            except Exception as e:
                err = str(e)
                self._set_status("Errore ingest")
                self._log(f"\n✗ ERRORE: {err}")
                self.after(0, lambda m=err: messagebox.showerror("Errore", m))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()

    def _chiedi_alla_wiki(self):
        if self.in_esecuzione:
            return
        cartella = self._scegli_cartella_wiki("Seleziona la Wiki da interrogare")
        if not cartella:
            return
        cfg = self._leggi_config_llm(per_wiki=True)
        if cfg is None:
            return
        dlg = ctk.CTkInputDialog(
            title="Chiedi alla Wiki",
            text=f"Wiki: {os.path.basename(os.path.normpath(cartella))}\n\nScrivi la tua domanda:")
        domanda = (dlg.get_input() or "").strip()
        if not domanda:
            return
        self._salva_impostazioni()
        self._set_in_esecuzione(True)
        self._apri_log_file("domanda_wiki")
        self._log(f"\n{'━'*50}")
        self._log("💬  DOMANDA ALLA WIKI")
        self._log(f"{'━'*50}")
        self._log(f"   {domanda}\n")
        self._log_sistema(cfg_llm=cfg)
        self._set_status("Il cervello sta leggendo la Wiki...")

        def lavoro():
            try:
                risposta, path, _ = llm.chiedi_alla_wiki(
                    cartella, domanda, cfg["spec"], cfg["opzioni"], **self._llm_kwargs())
                self._log(f"\n{'─'*50}")
                self._log("💬 RISPOSTA")
                self._log(f"{'─'*50}")
                self._log(risposta)
                self._log(f"\n💾 Salvata nella Wiki: {path}")
                self._set_status("Risposta pronta (salvata nella Wiki)")
                self.after(0, lambda m=path: messagebox.showinfo(
                    "Risposta pronta", f"La risposta è nel log ed è stata archiviata in:\n{m}"))
            except llm.Interrotto:
                self._log("⏹  Domanda interrotta.")
                self._set_status("Interrotto")
            except Exception as e:
                err = str(e)
                self._set_status("Errore")
                self._log(f"\n✗ ERRORE: {err}")
                self.after(0, lambda m=err: messagebox.showerror("Errore", m))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
