"""
perf.py — Informazioni di sistema (CPU / RAM / GPU) e registro delle performance.

Cross-platform (macOS, Windows, Linux) e senza dipendenze esterne: usa sysctl
su macOS, registro + API Win32 su Windows, /proc su Linux (psutil solo se c'è).

Ogni fase misurabile (estrazione audio, trascrizione, sintesi LLM) aggiunge una
riga a logs/performance.csv, così è possibile confrontare test con modelli,
thread e accelerazioni diverse.
"""

import os
import re
import sys
import csv
import ctypes
import platform
import subprocess
import threading
from datetime import datetime

# Con PyInstaller (__file__ punta a una cartella temporanea) i log vanno accanto all'eseguibile
BASE_DIR = (os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PERF_CSV = os.path.join(LOGS_DIR, "performance.csv")

# Su Windows evita che ogni subprocess apra una finestra di console
SUBPROCESS_KW = ({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
                 if os.name == "nt" else {})

# Default di faster-whisper / CTranslate2 quando cpu_threads=0
WHISPER_THREAD_DEFAULT = 4

COLONNE_CSV = [
    "timestamp", "fase", "sorgente", "modello", "backend", "device", "compute",
    "thread", "gpu_offload", "num_ctx", "audio_min", "input_mb", "tempo_s",
    "x_realtime", "token_in", "token_out", "tok_s_prompt", "tok_s_gen",
    "chiamate", "os", "cpu", "core", "ram_gb", "note",
]

_INFO = None
_LOCK = threading.Lock()


def _run(cmd, timeout=4):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           **SUBPROCESS_KW)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _nome_os():
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if os.name == "nt":
        ver = platform.version()            # es. 10.0.22631
        build = int(ver.split(".")[-1]) if ver.split(".")[-1].isdigit() else 0
        rel = "11" if build >= 22000 else platform.release()
        return f"Windows {rel} (build {build})" if build else f"Windows {rel}"
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME", "Linux")
    except Exception:
        return f"Linux {platform.release()}"


def _info_mac(info):
    def sysctl(k):
        return _run(["sysctl", "-n", k])
    info["cpu"] = sysctl("machdep.cpu.brand_string")
    fis, log = sysctl("hw.physicalcpu"), sysctl("hw.logicalcpu")
    info["core_fisici"] = int(fis) if fis.isdigit() else None
    info["core_logici"] = int(log) if log.isdigit() else info["core_logici"]
    livelli = []
    n = sysctl("hw.nperflevels")
    for i in range(int(n) if n.isdigit() else 0):
        c = sysctl(f"hw.perflevel{i}.physicalcpu")
        nome = sysctl(f"hw.perflevel{i}.name") or ("P" if i == 0 else "E")
        if c.isdigit():
            livelli.append(f"{c} {nome}")
    info["core_dettaglio"] = " + ".join(livelli)
    mem = sysctl("hw.memsize")
    if mem.isdigit():
        info["ram_gb"] = int(mem) / 1024 ** 3
    if platform.machine() == "arm64":
        info["gpu"] = "Apple Silicon (Metal, memoria unificata)"


def _core_fisici_windows():
    from ctypes import wintypes

    class SLPI(ctypes.Structure):
        _fields_ = [("ProcessorMask", ctypes.c_size_t),
                    ("Relationship", ctypes.c_int),
                    ("Reserved", ctypes.c_ulonglong * 2)]

    k32 = ctypes.windll.kernel32
    size = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
    n = size.value // ctypes.sizeof(SLPI)
    if n <= 0:
        return None
    buf = (SLPI * n)()
    if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
        return None
    return sum(1 for x in buf if x.Relationship == 0) or None   # RelationProcessorCore


def _info_windows(info):
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            info["cpu"] = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    try:
        info["core_fisici"] = _core_fisici_windows()
    except Exception:
        pass
    try:
        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMSTAT()
        st.dwLength = ctypes.sizeof(MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            info["ram_gb"] = st.ullTotalPhys / 1024 ** 3
    except Exception:
        pass


def _info_linux(info):
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        m = re.search(r"^model name\s*:\s*(.+)$", txt, flags=re.M)
        if m:
            info["cpu"] = m.group(1).strip()
        coppie = set(re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", txt, flags=re.S))
        if coppie:
            info["core_fisici"] = len(coppie)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            m = re.search(r"MemTotal:\s*(\d+) kB", f.read())
        if m:
            info["ram_gb"] = int(m.group(1)) / 1024 ** 2
    except Exception:
        pass


def _gpu_nvidia():
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits"], timeout=4)
    if not out:
        return ""
    gpus = []
    for riga in out.splitlines():
        parti = [p.strip() for p in riga.split(",")]
        if len(parti) >= 2 and parti[1].replace(".", "").isdigit():
            gpus.append(f"{parti[0]} ({float(parti[1]) / 1024:.0f} GB)")
        elif parti and parti[0]:
            gpus.append(parti[0])
    return ", ".join(gpus)


def info_sistema():
    """Dizionario con os, arch, python, cpu, core, ram_gb, gpu (in cache)."""
    global _INFO
    with _LOCK:
        if _INFO is not None:
            return _INFO
        info = {"os": _nome_os(), "arch": platform.machine(),
                "python": platform.python_version(), "cpu": "",
                "core_fisici": None, "core_logici": os.cpu_count() or 1,
                "core_dettaglio": "", "ram_gb": None, "gpu": ""}
        try:
            if sys.platform == "darwin":
                _info_mac(info)
            elif os.name == "nt":
                _info_windows(info)
            else:
                _info_linux(info)
        except Exception:
            pass
        try:
            import psutil
            info["core_fisici"] = info["core_fisici"] or psutil.cpu_count(logical=False)
            info["ram_gb"] = info["ram_gb"] or psutil.virtual_memory().total / 1024 ** 3
        except Exception:
            pass
        if sys.platform != "darwin":
            info["gpu"] = _gpu_nvidia() or "nessuna GPU NVIDIA rilevata"
        info["cpu"] = info["cpu"] or platform.processor() or "CPU sconosciuta"
        _INFO = info
        return info


def descrizione_core(info=None):
    info = info or info_sistema()
    fis, log = info.get("core_fisici"), info.get("core_logici")
    if fis and log and fis != log:
        base = f"{fis} core / {log} thread"
    else:
        base = f"{fis or log} core"
    if info.get("core_dettaglio"):
        base += f" ({info['core_dettaglio']})"
    return base


def descrizione_breve(info=None):
    info = info or info_sistema()
    parti = [info["cpu"], descrizione_core(info)]
    if info.get("ram_gb"):
        parti.append(f"{info['ram_gb']:.0f} GB RAM")
    return " · ".join(parti)


def ram_gb():
    return info_sistema().get("ram_gb")


def thread_whisper_effettivi(cpu_threads, device="cpu"):
    """Thread CPU realmente usati da faster-whisper (None se gira su GPU)."""
    if device == "cuda":
        return None
    if cpu_threads and int(cpu_threads) > 0:
        return int(cpu_threads)
    env = os.environ.get("OMP_NUM_THREADS", "")
    return int(env) if env.isdigit() and int(env) > 0 else WHISPER_THREAD_DEFAULT


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    return "" if v is None else str(v)


def registra_performance(riga):
    """Aggiunge una riga a logs/performance.csv. Ritorna il percorso (o None)."""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        info = info_sistema()
        dati = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                "os": info["os"], "cpu": info["cpu"], "core": descrizione_core(info),
                "ram_gb": round(info["ram_gb"], 1) if info.get("ram_gb") else ""}
        dati.update({k: v for k, v in riga.items() if v is not None})
        nuovo = not os.path.exists(PERF_CSV)
        with _LOCK:
            with open(PERF_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLONNE_CSV, extrasaction="ignore")
                if nuovo:
                    w.writeheader()
                w.writerow({k: _fmt(dati.get(k)) for k in COLONNE_CSV})
        return PERF_CSV
    except Exception:
        return None
