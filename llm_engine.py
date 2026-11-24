"""
llm_engine.py — Sintesi delle trascrizioni e manutenzione della Wiki con LLM
open source eseguiti in locale.

Backend supportati
------------------
• Ollama (consigliato): nessuna dipendenza Python extra. L'app parla con il
  server locale (default http://127.0.0.1:11434, variabile OLLAMA_HOST) e
  scarica i modelli via API (`/api/pull`) al primo utilizzo.
  Installazione: https://ollama.com/download
• HuggingFace GGUF via llama.cpp (opzionale):
      pip install -r requirements-hf.txt   (llama-cpp-python + huggingface_hub)
  I file GGUF vengono scaricati nella cache di HuggingFace (~/.cache/huggingface).

Finestra di contesto
--------------------
Ogni modello ha un contesto massimo (token che può "vedere" in un colpo:
prompt + testo + risposta). Ollama di default usa un contesto piccolo e TRONCA
in silenzio i testi lunghi: qui il contesto (`num_ctx`) viene sempre impostato
in modo esplicito.

  • Il testo viene misurato in token (esatto con llama.cpp, stimato con Ollama).
  • Se testo + istruzioni + output stanno nel contesto → un solo passaggio.
  • Altrimenti map-reduce: la trascrizione viene divisa in blocchi con overlap,
    da ogni blocco si estraggono appunti (map), gli appunti vengono fusi nella
    nota finale (reduce, ricorsivo se ancora troppo lunghi).
  • Il contesto resta fisso per tutta la sessione: con Ollama cambiarlo
    costringerebbe a ricaricare il modello a ogni chiamata.
"""

import os
import re
import gc
import sys
import json
import time
import fnmatch
import shutil
import subprocess
import threading
import unicodedata
import urllib.request
import urllib.error
from datetime import datetime

import perf


APP_DIR = os.path.join(os.path.expanduser("~"), ".from_video_to_wiki")
HF_CACHE_INDEX = os.path.join(APP_DIR, "hf_models.json")


# ===========================================================================
# Parametri di gestione del contesto
# ===========================================================================
CHARS_PER_TOKEN = 3.2      # stima conservativa (IT/EN, tokenizer moderni ~3.5-4.2)
PROMPT_OVERHEAD = 700      # token riservati a system prompt + istruzioni
MARGINE = 0.10             # 10% di sicurezza sulla finestra
OVERLAP_TOKEN = 120        # sovrapposizione tra blocchi consecutivi (map-reduce)
CTX_OPZIONI = [4096, 8192, 16384, 32768, 65536, 131072]
CTX_AUTO_CAP = 32768       # in "Auto" non si supera questo valore (RAM)
KV_MB_PER_TOKEN = 0.14     # stima KV cache fp16 per modelli 4-8B con GQA


class Interrotto(Exception):
    """Sollevata quando l'utente preme Stop durante una chiamata LLM."""


class ErroreLLM(Exception):
    """Errore di configurazione/esecuzione del backend LLM."""


# ===========================================================================
# Catalogo modelli
# ===========================================================================
CUSTOM_LABEL = "✏️ Personalizzato…"

CATALOGO_LLM = [
    # ------------------------------ Ollama ------------------------------
    dict(label="Qwen3 4B · Ollama", backend="ollama", id="qwen3:4b",
         ctx_max=32768, size_gb=2.6, note="veloce, ok con 8 GB RAM"),
    dict(label="Qwen3 8B · Ollama", backend="ollama", id="qwen3:8b",
         ctx_max=32768, size_gb=5.2, note="consigliato, 16 GB RAM"),
    dict(label="Qwen3 14B · Ollama", backend="ollama", id="qwen3:14b",
         ctx_max=32768, size_gb=9.3, note="alta qualità, 32 GB RAM"),
    dict(label="Qwen2.5 7B Instruct · Ollama", backend="ollama", id="qwen2.5:7b",
         ctx_max=32768, size_gb=4.7, note="ottimo italiano, 16 GB RAM"),
    dict(label="Llama 3.1 8B · Ollama", backend="ollama", id="llama3.1:8b",
         ctx_max=131072, size_gb=4.9, note="contesto lungo, 16 GB RAM"),
    dict(label="Llama 3.2 3B · Ollama", backend="ollama", id="llama3.2:3b",
         ctx_max=131072, size_gb=2.0, note="leggero, 8 GB RAM"),
    dict(label="Gemma 3 4B · Ollama", backend="ollama", id="gemma3:4b",
         ctx_max=131072, size_gb=3.3, note="leggero, contesto lungo"),
    dict(label="Gemma 3 12B · Ollama", backend="ollama", id="gemma3:12b",
         ctx_max=131072, size_gb=8.1, note="alta qualità, 24-32 GB RAM"),
    dict(label="Mistral Nemo 12B · Ollama", backend="ollama", id="mistral-nemo:12b",
         ctx_max=131072, size_gb=7.1, note="multilingue, 24 GB RAM"),
    dict(label="Phi-4 mini 3.8B · Ollama", backend="ollama", id="phi4-mini",
         ctx_max=131072, size_gb=2.5, note="leggero, contesto lungo"),
    dict(label="gpt-oss 20B · Ollama", backend="ollama", id="gpt-oss:20b",
         ctx_max=131072, size_gb=14.0, note="ragionamento, 32 GB RAM / GPU 16 GB"),
    # ---------------------- Ollama: modelli grandi ----------------------
    dict(label="Mistral Small 3.2 24B · Ollama", backend="ollama", id="mistral-small3.2:24b",
         ctx_max=131072, size_gb=15.0, note="grande, 32 GB RAM"),
    dict(label="Gemma 3 27B · Ollama", backend="ollama", id="gemma3:27b",
         ctx_max=131072, size_gb=17.0, note="grande, 32 GB RAM"),
    dict(label="Qwen3 30B-A3B (MoE) · Ollama", backend="ollama", id="qwen3:30b",
         ctx_max=32768, size_gb=19.0, note="qualità ~30B, velocità ~3B (MoE), 32 GB RAM"),
    dict(label="Qwen3 32B · Ollama", backend="ollama", id="qwen3:32b",
         ctx_max=32768, size_gb=20.0, note="grande, 48 GB RAM"),
    dict(label="Qwen2.5 32B Instruct · Ollama", backend="ollama", id="qwen2.5:32b",
         ctx_max=32768, size_gb=20.0, note="grande, ottimo italiano, 48 GB RAM"),
    dict(label="DeepSeek-R1 32B · Ollama", backend="ollama", id="deepseek-r1:32b",
         ctx_max=131072, size_gb=20.0, note="ragionamento, 48 GB RAM"),
    dict(label="Llama 3.3 70B · Ollama", backend="ollama", id="llama3.3:70b",
         ctx_max=131072, size_gb=43.0, note="molto grande, 64 GB RAM / workstation"),
    dict(label="gpt-oss 120B · Ollama", backend="ollama", id="gpt-oss:120b",
         ctx_max=131072, size_gb=65.0, note="molto grande, 96 GB+ RAM / workstation"),
    # ------------------------- HuggingFace GGUF -------------------------
    dict(label="Qwen3 4B Q4_K_M · HuggingFace", backend="hf",
         id="Qwen/Qwen3-4B-GGUF", pattern="*Q4_K_M.gguf",
         ctx_max=32768, size_gb=2.5, note="llama.cpp"),
    dict(label="Qwen3 8B Q4_K_M · HuggingFace", backend="hf",
         id="Qwen/Qwen3-8B-GGUF", pattern="*Q4_K_M.gguf",
         ctx_max=32768, size_gb=5.0, note="llama.cpp"),
    dict(label="Qwen2.5 7B Instruct Q4_K_M · HuggingFace", backend="hf",
         id="bartowski/Qwen2.5-7B-Instruct-GGUF", pattern="*Q4_K_M.gguf",
         ctx_max=32768, size_gb=4.7, note="llama.cpp"),
    dict(label="Llama 3.1 8B Instruct Q4_K_M · HuggingFace", backend="hf",
         id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", pattern="*Q4_K_M.gguf",
         ctx_max=131072, size_gb=4.9, note="llama.cpp"),
    dict(label="Gemma 3 4B it Q4_K_M · HuggingFace", backend="hf",
         id="bartowski/google_gemma-3-4b-it-GGUF", pattern="*Q4_K_M.gguf",
         ctx_max=131072, size_gb=2.5, note="llama.cpp"),
    dict(label="Mistral Small 3.2 24B Q4_K_M · HuggingFace", backend="hf",
         id="bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF",
         pattern="*Q4_K_M.gguf", ctx_max=131072, size_gb=14.3, note="llama.cpp, 32 GB RAM"),
    dict(label="Gemma 3 27B it Q4_K_M · HuggingFace", backend="hf",
         id="bartowski/google_gemma-3-27b-it-GGUF", pattern="*Q4_K_M.gguf",
         ctx_max=131072, size_gb=16.5, note="llama.cpp, 32 GB RAM"),
    dict(label="Qwen3 30B-A3B Q4_K_M · HuggingFace", backend="hf",
         id="Qwen/Qwen3-30B-A3B-GGUF", pattern="*Q4_K_M.gguf",
         ctx_max=32768, size_gb=18.6, note="llama.cpp, MoE, 32 GB RAM"),
    dict(label="Qwen3 32B Q4_K_M · HuggingFace", backend="hf",
         id="Qwen/Qwen3-32B-GGUF", pattern="*Q4_K_M.gguf",
         ctx_max=32768, size_gb=19.8, note="llama.cpp, 48 GB RAM"),
]

DEFAULT_LLM_LABEL = "Qwen3 8B · Ollama"


def normalizza_nome_ollama(nome):
    nome = (nome or "").strip()
    return nome if ":" in nome else f"{nome}:latest"


def spec_personalizzata(testo):
    """Crea una spec da testo libero.

    • 'qwen3:30b-a3b' oppure 'ollama:qwen3:30b-a3b'      → Ollama
    • 'hf:owner/repo'  oppure 'hf:owner/repo:*Q5_K_M.gguf' → HuggingFace GGUF
    """
    t = (testo or "").strip()
    if not t:
        raise ValueError("Nome modello vuoto")
    if t.lower().startswith("hf:"):
        resto = t[3:].strip()
        if ":" in resto:
            repo, pattern = resto.split(":", 1)
        else:
            repo, pattern = resto, "*Q4_K_M.gguf"
        repo, pattern = repo.strip(), (pattern.strip() or "*Q4_K_M.gguf")
        if "/" not in repo:
            raise ValueError("Formato HuggingFace: hf:<owner>/<repo>[:<pattern>]")
        return dict(label=f"{repo.split('/')[-1]} [{pattern}] · HuggingFace",
                    backend="hf", id=repo, pattern=pattern, ctx_max=None,
                    size_gb=None, note="personalizzato", testo=t)
    nome = t[7:].strip() if t.lower().startswith("ollama:") else t
    return dict(label=f"{nome} · Ollama", backend="ollama", id=nome,
                ctx_max=None, size_gb=None, note="personalizzato", testo=t)


def stima_kv_gb(num_ctx):
    return num_ctx * KV_MB_PER_TOKEN / 1024


def stima_ram_gb(spec, num_ctx=8192):
    """RAM indicativa: pesi del modello + KV cache + ~1.5 GB di runtime."""
    if not spec.get("size_gb"):
        return None
    return spec["size_gb"] + stima_kv_gb(num_ctx) + 1.5


def avviso_ram(spec, num_ctx=8192, ram_sistema=None):
    """Stringa di avviso se il modello rischia di non stare in memoria (o None)."""
    ram_sistema = ram_sistema or perf.ram_gb()
    serve = stima_ram_gb(spec, num_ctx)
    if not serve or not ram_sistema:
        return None
    if serve > ram_sistema * 0.9:
        return (f"servono ~{serve:.0f} GB ma il sistema ha {ram_sistema:.0f} GB: rischio swap "
                "e forte rallentamento (scegli un modello più piccolo)")
    if sys.platform == "darwin" and perf.info_sistema().get("arch") == "arm64" \
            and serve > ram_sistema * 0.70:
        return (f"~{serve:.0f} GB su {ram_sistema:.0f} GB: oltre il limite GPU di macOS "
                "(~70% della RAM) una parte del modello girerà su CPU, più lentamente")
    return None


# ===========================================================================
# Utility testo
# ===========================================================================

def stima_token(testo):
    return max(1, int(len(testo or "") / CHARS_PER_TOKEN) + 1)


def slug(testo, max_len=60):
    t = unicodedata.normalize("NFKD", str(testo)).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()
    return t[:max_len].strip("_") or "senza_nome"


def _yaml_str(s):
    # Una stringa JSON è anche una stringa YAML valida (double-quoted)
    return json.dumps(str(s), ensure_ascii=False)


def _yaml_lista(valori):
    return "[" + ", ".join(valori) + "]"


def _oggi():
    return datetime.now().strftime("%Y-%m-%d")


def testo_pulito_trascrizione(contenuto):
    """Estrae il testo 'parlato' da una trascrizione .txt o da una sorgente
    raw/*.md (toglie frontmatter, intestazioni e, nel formato 'dettagliato',
    tiene solo la sezione TESTO COMPLETO)."""
    t = contenuto or ""
    if t.startswith("---"):
        fine = t.find("\n---", 3)
        if fine != -1:
            t = t[fine + 4:]
    idx = t.rfind("TESTO COMPLETO")
    if idx != -1:
        dopo = t[idx:].split("\n", 1)[1] if "\n" in t[idx:] else ""
        dopo = re.sub(r"^\s*=+\s*\n", "", dopo)
        if dopo.strip():
            t = dopo
    t = re.sub(r"^\s*#\s*Trascrizione\s*—.*\n", "", t)
    t = t.strip()
    if t.startswith("_(Nessun parlato"):
        return ""
    return t


def _leggi_frontmatter(contenuto):
    dati = {}
    if contenuto.startswith("---"):
        fine = contenuto.find("\n---", 3)
        if fine != -1:
            for riga in contenuto[3:fine].splitlines():
                if ":" in riga:
                    k, v = riga.split(":", 1)
                    dati[k.strip()] = v.strip().strip('"')
    return dati


def pulisci_output(testo):
    """Rimuove blocchi <think> e l'eventuale recinto ```markdown```."""
    t = re.sub(r"<think>.*?</think>", "", testo or "", flags=re.S)
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    t = t.strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    return t


def estrai_tags(corpo):
    righe = corpo.rstrip().splitlines()
    tags = []
    for i in range(len(righe) - 1, max(-1, len(righe) - 6), -1):
        m = re.match(r"^\s*[*_`>]*\s*tags?\s*[*_`]*\s*:\s*(.+)$", righe[i], flags=re.I)
        if m:
            tags = [slug(x.strip().lstrip("#"), 30).replace("_", "-")
                    for x in re.split(r"[,;]", m.group(1)) if x.strip()]
            del righe[i]
            break
    tags = [t for t in dict.fromkeys(tags)
            if t and t != "senza-nome" and not re.fullmatch(r"tag-?\d*", t)][:8]
    return "\n".join(righe).rstrip(), tags


def estrai_titolo(corpo):
    for riga in corpo.splitlines():
        m = re.match(r"^#\s+(.+)$", riga.strip())
        if m:
            titolo = re.sub(r"^(titolo|title)\s*:\s*", "", m.group(1), flags=re.I)
            titolo = titolo.strip().strip("<>").strip()
            if titolo:
                return titolo[:150]
    return None


def _frasi(testo):
    parti = re.split(r"(?<=[\.\!\?…])\s+|\n{2,}", testo)
    return [p.strip() for p in parti if p.strip()]


def dividi_in_blocchi(testo, conta, max_token, overlap=OVERLAP_TOKEN):
    """Divide il testo in blocchi da ~max_token, sui confini di frase,
    con una piccola sovrapposizione per non perdere il filo tra blocchi."""
    unita = []
    for f in _frasi(testo):
        n = conta(f)
        if n <= max_token:
            unita.append((f, n))
            continue
        parole = f.split()
        step = max(50, int(len(parole) * max_token / n * 0.9))
        for k in range(0, len(parole), step):
            s = " ".join(parole[k:k + step])
            unita.append((s, conta(s)))

    blocchi, corrente, n_corr = [], [], 0
    for f, n in unita:
        if corrente and n_corr + n > max_token:
            blocchi.append(" ".join(x for x, _ in corrente))
            coda, n_coda = [], 0
            for x, nx in reversed(corrente):
                if n_coda + nx > overlap:
                    break
                coda.insert(0, (x, nx))
                n_coda += nx
            corrente, n_corr = coda, n_coda
        corrente.append((f, n))
        n_corr += n
    if corrente:
        blocchi.append(" ".join(x for x, _ in corrente))
    return blocchi


def pianifica_contesto(n_token, ctx_scelto, ctx_max, max_output):
    """Decide la finestra di contesto della sessione.

    Ritorna (num_ctx, budget_input, max_output, note):
      • num_ctx: contesto da passare al modello
      • budget_input: token di testo sorgente utilizzabili per passaggio
      • max_output: eventualmente ridotto per lasciare spazio all'input
    """
    ctx_max = int(ctx_max or 32768)
    note = []
    if ctx_scelto in (None, "auto", "Auto"):
        necessari = int((n_token + PROMPT_OVERHEAD + max_output) * (1 + MARGINE)) + 64
        cap = min(ctx_max, CTX_AUTO_CAP)
        num_ctx = next((c for c in CTX_OPZIONI if necessari <= c <= cap), cap)
    else:
        num_ctx = int(ctx_scelto)
        if num_ctx > ctx_max:
            note.append(f"contesto ridotto a {ctx_max} token (massimo del modello)")
            num_ctx = ctx_max
    num_ctx = max(num_ctx, 2048)

    def _budget(mo):
        return int(num_ctx * (1 - MARGINE)) - PROMPT_OVERHEAD - mo

    minimo_input = min(max(n_token, 256), num_ctx // 3)
    if _budget(max_output) < minimo_input:
        nuovo = max(512, int(num_ctx * (1 - MARGINE)) - PROMPT_OVERHEAD - minimo_input)
        if nuovo < max_output:
            note.append(f"max output ridotto a {nuovo} token per lasciare spazio al testo")
            max_output = nuovo
    budget = max(256, _budget(max_output))
    return num_ctx, budget, max_output, note


# ===========================================================================
# Backend: Ollama
# ===========================================================================

def _ollama_base():
    host = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    host = host.replace("://0.0.0.0", "://127.0.0.1").rstrip("/")
    if re.match(r"^https?://[^:/]+$", host):
        host += ":11434"
    return host


OLLAMA_URL = _ollama_base()
# Niente proxy per il server locale (le reti aziendali impostano spesso HTTP_PROXY)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_json(percorso, payload=None, timeout=10):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        OLLAMA_URL + percorso, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    with _OPENER.open(req, timeout=timeout) as r:
        corpo = r.read().decode("utf-8", errors="replace")
    return json.loads(corpo) if corpo.strip() else {}


def _http_stream(percorso, payload, timeout=3600):
    req = urllib.request.Request(
        OLLAMA_URL + percorso, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return _OPENER.open(req, timeout=timeout)


def _errore_http(e):
    try:
        corpo = e.read().decode("utf-8", errors="ignore")
        return json.loads(corpo).get("error", corpo)
    except Exception:
        return str(e)


def ollama_raggiungibile(timeout=2):
    try:
        _http_json("/api/version", timeout=timeout)
        return True
    except Exception:
        return False


def ollama_modelli_installati():
    dati = _http_json("/api/tags", timeout=5)
    return [normalizza_nome_ollama(m.get("name") or m.get("model"))
            for m in dati.get("models", [])]


def _trova_eseguibile_ollama():
    candidati = [
        shutil.which("ollama"),
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        "/Applications/Ollama.app/Contents/Resources/ollama",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
    ]
    for c in candidati:
        if c and os.path.isfile(c):
            return c
    return None


def assicura_server_ollama(log=None):
    if ollama_raggiungibile():
        return
    exe = _trova_eseguibile_ollama()
    if exe:
        if log:
            log("⏳ Ollama non attivo: provo ad avviare 'ollama serve'...")
        kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.Popen([exe, "serve"], **kwargs)
        except Exception as e:
            if log:
                log(f"  ⚠ Avvio non riuscito: {e}")
        for _ in range(30):
            time.sleep(0.5)
            if ollama_raggiungibile():
                if log:
                    log("✓ Ollama avviato")
                return
    raise ErroreLLM(
        f"Ollama non raggiungibile su {OLLAMA_URL}.\n"
        "Installalo da https://ollama.com/download e avvialo, poi riprova.\n"
        "(In alternativa usa un modello '· HuggingFace' con llama.cpp.)")


class OllamaBackend:
    tipo = "ollama"

    def __init__(self, spec, opzioni=None):
        self.spec = spec
        self.model = spec["id"]
        self._ctx_max = None
        opz = opzioni or {}
        self.num_thread = int(opz.get("num_thread") or 0)
        self.solo_cpu = opz.get("gpu") == "cpu"

    def _opzioni_runner(self, num_ctx):
        """Opzioni che determinano come Ollama carica il modello (vanno passate
        identiche a ogni chiamata, altrimenti il modello viene ricaricato)."""
        o = {"num_ctx": num_ctx}
        if self.num_thread > 0:
            o["num_thread"] = self.num_thread
        if self.solo_cpu:
            o["num_gpu"] = 0
        return o

    def descrizione_esecuzione(self):
        thr = f"{self.num_thread} thread CPU" if self.num_thread else "thread CPU auto (scelti da Ollama)"
        return f"{thr} · {'solo CPU' if self.solo_cpu else 'GPU se disponibile'}"

    def stato_memoria(self):
        """(GB occupati, % su GPU) del modello caricato, da /api/ps."""
        try:
            for m in _http_json("/api/ps", timeout=5).get("models", []):
                nome = normalizza_nome_ollama(m.get("name") or m.get("model"))
                if nome == normalizza_nome_ollama(self.model):
                    size, vram = m.get("size") or 0, m.get("size_vram") or 0
                    return size / 1e9, (100.0 * vram / size if size else 0.0)
        except Exception:
            pass
        return None

    # --- stato / download ---------------------------------------------
    def installato(self):
        try:
            return normalizza_nome_ollama(self.model) in ollama_modelli_installati()
        except Exception:
            return False

    def prepara(self, log, progress_cb=None, stop_event=None):
        assicura_server_ollama(log)
        if not self.installato():
            log(f"⬇ '{self.model}' non è ancora in Ollama: download automatico...")
            self.scarica(log, progress_cb, stop_event)

    def scarica(self, log, progress_cb=None, stop_event=None):
        assicura_server_ollama(log)
        payload = {"model": self.model, "name": self.model, "stream": True}
        log("   (non chiudere l'app durante il download: Ollama lo interrompe; "
            "riaprendo riprende da dove si era fermato)")
        ultimo_stato, digest = None, None
        stato_dl = {"t_dato": time.time(), "attivo": True, "avvisato": False}
        rif = {"t": 0.0, "pct": -5, "fatto": 0, "t_vel": time.time(), "fatto_vel": 0, "vel": 0.0}

        def _watchdog():
            # Se la rete si blocca lo stream non manda righe: avvisa dopo 60 s
            while stato_dl["attivo"]:
                time.sleep(5)
                fermo = time.time() - stato_dl["t_dato"]
                if stato_dl["attivo"] and fermo > 60 and not stato_dl["avvisato"]:
                    stato_dl["avvisato"] = True
                    log(f"  ⚠ Nessun dato ricevuto da {fermo:.0f} s: rete lenta o bloccata "
                        "(VPN/proxy?). Il download riprende appena la connessione torna.")

        try:
            resp = _http_stream("/api/pull", payload)
        except urllib.error.HTTPError as e:
            raise ErroreLLM(f"Download '{self.model}' fallito: {_errore_http(e)}")
        threading.Thread(target=_watchdog, daemon=True).start()
        try:
            with resp:
                for raw in resp:
                    if stop_event is not None and stop_event.is_set():
                        raise Interrotto()
                    riga = raw.decode("utf-8", errors="replace").strip()
                    if not riga:
                        continue
                    obj = json.loads(riga)
                    if obj.get("error"):
                        raise ErroreLLM(f"Download '{self.model}' fallito: {obj['error']}")
                    stato = obj.get("status", "")
                    tot, fatto = obj.get("total"), obj.get("completed")
                    if tot and fatto is not None:
                        ora = time.time()
                        if obj.get("digest") != digest:
                            digest = obj.get("digest")
                            rif.update(t=0.0, pct=-5, t_vel=ora, fatto_vel=fatto, vel=0.0)
                        if fatto != rif["fatto"]:
                            stato_dl["t_dato"] = ora
                            if stato_dl["avvisato"]:
                                stato_dl["avvisato"] = False
                                log("  ✓ Il download è ripartito")
                        rif["fatto"] = fatto
                        if ora - rif["t_vel"] >= 3:      # velocità su finestre di ~3 s
                            rif["vel"] = (fatto - rif["fatto_vel"]) / (ora - rif["t_vel"])
                            rif["t_vel"], rif["fatto_vel"] = ora, fatto
                        frac = fatto / tot
                        if progress_cb:
                            progress_cb(frac)
                        pct = int(frac * 100)
                        if tot > 50e6 and (pct >= rif["pct"] + 5 or ora - rif["t"] >= 20 or fatto >= tot):
                            rif["t"], rif["pct"] = ora, pct - pct % 5
                            dettaglio = ""
                            if rif["vel"] > 0 and fatto < tot:
                                resta = (tot - fatto) / rif["vel"]
                                dettaglio = (f" · {rif['vel'] / 1e6:.1f} MB/s · restano "
                                             f"~{resta / 60:.0f} min" if resta >= 90 else
                                             f" · {rif['vel'] / 1e6:.1f} MB/s · restano ~{resta:.0f} s")
                            log(f"  ⬇ {pct:3d}%  {fatto / 1e6:.0f} / {tot / 1e6:.0f} MB{dettaglio}")
                    elif stato and stato != ultimo_stato:
                        log(f"  · {stato}")
                    ultimo_stato = stato
        finally:
            stato_dl["attivo"] = False
        log(f"✓ Modello '{self.model}' pronto in Ollama")

    # --- contesto / caricamento -----------------------------------------
    def ctx_max(self):
        if self._ctx_max is None:
            try:
                info = _http_json("/api/show", {"model": self.model, "name": self.model},
                                  timeout=10)
                for k, v in (info.get("model_info") or {}).items():
                    if k.endswith(".context_length"):
                        self._ctx_max = int(v)
                        break
            except Exception:
                pass
        return self._ctx_max

    def carica(self, num_ctx, log):
        log(f"⏳ Caricamento '{self.model}' in memoria (num_ctx={num_ctx})...")
        t0 = time.time()
        try:
            _http_json("/api/generate", {
                "model": self.model, "prompt": "", "stream": False,
                "keep_alive": "15m", "options": self._opzioni_runner(num_ctx)}, timeout=900)
        except urllib.error.HTTPError as e:
            raise ErroreLLM(f"Caricamento '{self.model}' fallito: {_errore_http(e)}")
        log(f"✓ Modello pronto in {time.time() - t0:.1f}s")

    def conta_token(self, testo):
        return stima_token(testo)

    # --- chat ------------------------------------------------------------
    def _supporta_think_off(self):
        m = self.model.lower()
        return m.startswith(("qwen3", "deepseek-r1", "magistral")) or "qwen3" in m

    def chat(self, messages, num_ctx, temperatura, max_tokens, json_mode=False,
             stop_event=None, on_token=None):
        opzioni = self._opzioni_runner(num_ctx)
        opzioni.update({"temperature": temperatura, "num_predict": max_tokens,
                        "top_p": 0.9, "repeat_penalty": 1.05})
        payload = {
            "model": self.model, "messages": messages, "stream": True,
            "keep_alive": "15m", "options": opzioni,
        }
        if json_mode:
            payload["format"] = "json"
        if self._supporta_think_off():
            payload["think"] = False
        try:
            return self._stream_chat(payload, stop_event, on_token)
        except urllib.error.HTTPError as e:
            msg = _errore_http(e)
            if "think" in payload and e.code == 400:
                payload.pop("think")
                return self._stream_chat(payload, stop_event, on_token)
            raise ErroreLLM(f"Ollama: {msg}")

    def _stream_chat(self, payload, stop_event, on_token):
        testo, stats = [], {}
        with _http_stream("/api/chat", payload) as resp:
            for raw in resp:
                if stop_event is not None and stop_event.is_set():
                    stats["interrotto"] = True
                    break
                riga = raw.decode("utf-8", errors="replace").strip()
                if not riga:
                    continue
                obj = json.loads(riga)
                if obj.get("error"):
                    raise ErroreLLM(f"Ollama: {obj['error']}")
                pezzo = (obj.get("message") or {}).get("content") or ""
                if pezzo:
                    testo.append(pezzo)
                    if on_token:
                        on_token(1)
                if obj.get("done"):
                    # Le durate di Ollama sono in nanosecondi
                    stats.update(
                        prompt_tokens=obj.get("prompt_eval_count"),
                        output_tokens=obj.get("eval_count"),
                        t_prompt=(obj.get("prompt_eval_duration") or 0) / 1e9 or None,
                        t_gen=(obj.get("eval_duration") or 0) / 1e9 or None,
                        t_load=(obj.get("load_duration") or 0) / 1e9 or None,
                        troncato=obj.get("done_reason") == "length")
                    break
        return "".join(testo), stats

    def chiudi(self):
        try:
            _http_json("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=10)
        except Exception:
            pass


# ===========================================================================
# Backend: HuggingFace GGUF via llama.cpp
# ===========================================================================
_LIBRERIE_HF = None


def librerie_hf_disponibili():
    global _LIBRERIE_HF
    if _LIBRERIE_HF is None:
        try:
            import llama_cpp  # noqa: F401
            import huggingface_hub  # noqa: F401
            _LIBRERIE_HF = True
        except Exception:
            _LIBRERIE_HF = False
    return _LIBRERIE_HF


def _carica_indice_hf():
    try:
        with open(HF_CACHE_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _salva_indice_hf(dati):
    try:
        os.makedirs(APP_DIR, exist_ok=True)
        with open(HF_CACHE_INDEX, "w", encoding="utf-8") as f:
            json.dump(dati, f, indent=2)
    except Exception:
        pass


def hf_percorso_locale(spec):
    """Percorso del GGUF già scaricato (o None)."""
    if not librerie_hf_disponibili():
        return None
    files = _carica_indice_hf().get(f"{spec['id']}|{spec.get('pattern', '')}")
    if not files:
        return None
    try:
        from huggingface_hub import hf_hub_download
        percorsi = [hf_hub_download(spec["id"], f, local_files_only=True) for f in files]
        return percorsi[0]
    except Exception:
        return None


class LlamaCppBackend:
    tipo = "hf"

    def __init__(self, spec, opzioni=None):
        self.spec = spec
        self.repo_id = spec["id"]
        self.pattern = spec.get("pattern") or "*Q4_K_M.gguf"
        self.model_path = None
        self.llm = None
        self._vocab = None
        self.n_ctx = None
        opz = opzioni or {}
        self.num_thread = int(opz.get("num_thread") or 0)
        self.solo_cpu = opz.get("gpu") == "cpu"

    def descrizione_esecuzione(self):
        default = max((os.cpu_count() or 2) // 2, 1)     # default di llama-cpp-python
        thr = f"{self.num_thread} thread CPU" if self.num_thread else f"thread CPU auto ({default})"
        return f"{thr} · {'solo CPU' if self.solo_cpu else 'GPU/Metal se disponibile'}"

    def stato_memoria(self):
        return None

    def _verifica_librerie(self):
        if not librerie_hf_disponibili():
            raise ErroreLLM(
                "Per i modelli HuggingFace servono llama-cpp-python e huggingface_hub:\n"
                "  pip install -r requirements-hf.txt\n"
                "(oppure scegli un modello '· Ollama').")

    def installato(self):
        return hf_percorso_locale(self.spec) is not None

    def _risolvi_file(self):
        from huggingface_hub import list_repo_files
        try:
            files = list_repo_files(self.repo_id)
        except Exception as e:
            raise ErroreLLM(f"Impossibile leggere il repo HuggingFace '{self.repo_id}': {e}\n"
                            "Se il modello è 'gated' esegui prima: huggingface-cli login")
        gguf = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
        match = [f for f in gguf
                 if fnmatch.fnmatch(f, self.pattern) or fnmatch.fnmatch(os.path.basename(f), self.pattern)]
        if not match:
            disponibili = ", ".join(os.path.basename(f) for f in gguf[:12]) or "nessuno"
            raise ErroreLLM(f"Nessun file '{self.pattern}' in {self.repo_id}.\nDisponibili: {disponibili}")
        split = sorted(f for f in match if re.search(r"-\d{5}-of-\d{5}\.gguf$", f))
        if split:
            return split
        return [sorted(match, key=len)[0]]

    def prepara(self, log, progress_cb=None, stop_event=None):
        self._verifica_librerie()
        self.model_path = hf_percorso_locale(self.spec)
        if not self.model_path:
            self.scarica(log, progress_cb, stop_event)

    def scarica(self, log, progress_cb=None, stop_event=None):
        self._verifica_librerie()
        from huggingface_hub import hf_hub_download
        files = self._risolvi_file()
        if progress_cb:
            progress_cb(-1)
        percorsi = []
        for f in files:
            if stop_event is not None and stop_event.is_set():
                raise Interrotto()
            log(f"⬇ Download da HuggingFace: {self.repo_id}/{f}")
            log("   (può richiedere diversi minuti; il dettaglio è nel terminale)")
            percorsi.append(hf_hub_download(self.repo_id, f))
        indice = _carica_indice_hf()
        indice[f"{self.repo_id}|{self.pattern}"] = files
        _salva_indice_hf(indice)
        self.model_path = percorsi[0]
        log(f"✓ Scaricato: {self.model_path}")

    def _tokenizer(self):
        if self.llm is not None:
            return self.llm
        if self._vocab is None and self.model_path:
            from llama_cpp import Llama
            self._vocab = Llama(model_path=self.model_path, vocab_only=True, verbose=False)
        return self._vocab

    def ctx_max(self):
        try:
            tok = self._tokenizer()
            md = getattr(tok, "metadata", {}) or {}
            arch = md.get("general.architecture")
            v = md.get(f"{arch}.context_length") if arch else None
            return int(v) if v else None
        except Exception:
            return None

    def conta_token(self, testo):
        try:
            tok = self._tokenizer()
            if tok is not None:
                return len(tok.tokenize((testo or "").encode("utf-8"), add_bos=False))
        except Exception:
            pass
        return stima_token(testo)

    def carica(self, num_ctx, log):
        from llama_cpp import Llama
        if self.llm is not None and self.n_ctx == num_ctx:
            return
        self._libera()
        log(f"⏳ Caricamento {os.path.basename(self.model_path)} (n_ctx={num_ctx}, "
            f"{self.descrizione_esecuzione()})...")
        t0 = time.time()
        kwargs = dict(model_path=self.model_path, n_ctx=num_ctx, n_batch=512, verbose=False,
                      n_gpu_layers=0 if self.solo_cpu else -1)
        if self.num_thread > 0:
            kwargs.update(n_threads=self.num_thread, n_threads_batch=self.num_thread)
        self.llm = Llama(**kwargs)
        self.n_ctx = num_ctx
        self._vocab = None
        log(f"✓ Modello pronto in {time.time() - t0:.1f}s")

    def chat(self, messages, num_ctx, temperatura, max_tokens, json_mode=False,
             stop_event=None, on_token=None):
        msgs = [dict(m) for m in messages]
        if "qwen3" in self.repo_id.lower():
            msgs[-1]["content"] += "\n/no_think"
        n_prompt = sum(self.conta_token(m["content"]) for m in msgs) + 16 * len(msgs)
        spazio = self.n_ctx - n_prompt - 16
        if spazio < 128:
            raise ErroreLLM(f"Prompt di {n_prompt} token troppo lungo per il contesto {self.n_ctx}")
        max_tokens = min(max_tokens, spazio)
        kwargs = dict(messages=msgs, temperature=temperatura, max_tokens=max_tokens,
                      top_p=0.9, repeat_penalty=1.05, stream=True)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        testo, stats, n_out = [], {"prompt_tokens": n_prompt}, 0
        t0, t_primo = time.time(), None
        for chunk in self.llm.create_chat_completion(**kwargs):
            if stop_event is not None and stop_event.is_set():
                stats["interrotto"] = True
                break
            scelta = chunk["choices"][0]
            pezzo = (scelta.get("delta") or {}).get("content") or ""
            if pezzo:
                if t_primo is None:
                    t_primo = time.time()      # fine lettura prompt ≈ primo token
                testo.append(pezzo)
                n_out += 1
                if on_token:
                    on_token(1)
            if scelta.get("finish_reason"):
                stats["troncato"] = scelta["finish_reason"] == "length"
        t_fine = time.time()
        stats["output_tokens"] = n_out
        if t_primo:
            stats["t_prompt"] = t_primo - t0
            stats["t_gen"] = t_fine - t_primo
        return "".join(testo), stats

    def _libera(self):
        self.llm = None
        self._vocab = None
        gc.collect()

    def chiudi(self):
        self._libera()


def crea_backend(spec, opzioni=None):
    if spec["backend"] == "hf":
        return LlamaCppBackend(spec, opzioni)
    return OllamaBackend(spec, opzioni)


def scarica_modello(spec, log, progress_cb=None, stop_event=None):
    """Scarica (se serve) il modello. Usato dal pulsante '⬇ Scarica'."""
    backend = crea_backend(spec)
    if spec["backend"] == "ollama":
        assicura_server_ollama(log)
    elif not librerie_hf_disponibili():
        backend._verifica_librerie()
    if backend.installato():
        log(f"✓ '{spec['label']}' è già scaricato.")
        return
    backend.scarica(log, progress_cb, stop_event)


# ===========================================================================
# Sessione: un modello caricato con un contesto fisso
# ===========================================================================

class SessioneLLM:
    """Un modello caricato con contesto, thread e accelerazione fissi.
    Raccoglie le metriche di tutte le chiamate e, alla chiusura, scrive il
    riepilogo nel log e una riga in logs/performance.csv."""

    def __init__(self, spec, opzioni, callback=None, stato_cb=None,
                 progress_cb=None, stop_event=None, pause_event=None,
                 fase="llm", sorgente=None):
        self.spec = spec
        self.opz = dict(opzioni or {})
        self.backend = crea_backend(spec, self.opz)
        self.callback = callback
        self.stato_cb = stato_cb
        self.progress_cb = progress_cb
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.fase = fase
        self.sorgente = sorgente
        self.num_ctx = None
        self.budget_input = None
        self.max_output = int(self.opz.get("max_output", 2048))
        self.temperatura = float(self.opz.get("temperatura", 0.3))
        self.istruzione_lingua = LINGUE_NOTA.get(self.opz.get("lingua"), LINGUE_NOTA["Italiano"])
        self._gpu = ""
        self._t_avvio = None
        self._m = dict(chiamate=0, token_in=0, token_out=0, t_prompt=0.0, t_gen=0.0,
                       t_tot=0.0, tok_prompt_misurati=0, tok_gen_misurati=0, interrotto=False)

    @property
    def nome_modello(self):
        return self.spec["id"]

    def _log(self, msg):
        if self.callback:
            self.callback(msg)

    def _stato(self, msg):
        if self.stato_cb:
            self.stato_cb(msg)

    def fermato(self):
        return bool(self.stop_event and self.stop_event.is_set())

    def attendi_pausa(self):
        if self.pause_event and self.pause_event.is_set():
            self._log("⏸  In pausa... (clicca ▶ Riprendi per continuare)")
            while self.pause_event.is_set():
                if self.fermato():
                    raise Interrotto()
                time.sleep(0.3)
            self._log("▶  Ripresa...")
        if self.fermato():
            raise Interrotto()

    def conta(self, testo):
        return self.backend.conta_token(testo)

    def avvia(self, testi=None):
        """Assicura server/modello (download incluso), pianifica il contesto
        sul testo più lungo previsto e carica il modello."""
        self._log(f"🧠 Modello: {self.spec['label']}")
        self._log(f"⚙  Esecuzione LLM: {self.backend.descrizione_esecuzione()}")
        self.backend.prepara(self._log, self.progress_cb, self.stop_event)
        self._t_avvio = time.time()     # il download eventuale non entra nelle metriche
        if self.progress_cb:
            self.progress_cb(-1)
        ctx_modello = self.backend.ctx_max()
        ctx_max = ctx_modello or self.spec.get("ctx_max") or 32768
        if not ctx_modello and not self.spec.get("ctx_max"):
            self._log("  ⚠ Contesto massimo del modello sconosciuto: assumo 32768")
        n = max((self.conta(t) for t in (testi or [""])), default=0)
        self.num_ctx, self.budget_input, self.max_output, note = pianifica_contesto(
            n, self.opz.get("ctx", "auto"), ctx_max, self.max_output)
        self._log(f"🧮 Contesto: {self.num_ctx} token (max modello {ctx_max}) · "
                  f"testo per passaggio ≤ {self.budget_input} · output ≤ {self.max_output} · "
                  f"KV cache ≈ {stima_kv_gb(self.num_ctx):.1f} GB (stima)")
        for nota in note:
            self._log(f"  ⚠ {nota}")
        avviso = avviso_ram(self.spec, self.num_ctx)
        if avviso:
            self._log(f"  ⚠ RAM: {avviso}")
        self.backend.carica(self.num_ctx, self._log)
        mem = self.backend.stato_memoria()
        if mem:
            gb, pct = mem
            self._gpu = f"{pct:.0f}%"
            if pct >= 99:
                dove = "100% su GPU"
            elif pct <= 1:
                dove = "solo CPU"
            else:
                dove = f"{pct:.0f}% su GPU, il resto su CPU (più lento)"
            self._log(f"🖥  Modello in memoria: {gb:.1f} GB · {dove}")
        else:
            self._gpu = "cpu" if self.opz.get("gpu") == "cpu" else "auto"

    def chat(self, messages, max_tokens=None, json_mode=False, etichetta="LLM"):
        self.attendi_pausa()
        max_tokens = min(int(max_tokens or self.max_output), self.max_output)
        self._stato(f"🧠 {etichetta}…")
        t0 = time.time()
        conteggio = {"n": 0, "t": 0.0}

        def on_token(k):
            conteggio["n"] += k
            ora = time.time()
            if ora - conteggio["t"] > 0.7:
                conteggio["t"] = ora
                dt_live = ora - t0
                self._stato(f"🧠 {etichetta} — {conteggio['n']} token "
                            f"({conteggio['n'] / dt_live:.1f} tok/s)" if dt_live > 0 else "")

        try:
            testo, stats = self.backend.chat(
                messages, self.num_ctx, self.temperatura, max_tokens,
                json_mode=json_mode, stop_event=self.stop_event, on_token=on_token)
        finally:
            self._m["t_tot"] += time.time() - t0
        if stats.get("interrotto"):
            self._m["interrotto"] = True
            raise Interrotto()
        dt = time.time() - t0
        n_out = stats.get("output_tokens") or conteggio["n"]
        n_in = stats.get("prompt_tokens") or 0
        t_p, t_g = stats.get("t_prompt"), stats.get("t_gen")

        self._m["chiamate"] += 1
        self._m["token_in"] += n_in
        self._m["token_out"] += n_out
        if t_p and n_in:
            self._m["t_prompt"] += t_p
            self._m["tok_prompt_misurati"] += n_in
        if t_g and n_out:
            self._m["t_gen"] += t_g
            self._m["tok_gen_misurati"] += n_out

        parti = []
        if n_in:
            parti.append(f"input {n_in} tok" + (f" ({n_in / t_p:.0f} tok/s)" if t_p else ""))
        vel_gen = (n_out / t_g) if t_g else (n_out / dt if dt > 0 else 0)
        parti.append(f"output {n_out} tok ({vel_gen:.1f} tok/s)")
        parti.append(f"{dt:.1f}s")
        if stats.get("t_load") and stats["t_load"] > 1:
            parti.append(f"ricarica modello {stats['t_load']:.1f}s")
        self._log(f"  ✓ {etichetta}: " + " · ".join(parti))
        if n_in and n_in >= self.num_ctx * 0.98:
            self._log("  ⚠ L'input ha riempito il contesto: possibile troncamento. "
                      "Aumenta 'Contesto' o usa un modello con contesto più ampio.")
        if stats.get("troncato"):
            self._log("  ⚠ Risposta troncata dal limite 'Max output': aumentalo se la nota è incompleta.")
        return pulisci_output(testo)

    def riepilogo(self):
        m = self._m
        if not m["chiamate"]:
            return None
        tok_s_p = m["tok_prompt_misurati"] / m["t_prompt"] if m["t_prompt"] else None
        tok_s_g = m["tok_gen_misurati"] / m["t_gen"] if m["t_gen"] else None
        tempo = time.time() - self._t_avvio if self._t_avvio else m["t_tot"]
        return dict(chiamate=m["chiamate"], token_in=m["token_in"], token_out=m["token_out"],
                    tok_s_prompt=tok_s_p, tok_s_gen=tok_s_g, tempo_llm=m["t_tot"], tempo=tempo)

    def chiudi(self):
        r = self.riepilogo()
        if r:
            self._log(f"\n📈 Riepilogo LLM ({self.fase}): {r['chiamate']} chiamate · "
                      f"input {r['token_in']} tok · output {r['token_out']} tok · "
                      f"generazione {formatta_durata(r['tempo_llm'])} su {formatta_durata(r['tempo'])} totali")
            dett = []
            if r["tok_s_prompt"]:
                dett.append(f"lettura prompt {r['tok_s_prompt']:.0f} tok/s")
            if r["tok_s_gen"]:
                dett.append(f"generazione {r['tok_s_gen']:.1f} tok/s")
            dett.append(self.backend.descrizione_esecuzione())
            if self._gpu and self._gpu not in ("auto", "cpu"):
                dett.append(f"GPU {self._gpu}")
            self._log("   " + " · ".join(dett))
            if perf.registra_performance(dict(
                    fase=self.fase, sorgente=self.sorgente, modello=self.nome_modello,
                    backend=self.spec["backend"], device="cpu" if self.opz.get("gpu") == "cpu" else "auto",
                    thread=self.opz.get("num_thread") or "auto", gpu_offload=self._gpu,
                    num_ctx=self.num_ctx, tempo_s=r["tempo"], token_in=r["token_in"],
                    token_out=r["token_out"], tok_s_prompt=r["tok_s_prompt"],
                    tok_s_gen=r["tok_s_gen"], chiamate=r["chiamate"],
                    note="interrotto" if self._m["interrotto"] else "")):
                self._log(f"   📈 Metriche aggiunte a: {perf.PERF_CSV}")
        try:
            self.backend.chiudi()
        except Exception:
            pass


def formatta_durata(secondi):
    secondi = secondi or 0
    if secondi < 60:
        return f"{secondi:.1f}s"
    m, s = divmod(int(secondi), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ===========================================================================
# Prompt e tipi di nota
# ===========================================================================
LINGUE_NOTA = {
    "Italiano": "Scrivi la nota in italiano.",
    "English": "Write the note in English.",
    "Come la registrazione": "Scrivi la nota nella stessa lingua della trascrizione.",
}

SYSTEM_PROMPT = (
    "Sei un assistente esperto nel trasformare trascrizioni audio in note markdown "
    "per una knowledge base personale (vault Obsidian).\n"
    "Le trascrizioni sono automatiche: possono contenere errori di riconoscimento, "
    "ripetizioni, intercalari e frasi spezzate.\n"
    "Regole:\n"
    "- Sii fedele: non inventare fatti, nomi, numeri, date o decisioni assenti dal testo.\n"
    "- Se un passaggio è ambiguo o probabilmente trascritto male, segnalalo con (?).\n"
    "- Correggi gli errori di trascrizione evidenti solo quando il senso è chiaro.\n"
    "- Elimina intercalari, ripetizioni e divagazioni irrilevanti.\n"
    "- Markdown pulito: titoli ##/###, elenchi puntati, tabelle quando utili, "
    "**grassetto** per i termini chiave, $...$ per le formule.\n"
    "- Non racchiudere la risposta in un blocco di codice."
)

TIPI_NOTA = {
    "Auto (decide il modello)": dict(
        categoria="auto",
        descrizione="una registrazione (riunione, lezione, call, talk o altro)",
        focus="argomenti, fatti, definizioni, decisioni, compiti, numeri e nomi citati",
        struttura=(
            "# <Titolo descrittivo>\n"
            "> **Tipo:** <riunione | standup | lezione | talk | intervista | brainstorming | altro>\n"
            "> <Sintesi in 2-3 frasi>\n\n"
            "Poi scegli tu le sezioni ## più adatte al tipo riconosciuto. Esempi:\n"
            "- lezione → Concetti chiave, Spiegazione, Esempi, Glossario, Domande di ripasso\n"
            "- riunione/standup → Punti discussi, Decisioni, Action item (tabella Cosa|Chi|Entro)\n"
            "- talk/intervista → Tesi, Argomenti, Citazioni, Takeaway"),
    ),
    "Standup / Daily": dict(
        categoria="standup",
        descrizione="uno standup/daily meeting di un team",
        focus="cosa ha fatto ogni persona, cosa farà, blocchi, decisioni, action item con responsabile",
        struttura=(
            "# Standup — <team o progetto, se citato>\n"
            "> <Sintesi in 1-2 frasi>\n\n"
            "## Aggiornamenti per persona\n"
            "### <Nome>\n- **Fatto:** ...\n- **Prossimo:** ...\n- **Blocchi:** ...\n\n"
            "## Blocchi e rischi\n"
            "## Decisioni\n"
            "## Action item\n| Cosa | Chi | Entro |\n|---|---|---|"),
    ),
    "Riunione / Call di lavoro": dict(
        categoria="riunione",
        descrizione="una riunione o call di lavoro",
        focus="contesto, temi discussi, posizioni dei partecipanti, decisioni, action item, scadenze, domande aperte",
        struttura=(
            "# <Titolo della riunione>\n"
            "> <Sintesi in 2-3 frasi>\n\n"
            "## Contesto e obiettivo\n"
            "## Partecipanti\n(solo se identificabili dal testo)\n"
            "## Punti discussi\n### <Tema>\n...\n"
            "## Decisioni prese\n"
            "## Action item\n| Cosa | Chi | Entro |\n|---|---|---|\n"
            "## Domande aperte"),
    ),
    "Lezione / Studio": dict(
        categoria="lezione",
        descrizione="una lezione (università, corso o formazione) da studiare",
        focus="concetti, definizioni, teoremi/formule, passaggi logici, esempi, esercizi, riferimenti bibliografici, cose che il docente indica come importanti o da esame",
        struttura=(
            "# <Titolo della lezione>\n"
            "> <Sintesi in 2-3 frasi: di cosa parla e perché è importante>\n\n"
            "## Concetti chiave\n- **<termine>** — spiegazione chiara e completa\n\n"
            "## Spiegazione\n### <Argomento>\n(segui l'ordine logico della lezione; includi "
            "passaggi, formule $...$ ed esempi)\n\n"
            "## Esempi ed esercizi\n"
            "## Glossario\n| Termine | Definizione |\n|---|---|\n"
            "## Domande di ripasso\n(5-10 domande con risposta breve)\n**D:** ...\n**R:** ...\n\n"
            "## Da approfondire\n(SOLO argomenti accennati, testi o capitoli EFFETTIVAMENTE "
            "citati nella lezione e cose segnalate come importanti per l'esame; se non ce ne "
            "sono ometti la sezione)"),
    ),
    "Webinar / Talk / Presentazione": dict(
        categoria="talk",
        descrizione="un webinar, talk o presentazione",
        focus="tesi principale, argomenti, dati, esempi, citazioni, raccomandazioni, risorse citate",
        struttura=(
            "# <Titolo del talk>\n"
            "> <Tesi principale in 2-3 frasi>\n\n"
            "## Relatore e contesto\n(solo se citati)\n"
            "## Argomenti principali\n### <Argomento>\n...\n"
            "## Dati, esempi e citazioni\n"
            "## Key takeaway\n"
            "## Risorse citate"),
    ),
    "Intervista": dict(
        categoria="intervista",
        descrizione="un'intervista",
        focus="profilo dell'intervistato, domande, risposte, opinioni, aneddoti, citazioni testuali",
        struttura=(
            "# Intervista — <persona o tema>\n"
            "> <Sintesi in 2-3 frasi>\n\n"
            "## Profilo dell'intervistato\n"
            "## Temi trattati\n### <Tema>\n...\n"
            "## Citazioni salienti\n> \"...\"\n\n"
            "## Insight e spunti"),
    ),
    "Brainstorming / Idee": dict(
        categoria="brainstorming",
        descrizione="una sessione di brainstorming",
        focus="obiettivo, tutte le idee emerse, pro e contro, valutazioni, idee scelte, prossimi passi",
        struttura=(
            "# Brainstorming — <tema>\n"
            "> <Sintesi in 2-3 frasi>\n\n"
            "## Obiettivo\n"
            "## Idee emerse\n### <Gruppo tematico>\n...\n"
            "## Valutazione\n| Idea | Pro | Contro |\n|---|---|---|\n"
            "## Idee scelte e prossimi passi\n"
            "## Parcheggio\n(idee da riprendere)"),
    ),
    "Riassunto generico": dict(
        categoria="generico",
        descrizione="una registrazione generica",
        focus="argomenti, fatti, nomi, numeri, conclusioni",
        struttura=(
            "# <Titolo>\n"
            "> <Sintesi in 3-4 frasi>\n\n"
            "## Punti principali\n"
            "## Dettagli\n### <Argomento>\n...\n"
            "## Conclusioni\n"
            "## Da ricordare"),
    ),
}

DEFAULT_TIPO = "Auto (decide il modello)"


def _prompt_map(cfg, blocco, i, n, lingua):
    return (
        f"Stai leggendo la PARTE {i} di {n} della trascrizione automatica di {cfg['descrizione']}.\n\n"
        f"<trascrizione parte=\"{i}/{n}\">\n{blocco}\n</trascrizione>\n\n"
        "COMPITO: estrai appunti DETTAGLIATI e FEDELI da questa parte; serviranno a "
        "scrivere una nota completa insieme agli appunti delle altre parti.\n"
        f"Concentrati su: {cfg['focus']}.\n"
        "Regole: solo elenchi puntati raggruppati per argomento con un titoletto ###; "
        "conserva nomi, numeri, date, definizioni, esempi, decisioni e compiti con "
        "il responsabile; niente introduzioni né conclusioni; non inventare.\n"
        f"{lingua}"
    )


def _prompt_fondi(cfg, appunti, lingua):
    return (
        f"Qui sotto trovi appunti consecutivi estratti dalla trascrizione di {cfg['descrizione']}.\n\n"
        f"<appunti>\n{appunti}\n</appunti>\n\n"
        "COMPITO: fondili in un unico elenco di appunti, ordinato per argomento, "
        "eliminando i duplicati ma SENZA perdere dettagli (nomi, numeri, definizioni, "
        "decisioni, compiti). Solo elenchi puntati con titoletti ###.\n"
        f"{lingua}"
    )


def _prompt_finale(cfg, materiale, da_appunti, n_parti, lingua):
    if da_appunti:
        intro = (f"Qui sotto trovi gli appunti estratti, in ordine, dalle {n_parti} parti "
                 f"della trascrizione di {cfg['descrizione']}.")
        tag = "appunti"
    else:
        intro = f"Qui sotto trovi la trascrizione automatica completa di {cfg['descrizione']}."
        tag = "trascrizione"
    return (
        f"{intro}\n\n<{tag}>\n{materiale}\n</{tag}>\n\n"
        "COMPITO: trasforma il materiale in una nota markdown completa e ben organizzata "
        "seguendo questa struttura:\n\n"
        f"{cfg['struttura']}\n\n"
        "Regole di formato:\n"
        "- Il testo tra < > e tra parentesi nella struttura è un'ISTRUZIONE: sostituiscilo "
        "con il contenuto reale, non copiarlo mai nella nota.\n"
        "- Ometti le sezioni per cui il materiale non contiene informazioni.\n"
        "- Usa solo informazioni presenti nel materiale: niente libri, autori, dati o "
        "esempi inventati.\n"
        "- La nota deve essere utile a chi NON ha ascoltato la registrazione: completa "
        "ma senza ripetizioni.\n"
        f"{lingua}\n"
        "L'ultima riga della risposta deve essere 'Tags:' seguita da 3-6 parole chiave "
        "reali dell'argomento, minuscole e separate da virgola (esempio per una lezione "
        "di chimica: Tags: chimica, legami covalenti, elettronegatività)."
    )


def _messaggi(user):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def _comprimi_appunti(sess, cfg, appunti):
    """Riduce ricorsivamente gli appunti finché stanno in un solo passaggio."""
    budget = sess.budget_input
    livello = 1
    while sum(sess.conta(a) for a in appunti) > budget and len(appunti) > 1:
        gruppi, corrente, n_corr = [], [], 0
        for a in appunti:
            n = sess.conta(a)
            if corrente and n_corr + n > budget:
                gruppi.append(corrente)
                corrente, n_corr = [], 0
            corrente.append(a)
            n_corr += n
        if corrente:
            gruppi.append(corrente)
        if len(gruppi) >= len(appunti):
            break  # nessun progresso possibile: si tronca sotto
        sess._log(f"  ↪ Fusione appunti (livello {livello}): {len(appunti)} → {len(gruppi)}")
        nuovi = []
        for k, g in enumerate(gruppi, 1):
            if len(g) == 1:
                nuovi.append(g[0])
                continue
            nuovi.append(sess.chat(
                _messaggi(_prompt_fondi(cfg, "\n\n".join(g), sess.istruzione_lingua)),
                max_tokens=min(sess.max_output, max(512, budget // 3)),
                etichetta=f"fusione {livello}.{k}/{len(gruppi)}"))
        appunti = nuovi
        livello += 1
    unione = "\n\n".join(appunti)
    if sess.conta(unione) > budget:
        unione = unione[:int(budget * CHARS_PER_TOKEN * 0.95)]
        sess._log("  ⚠ Appunti troncati per stare nel contesto")
    return unione


def genera_nota(sess, testo, tipo, fonte=None):
    """Genera la nota markdown da una trascrizione.

    Ritorna dict: titolo, corpo, tags, categoria, md (con frontmatter).
    Solleva Interrotto se l'utente preme Stop.
    """
    cfg = TIPI_NOTA.get(tipo) or TIPI_NOTA[DEFAULT_TIPO]
    lingua = sess.istruzione_lingua
    n_tok = sess.conta(testo)
    sess._log(f"📏 Trascrizione: ~{n_tok} token · tipo nota: {tipo}")

    if n_tok <= sess.budget_input:
        sess._log("→ Passaggio singolo (la trascrizione sta tutta nel contesto)")
        corpo = sess.chat(_messaggi(_prompt_finale(cfg, testo, False, 1, lingua)),
                          etichetta="stesura nota")
    else:
        map_out = max(384, min(sess.max_output, sess.budget_input // 4, 1536))
        blocchi = dividi_in_blocchi(testo, sess.conta, sess.budget_input)
        sess._log(f"→ Map-reduce: {len(blocchi)} blocchi da ≤{sess.budget_input} token "
                  f"(overlap {OVERLAP_TOKEN}), appunti ≤{map_out} token ciascuno")
        appunti = []
        for i, b in enumerate(blocchi, 1):
            appunti.append(sess.chat(
                _messaggi(_prompt_map(cfg, b, i, len(blocchi), lingua)),
                max_tokens=map_out, etichetta=f"appunti blocco {i}/{len(blocchi)}"))
        materiale = _comprimi_appunti(sess, cfg, appunti)
        corpo = sess.chat(_messaggi(_prompt_finale(cfg, materiale, True, len(blocchi), lingua)),
                          etichetta="stesura nota finale")

    corpo, tags = estrai_tags(corpo)
    titolo = estrai_titolo(corpo)
    if not titolo:
        titolo = os.path.splitext(os.path.basename(fonte or "Nota"))[0]
        corpo = f"# {titolo}\n\n{corpo}"
    categoria = cfg["categoria"]
    if categoria == "auto":
        m = re.search(r"\*\*Tipo:\*\*\s*([A-Za-zÀ-ÿ/ ]+)", corpo)
        categoria = slug(m.group(1).split("|")[0], 20) if m else "generico"
    tags = list(dict.fromkeys([categoria] + tags))

    fm = ["---",
          f"title: {_yaml_str(titolo)}",
          "type: nota",
          f"categoria: {categoria}"]
    if fonte:
        fm.append(f"source: {_yaml_str(fonte)}")
    fm += [f"date: {_oggi()}",
           f"model: {_yaml_str(sess.nome_modello)}",
           f"tags: {_yaml_lista(tags)}",
           "---", ""]
    return dict(titolo=titolo, corpo=corpo.strip(), tags=tags, categoria=categoria,
                md="\n".join(fm) + corpo.strip() + "\n")


# ===========================================================================
# Wiki: ingest delle sorgenti e domande
# ===========================================================================
TIPI_ENTITA = {"persona", "progetto", "strumento", "organizzazione", "azienda",
               "prodotto", "luogo", "team"}


def _estrai_json(testo):
    try:
        return json.loads(testo)
    except Exception:
        pass
    m = re.search(r"\{.*\}", testo or "", flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def estrai_concetti(sess, corpo_nota):
    corpo = corpo_nota
    if sess.conta(corpo) > sess.budget_input:
        corpo = corpo[:int(sess.budget_input * CHARS_PER_TOKEN * 0.9)]
    prompt = (
        f"<nota>\n{corpo}\n</nota>\n\n"
        "COMPITO: dalla nota estrai i 3-8 concetti o entità più importanti da trasformare "
        "in pagine di una wiki personale (persone, progetti, strumenti, organizzazioni, "
        "teorie, termini tecnici, argomenti). Escludi elementi generici.\n"
        "Rispondi SOLO con un oggetto JSON valido, senza altro testo, con le chiavi:\n"
        '- "riassunto": una frase che riassume la nota\n'
        '- "concetti": lista di oggetti con "nome" (nome canonico), "tipo" (UNA sola '
        'parola tra: concetto, persona, progetto, strumento, organizzazione) e '
        '"descrizione" (1-2 frasi basate solo sulla nota)\n'
        'Esempio di forma: {"riassunto": "...", "concetti": [{"nome": "Teorema di '
        'Pitagora", "tipo": "concetto", "descrizione": "..."}]}\n'
        f"{sess.istruzione_lingua}"
    )
    try:
        out = sess.chat(_messaggi(prompt), max_tokens=min(1024, sess.max_output),
                        json_mode=True, etichetta="estrazione concetti")
    except ErroreLLM as e:
        sess._log(f"  ⚠ Estrazione concetti non riuscita: {e}")
        return "", []
    dati = _estrai_json(out)
    concetti = []
    for c in dati.get("concetti", []) if isinstance(dati, dict) else []:
        if isinstance(c, dict) and str(c.get("nome", "")).strip():
            concetti.append(c)
    riassunto = str(dati.get("riassunto", "")).strip() if isinstance(dati, dict) else ""
    return riassunto, concetti[:8]


def _aggiorna_index(index_path, sezione, voce, chiave, sostituisci=True):
    """Inserisce `voce` nella sezione '## <sezione>…' dell'index.
    Se una riga-elenco contiene `chiave` viene sostituita (o lasciata, se
    sostituisci=False). Rimuove il segnaposto _(vuoto)_ della sezione."""
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            righe = f.read().splitlines()
    else:
        righe = ["# Indice della Wiki", ""]
    for i, r in enumerate(righe):
        if chiave in r and r.lstrip().startswith("- "):
            if sostituisci:
                righe[i] = voce
                with open(index_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(righe) + "\n")
            return
    h = next((i for i, r in enumerate(righe)
              if r.strip().lower().startswith(f"## {sezione}".lower())), None)
    if h is None:
        while righe and not righe[-1].strip():
            righe.pop()
        righe += ["", f"## {sezione}", ""]
        h = len(righe) - 2
    fine = next((j for j in range(h + 1, len(righe)) if righe[j].startswith("## ")), len(righe))
    sez = [r for r in righe[h + 1:fine] if r.strip() != "_(vuoto)_"]
    while sez and not sez[-1].strip():
        sez.pop()
    if not sez:
        sez = [""]
    sez += [voce, ""]
    righe = righe[:h + 1] + sez + righe[fine:]
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(righe).rstrip() + "\n")


def _append_log(wiki_dir, tipo, titolo, punti):
    path = os.path.join(wiki_dir, "log.md")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Log della Wiki\n\nRegistro cronologico append-only.\n\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"## [{_oggi()}] {tipo} | {titolo}\n")
        for p in punti:
            f.write(f"- {p}\n")
        f.write("\n")


def _aggiungi_a_lista_yaml(testo, chiave, valore):
    def _sub(m):
        voci = [v.strip() for v in m.group(1).split(",") if v.strip()]
        if valore not in voci:
            voci.append(valore)
        return f"{chiave}: [{', '.join(voci)}]"
    return re.sub(rf"^{chiave}: \[(.*)\]$", _sub, testo, count=1, flags=re.M)


def _aggiorna_pagina_concetto(wiki_dir, c, pagina_sorgente, titolo_sorgente, raw_stem, modello):
    nome = str(c.get("nome", "")).strip()[:80]
    tipo = re.split(r"[|/,]", str(c.get("tipo") or "concetto"))[0].lower().strip() or "concetto"
    prefisso = "entita" if tipo in TIPI_ENTITA else "concetto"
    stem = f"{prefisso}_{slug(nome, 50)}"
    path = os.path.join(wiki_dir, stem + ".md")
    desc = str(c.get("descrizione", "")).strip()
    rif = f"- [[{pagina_sorgente}|{titolo_sorgente}]]" + (f" — {desc}" if desc else "")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        if f"[[{pagina_sorgente}" in txt:
            return stem, prefisso, nome, desc, None
        txt = re.sub(r"^aggiornato: .*$", f"aggiornato: {_oggi()}", txt, count=1, flags=re.M)
        txt = _aggiungi_a_lista_yaml(txt, "fonti", raw_stem)
        if "## Riferimenti" not in txt:
            txt = txt.rstrip() + "\n\n## Riferimenti\n"
        txt = txt.rstrip() + "\n" + rif + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        return stem, prefisso, nome, desc, False

    contenuto = (
        "---\n"
        f"title: {_yaml_str(nome)}\n"
        f"type: {prefisso}\n"
        f"sottotipo: {slug(tipo, 20)}\n"
        f"fonti: [{raw_stem}]\n"
        f"aggiornato: {_oggi()}\n"
        f"generato_da: {_yaml_str(modello)}\n"
        f"tags: [{prefisso}]\n"
        "---\n\n"
        f"# {nome}\n\n{desc or '_(da completare)_'}\n\n"
        f"## Riferimenti\n{rif}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(contenuto)
    return stem, prefisso, nome, desc, True


def ingest_sorgente(sess, cartella_wiki, raw_path, tipo, testo=None, nota=None):
    """Ingest di UNA sorgente raw/ nella wiki:
    nota di sintesi (wiki/sorgente_<raw>.md) + pagine concetto/entità +
    aggiornamento di index.md e log.md. Ritorna il percorso della pagina."""
    wiki_dir = os.path.join(cartella_wiki, "wiki")
    os.makedirs(wiki_dir, exist_ok=True)
    raw_stem = os.path.splitext(os.path.basename(raw_path))[0]
    with open(raw_path, "r", encoding="utf-8") as f:
        contenuto = f.read()
    fonte = _leggi_frontmatter(contenuto).get("source") or raw_stem

    if nota is None:
        testo = testo if testo is not None else testo_pulito_trascrizione(contenuto)
        if not testo.strip():
            sess._log(f"  ⚠ '{raw_stem}': nessun parlato, salto")
            return None
        nota = genera_nota(sess, testo, tipo, fonte=fonte)

    pagina = f"sorgente_{raw_stem}"
    riassunto, concetti = estrai_concetti(sess, nota["corpo"])
    if not riassunto:
        m = re.search(r"^>\s*(?!\*\*Tipo)(.+)$", nota["corpo"], flags=re.M)
        riassunto = m.group(1).strip() if m else ""

    creati, aggiornati, collegamenti = [], [], []
    for c in concetti:
        stem, prefisso, nome, desc, nuovo = _aggiorna_pagina_concetto(
            wiki_dir, c, pagina, nota["titolo"], raw_stem, sess.nome_modello)
        collegamenti.append(f"- [[{stem}|{nome}]]" + (f" — {desc}" if desc else ""))
        if nuovo is True:
            creati.append(stem)
        elif nuovo is False:
            aggiornati.append(stem)
        _aggiorna_index(os.path.join(wiki_dir, "index.md"),
                        "Entità" if prefisso == "entita" else "Concetti",
                        f"- [[{stem}|{nome}]] — {desc[:140]}", f"[[{stem}|", sostituisci=False)

    fm = ("---\n"
          f"title: {_yaml_str(nota['titolo'])}\n"
          "type: sorgente\n"
          f"categoria: {nota['categoria']}\n"
          f"fonti: [{raw_stem}]\n"
          f"aggiornato: {_oggi()}\n"
          f"generato_da: {_yaml_str(sess.nome_modello)}\n"
          f"tags: {_yaml_lista(nota['tags'])}\n"
          "---\n\n")
    corpo = nota["corpo"].rstrip()
    if collegamenti:
        corpo += "\n\n## Collegamenti\n" + "\n".join(collegamenti)
    corpo += f"\n\n## Fonte\n- Trascrizione: [[{raw_stem}]] (`{fonte}`)\n"
    path_pagina = os.path.join(wiki_dir, pagina + ".md")
    with open(path_pagina, "w", encoding="utf-8") as f:
        f.write(fm + corpo)

    voce = f"- [[{pagina}|{nota['titolo']}]] — {riassunto[:160]} · raw: [[{raw_stem}]]"
    _aggiorna_index(os.path.join(wiki_dir, "index.md"), "Sorgenti", voce, f"[[{raw_stem}]]")
    punti = [f"Sorgente: [[{raw_stem}]] → [[{pagina}]]"]
    if creati:
        punti.append("Pagine create: " + ", ".join(f"[[{s}]]" for s in creati))
    if aggiornati:
        punti.append("Pagine aggiornate: " + ", ".join(f"[[{s}]]" for s in aggiornati))
    punti.append(f"Modello: {sess.nome_modello}")
    _append_log(wiki_dir, "ingest", nota["titolo"], punti)
    sess._log(f"  📚 Wiki: [[{pagina}]] · {len(creati)} pagine nuove, {len(aggiornati)} aggiornate")
    return path_pagina


def sorgenti_da_ingerire(cartella_wiki):
    raw_dir = os.path.join(cartella_wiki, "raw")
    wiki_dir = os.path.join(cartella_wiki, "wiki")
    if not os.path.isdir(raw_dir):
        return [], []

    def _ord(n):
        num = re.findall(r"\d+", n)
        return (int(num[0]) if num else 0, n)

    tutte = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(".md")], key=_ord)
    da_fare = [f for f in tutte if not os.path.exists(
        os.path.join(wiki_dir, f"sorgente_{os.path.splitext(f)[0]}.md"))]
    return tutte, da_fare


def ingest_cartella_wiki(cartella_wiki, spec, opzioni, tipo, callback=None,
                         stato_cb=None, progress_cb=None, stop_event=None,
                         pause_event=None):
    """Ingest di tutte le sorgenti raw/ non ancora sintetizzate (ripartibile:
    una sorgente è 'fatta' quando esiste wiki/sorgente_<raw>.md)."""
    log = callback or (lambda m: None)
    tutte, da_fare = sorgenti_da_ingerire(cartella_wiki)
    if not tutte:
        log("⚠ Nessuna sorgente in raw/.")
        return {"n_tot": 0, "n_nuovi": 0, "interrotto": False}
    if not da_fare:
        log(f"✅ Tutte le {len(tutte)} sorgenti sono già nella wiki.")
        return {"n_tot": len(tutte), "n_nuovi": 0, "interrotto": False}

    testi = {}
    for f in da_fare:
        with open(os.path.join(cartella_wiki, "raw", f), "r", encoding="utf-8") as fh:
            testi[f] = testo_pulito_trascrizione(fh.read())
    vuote = [f for f in da_fare if not testi[f].strip()]
    da_fare = [f for f in da_fare if testi[f].strip()]
    if vuote:
        log(f"⏭  Senza parlato (saltate): {', '.join(vuote)}")
    if not da_fare:
        log("✅ Nessuna nuova sorgente con contenuto da ingerire.")
        return {"n_tot": len(tutte), "n_nuovi": 0, "interrotto": False}
    log(f"📚 Sorgenti: {len(tutte)} · da ingerire: {len(da_fare)}")

    sess = SessioneLLM(spec, opzioni, callback, stato_cb, progress_cb, stop_event, pause_event,
                       fase="llm_ingest", sorgente=os.path.basename(os.path.normpath(cartella_wiki)))
    n_ok, interrotto = 0, False
    try:
        try:
            sess.avvia(list(testi.values()))
        except Interrotto:
            return {"n_tot": len(tutte), "n_nuovi": 0, "interrotto": True}
        for i, f in enumerate(da_fare, 1):
            if sess.fermato():
                interrotto = True
                break
            log(f"\n🧠 [{i}/{len(da_fare)}] Ingest {f}")
            if progress_cb:
                progress_cb((i - 1) / len(da_fare))
            try:
                if ingest_sorgente(sess, cartella_wiki, os.path.join(cartella_wiki, "raw", f),
                                   tipo, testo=testi[f]):
                    n_ok += 1
            except Interrotto:
                interrotto = True
                break
            except ErroreLLM as e:
                log(f"  ✗ {f}: {e} — passo alla prossima")
    finally:
        sess.chiudi()
    return {"n_tot": len(tutte), "n_nuovi": n_ok, "interrotto": interrotto}


_STOPWORDS = set("""
il lo la i gli le un una uno di da in con su per tra fra che chi cosa come quando dove
perche perché quale quali quanto quanti questo questa questi queste quello quella quelli
sono essere stato stata era erano hanno abbiamo avere della delle degli dello dei del alla
alle agli allo nel nella nelle negli nei sul sulla sui sulle non più piu anche ancora molto
poi però pero oppure ovvero cioè cioe tutto tutti tutte dire detto fare fatto parla parlato
the and for with what which when where how why that this from into about have has was were
are is does did can could should would
""".split())


def _termini(domanda):
    parole = re.findall(r"\w+", domanda.lower())
    termini = [p for p in dict.fromkeys(parole) if len(p) >= 4 and p not in _STOPWORDS]
    return [t if len(t) <= 5 else t[:-2] for t in termini]


def chiedi_alla_wiki(cartella_wiki, domanda, spec, opzioni, callback=None,
                     stato_cb=None, progress_cb=None, stop_event=None,
                     pause_event=None, salva=True):
    """Risponde a una domanda usando le pagine della wiki (+ raw/) come contesto.
    Seleziona le pagine più pertinenti finché stanno nel contesto, salva la
    risposta come wiki/query_<data>_<slug>.md e aggiorna index/log.
    Ritorna (risposta_md, percorso_pagina, pagine_usate)."""
    log = callback or (lambda m: None)
    wiki_dir = os.path.join(cartella_wiki, "wiki")
    raw_dir = os.path.join(cartella_wiki, "raw")
    os.makedirs(wiki_dir, exist_ok=True)

    pagine = []  # (stem, contenuto, peso)
    for d, peso in ((wiki_dir, 1.0), (raw_dir, 0.6)):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".md") or f in ("log.md", "index.md"):
                continue
            with open(os.path.join(d, f), "r", encoding="utf-8") as fh:
                pagine.append((os.path.splitext(f)[0], fh.read(), peso))
    index_txt = ""
    if os.path.exists(os.path.join(wiki_dir, "index.md")):
        with open(os.path.join(wiki_dir, "index.md"), "r", encoding="utf-8") as fh:
            index_txt = fh.read()
    if not pagine and not index_txt:
        raise ErroreLLM("La wiki è vuota: esegui prima un batch o un ingest.")

    sess = SessioneLLM(spec, opzioni, callback, stato_cb, progress_cb, stop_event, pause_event,
                       fase="llm_query", sorgente=os.path.basename(os.path.normpath(cartella_wiki)))
    try:
        sess.avvia(["\n".join(p[1] for p in pagine) + index_txt])
        budget = sess.budget_input - sess.conta(domanda) - 200

        termini = _termini(domanda)
        punteggi = []
        for stem, txt, peso in pagine:
            basso = txt.lower()
            s = sum(min(basso.count(t), 20) + (10 if t in stem.lower() else 0) for t in termini)
            punteggi.append((s * peso, stem, txt))
        punteggi.sort(key=lambda x: -x[0])
        scelte = [p for p in punteggi if p[0] > 0]
        if not scelte:  # nessun match: usa le sintesi, poi le trascrizioni
            scelte = sorted(punteggi, key=lambda x: (not x[1].startswith("sorgente_"), x[1]))

        blocchi, usate = [], []
        if index_txt:
            idx = index_txt
            if sess.conta(idx) > budget // 5:
                idx = idx[:int(budget // 5 * CHARS_PER_TOKEN)]
            blocchi.append(f'<pagina nome="index">\n{idx}\n</pagina>')
            budget -= sess.conta(idx) + 20
        for _, stem, txt in scelte:
            if budget < 200:
                break
            n = sess.conta(txt) + 20
            if n > budget:
                if budget < 400:
                    continue
                txt = txt[:int((budget - 20) * CHARS_PER_TOKEN * 0.95)] + "\n[…troncato]"
                n = budget
            blocchi.append(f'<pagina nome="{stem}">\n{txt}\n</pagina>')
            usate.append(stem)
            budget -= n
        log(f"🔎 Pagine usate come contesto: {len(usate)} su {len(pagine)}")

        prompt = (
            f"Domanda: {domanda}\n\n"
            "Qui sotto trovi le pagine pertinenti di una wiki personale (pagine di sintesi "
            "e, se presenti, trascrizioni originali).\n\n"
            "<pagine>\n" + "\n\n".join(blocchi) + "\n</pagine>\n\n"
            "COMPITO: rispondi alla domanda in markdown usando SOLO le informazioni nelle pagine.\n"
            "- Cita le fonti accanto alle affermazioni con wikilink [[nome]] usando i nomi "
            "indicati in nome=\"...\".\n"
            "- Se le pagine non contengono la risposta, dillo chiaramente e indica cosa manca.\n"
            "- Struttura: risposta diretta (1-3 frasi), poi i dettagli, poi '## Fonti' "
            "con l'elenco dei wikilink usati.\n"
            f"{sess.istruzione_lingua}"
        )
        risposta = sess.chat(_messaggi(prompt), etichetta="risposta")
    finally:
        sess.chiudi()

    path = None
    if salva:
        stem = f"query_{_oggi()}_{slug(domanda, 50)}"
        path = os.path.join(wiki_dir, stem + ".md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("---\n"
                    f"title: {_yaml_str(domanda)}\n"
                    "type: query\n"
                    f"fonti: {_yaml_lista(usate)}\n"
                    f"aggiornato: {_oggi()}\n"
                    f"generato_da: {_yaml_str(sess.nome_modello)}\n"
                    "tags: [domanda]\n"
                    "---\n\n"
                    f"# {domanda}\n\n{risposta}\n")
        _aggiorna_index(os.path.join(wiki_dir, "index.md"), "Domande",
                        f"- [[{stem}|{domanda[:120]}]] — {_oggi()}", f"[[{stem}|")
        _append_log(wiki_dir, "query", domanda[:120],
                    [f"Risposta: [[{stem}]]", f"Pagine usate: {len(usate)}",
                     f"Modello: {sess.nome_modello}"])
    return risposta, path, usate
