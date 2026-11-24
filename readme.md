![Audio Splitter & Transcriber](from_video_to_wiki.png)

# 🎬 From Video to Wiki - on local

## Il problema

Hai ore di registrazioni di chiamate, riunioni, lezioni da recuperare e sintetizzare — ma non hai il tempo di riascoltarle una per una e dovresti fare altro di prioritario.

Non parliamo di video online come YouTube: per quelli esistono già tantissimi tool in grado di trascrivere e creare chatbot interrogabili sulle fonti.

Parliamo di **registrazioni video e audio in locale**, sul tuo device. File che restano nei tuoi hard disk, nei NAS aziendali, nelle cartelle condivise. Come trasformare GB di video in **testo ricercabile e utilizzabile** dai tuoi strumenti AI — aziendali e non?

## La soluzione

Ho creato un processo **interamente in locale**, open source, che estrae e trascrive il contenuto audio di qualsiasi registrazione video — senza inviare nulla al cloud, senza abbonamenti, senza limiti di utilizzo.

Un'applicazione desktop con interfaccia grafica moderna che **estrae l'audio**, lo **segmenta** in parti di dimensione configurabile e lo **trascrive automaticamente** usando modelli Whisper. Supporta **italiano** e **inglese** con rilevamento automatico della lingua. Funziona su **CPU** o **GPU NVIDIA** (con accelerazione CUDA), con fallback automatico su CPU se CUDA non è disponibile.

In più, può elaborare **un'intera cartella di video in batch** e generare lo scaffold di una **LLM Wiki** — una base di conoscenza in markdown, pronta per Obsidian, che un coding agent (es. Claude Code) può poi popolare e mantenere. Per governance del dato, utilizzare coding agent sotto licenze aziendali, altrimenti preferire soluzioni interamente locali se si ha un hardware opportuno.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)

---

## ✨ Funzionalità

- **Estrazione + segmentazione audio** da qualsiasi formato video (MP4, AVI, MKV, MOV, ecc.) con dimensione segmenti configurabile
- **Trascrizione locale** con [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — modello consigliato `large-v3-turbo`, accurato su italiano e inglese
- **Accelerazione GPU NVIDIA** opzionale: rilevamento automatico delle DLL CUDA installate via pip (cuBLAS, cuDNN); selezione `cuda`, `cpu` o `auto` direttamente dalla GUI
- **Fallback automatico CUDA → CPU** se la GPU non è disponibile o le librerie CUDA non sono caricabili — l'app non si blocca, segnala l'errore nel log e prosegue su CPU
- **Pipeline one-click**: un unico pulsante esegue splitting + trascrizione in sequenza senza intervento manuale
- **Batch cartella → LLM Wiki**: elabora in sequenza tutti i video di una cartella e genera una struttura "LLM Wiki" pronta per Obsidian (sorgenti immutabili in `raw/`, schema agente `CLAUDE.md`/`AGENTS.md`), da far popolare a un coding agent come Claude Code. Il modello Whisper viene caricato una sola volta e riusato per tutti i video; il batch è ripartibile se interrotto
- **Cleanup deterministico delle risorse GPU** a fine trascrizione (svuotamento esplicito della cache CUDA): l'app resta aperta dopo il completamento e ti permette di consultare con calma le metriche finali
- **Pulizia automatica**: i file audio segmentati vengono eliminati al termine della trascrizione completata con successo
- **Checkpoint/resume**: il progresso viene salvato in `_progresso.json` dopo ogni segmento — se il processo viene interrotto, alla ripresa parte dal punto in cui si era fermato
- **Pausa e Stop durante l'esecuzione**: pulsanti ⏸ e ⏹ accessibili dal log; allo stop viene generato comunque il file di trascrizione parziale con quanto elaborato fino a quel momento
- **Filtro VAD** integrato contro le allucinazioni nei silenzi
- **Output flessibile**: testo semplice o dettagliato con timestamp, con report completo delle metriche
- **Sintesi con LLM open source in locale**: la trascrizione viene passata a un modello locale (Qwen3, Llama 3.1, Gemma 3, Mistral Nemo, Phi-4, gpt-oss…) via [Ollama](https://ollama.com/) o HuggingFace GGUF (llama.cpp) e trasformata in una **nota `.md`** strutturata in base al tipo di registrazione (standup, riunione, lezione da studiare, talk, intervista, brainstorming…)
- **Gestione esplicita della finestra di contesto**: contesto adattato al testo, misurazione dei token e **map-reduce** automatico per registrazioni lunghe (niente troncamenti silenziosi)
- **Download automatico dei modelli** al primo utilizzo (Ollama o HuggingFace), con avanzamento nel log
- **Knowledge base che cresce**: ogni nota può essere archiviata in una Wiki (sorgente in `raw/`, sintesi, pagine concetto/entità collegate, `index.md`, `log.md`); il "cervello" LLM può fare l'ingest di intere cartelle e **rispondere a domande** sulla Wiki, archiviando le risposte
- **Thread CPU configurabili** per Whisper e per l'LLM, accelerazione GPU o *solo CPU*: per sfruttare tutti i core e fare **test di performance** confrontabili
- **Log di performance**: intestazione hardware in ogni log, velocità per segmento (x tempo reale), tok/s di lettura prompt e generazione, quota del modello su GPU e una riga per test in `logs/performance.csv`
- **macOS e Windows** con lo stesso codice: rilevamento hardware nativo, finestra adattiva allo scaling DPI di Windows, nessuna console che lampeggia
- **Interfaccia moderna** con CustomTkinter (dark/light), impostazioni ricordate tra un avvio e l'altro + modalità CLI standalone (`transcriber.py`)

---

## 📋 Prerequisiti

### 1. Python 3.8 o superiore

Verifica se Python è già installato aprendo un terminale:

```bash
python --version
```

Se non è installato:

- **Windows**: scarica da [python.org/downloads](https://www.python.org/downloads/). Durante l'installazione **spunta "Add Python to PATH"**.
- **macOS**: `brew install python@3.12 python-tk@3.12` oppure scarica da [python.org](https://www.python.org/downloads/) (l'installer include Tk). ⚠️ **Non usare il Python di Apple** (quello delle Command Line Tools, `/Library/Developer/CommandLineTools`): include Tk 8.5 e la GUI si apre con la **finestra bianca**. Serve Tk ≥ 8.6 — vedi [Utilizzo su macOS](#-utilizzo-su-macos).
- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install python3 python3-pip python3-venv`

### 2. FFmpeg

FFmpeg è necessario per estrarre e segmentare l'audio dai video. Verifica se è installato:

```bash
ffmpeg -version
```

Se non è installato:

<details>
<summary><b>🪟 Windows</b></summary>

**Opzione A — winget (consigliata, Windows 10/11):**
```bash
winget install FFmpeg
```
Riavvia il terminale dopo l'installazione.

**Opzione B — Installazione manuale:**
1. Vai su [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) e scarica `ffmpeg-release-essentials.zip`
2. Estrai lo ZIP in una cartella, es. `C:\ffmpeg`
3. Aggiungi `C:\ffmpeg\bin` al PATH di sistema:
   - Cerca "Variabili d'ambiente" nel menu Start
   - Modifica la variabile `Path` → Aggiungi `C:\ffmpeg\bin`
4. Riavvia il terminale e verifica con `ffmpeg -version`
</details>

<details>
<summary><b>🍎 macOS</b></summary>

```bash
brew install ffmpeg
```
Se non hai Homebrew: [brew.sh](https://brew.sh/)
</details>

<details>
<summary><b>🐧 Linux (Ubuntu/Debian)</b></summary>

```bash
sudo apt update
sudo apt install ffmpeg
```
</details>

### 3. Git (per clonare il repository)

```bash
git --version
```

Se non è installato:
- **Windows**: scarica da [git-scm.com](https://git-scm.com/)
- **macOS**: `brew install git` oppure `xcode-select --install`
- **Linux**: `sudo apt install git`

---

## 🚀 Installazione del progetto

### Installazione base (CPU)

```bash
# 1. Clona il repository
git clone https://github.com/PierpaoloPalmiotti/from_video_to_wiki.git
cd from_video_to_wiki

# 2. Crea un virtual environment (consigliato)
python -m venv venv

# 3. Attiva il virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Installa le dipendenze base
pip install -r requirements.txt
```

### 🧠 LLM locale per le note (opzionale ma consigliato)

Per generare le note `.md` e lavorare con la Wiki serve un LLM locale. Due opzioni:

- **Ollama (consigliato)** — installa da [ollama.com/download](https://ollama.com/download). Nessun pacchetto Python aggiuntivo: l'app avvia Ollama se non è attivo e **scarica da sola** il modello scelto al primo utilizzo (oppure con **⬇ Scarica**).
- **HuggingFace GGUF via llama.cpp** — `pip install -r requirements-hf.txt`, poi scegli un modello `· HuggingFace`. Il file GGUF viene scaricato nella cache di HuggingFace (`~/.cache/huggingface/`).

Senza LLM l'app continua a funzionare come prima: basta togliere la spunta **🧠 Genera nota .md con LLM**.

### 🍎 Utilizzo su macOS

Su Mac l'app funziona, ma con alcune differenze rispetto a Windows/Linux con GPU NVIDIA.

**Installazione consigliata (Apple Silicon e Intel):**

```bash
# 1. Python con Tk 8.6+ e FFmpeg
brew install python@3.12 python-tk@3.12 ffmpeg

# 2. Verifica la versione di Tk (deve essere >= 8.6, non 8.5)
/opt/homebrew/bin/python3.12 -c "import tkinter; print(tkinter.TkVersion)"

# 3. Virtual environment con quel Python
cd from_video_to_wiki
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate

# 4. Dipendenze (senza torch e senza requirements-gpu.txt) + av compatibile
pip install customtkinter moviepy faster-whisper "av<17"

# 5. Avvio
python main.py
```

**Cosa cambia su Mac:**

- **FFmpeg**: `ffmpeg.exe` nella cartella del progetto è un binario Windows e su Mac non viene usato. L'app usa `ffmpeg` dal PATH, quindi serve `brew install ffmpeg`.
- **Device**: lascia **`auto`** (diventa CPU con `int8`). `cuda` non esiste su Mac e faster-whisper non supporta la GPU Apple (Metal/MPS), quindi la GPU del Mac non viene sfruttata.
- **`requirements-gpu.txt`**: ignoralo, è solo per NVIDIA.
- **`torch`**: su Mac non serve (nessuna GPU CUDA da rilevare) e pesa circa 2 GB. L'app lo gestisce comunque se manca e usa la CPU.
- **Precisione**: il tipo di calcolo su CPU è fisso a `int8`. Per Whisper la perdita di qualità rispetto a `float32` è minima, mentre `float32` userebbe circa 4 volte la memoria e sarebbe più lento. Per migliorare i risultati è più efficace forzare la lingua (`it`) che cambiare precisione.
- **Font del log**: il font `Consolas` non esiste su Mac, Tk ne usa uno di default. È solo estetico.

**Modello consigliato su Mac:**

| Mac | Modello |
|---|---|
| Prima prova per verificare che tutto funzioni | `small` |
| Uso reale (≥ 16 GB di RAM) | `large-v3-turbo` |
| 8 GB di RAM o poco tempo a disposizione | `medium` |

**Prestazioni misurate (MacBook M5 Pro, 24 GB, `large-v3-turbo`, CPU int8):** 118 min di audio trascritti in circa 55 minuti, cioè **~2,2x tempo reale**. Su un video `.mov` da 460 MB (registrazione schermo) il tempo end-to-end è stato di circa **7 sec/MB**.

> 💡 **Segmenti**: con "Max MB/segmento" = 3 una registrazione lunga produce centinaia di segmenti (398 nel test sopra). Per registrazioni lunghe alza il valore a 10-25 MB.

> 💡 **Per sfruttare la GPU Apple** servirebbe un backend diverso, come [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper): l'app oggi non lo supporta.

### Installazione opzionale per GPU NVIDIA

Se hai una GPU NVIDIA e vuoi sfruttare l'accelerazione CUDA (drasticamente più veloce della CPU), installa anche le librerie cuBLAS e cuDNN. Sono pacchetti pesanti (~700 MB), quindi alza il timeout di pip:

```bash
pip install --timeout 300 -r requirements-gpu.txt
```

> 💡 Non serve installare manualmente CUDA Toolkit o NVIDIA Driver Pack: i pacchetti `nvidia-cublas-cu12` e `nvidia-cudnn-cu12` contengono le DLL già pronte. L'app le rileva automaticamente e le aggiunge al `PATH` al primo avvio.
>
> ⚠️ **Driver NVIDIA**: assicurati comunque di avere i driver NVIDIA aggiornati installati a livello di sistema (i pacchetti pip portano solo le librerie utente, non il driver kernel-mode).

### ⚠️ Nota sul primo avvio

Al primo avvio della trascrizione, il modello Whisper scelto verrà **scaricato automaticamente** da Hugging Face. Le dimensioni variano in base al modello:

| Modello | Download |
|---|---|
| `tiny` | ~75 MB |
| `small` | ~460 MB |
| `medium` | ~1.5 GB |
| `large-v3-turbo` | ~1.6 GB |
| `large-v3` | ~3 GB |

Il download avviene una sola volta — **la connessione internet è necessaria solo per questo passaggio**. Tutte le esecuzioni successive funzionano completamente offline.

Se preferisci scaricare il modello in anticipo (ad esempio quando hai una buona connessione), puoi farlo da terminale:

```bash
# Pre-download del modello consigliato
huggingface-cli download Systran/faster-whisper-large-v3-turbo

# Oppure di un altro modello
huggingface-cli download Systran/faster-whisper-medium
```

---

## 📂 Struttura del progetto

```
from_video_to_wiki/
│
├── main.py                  # Applicazione GUI (CustomTkinter)
├── llm_engine.py            # Sintesi LLM locale (Ollama / HuggingFace), contesto, Wiki
├── perf.py                  # Info di sistema (CPU/RAM/GPU) e logs/performance.csv
├── transcriber.py           # Modulo CLI standalone per trascrizione
├── requirements.txt         # Dipendenze base (CPU)
├── requirements-gpu.txt     # Dipendenze opzionali per GPU NVIDIA
├── requirements-hf.txt      # Dipendenze opzionali per LLM HuggingFace (llama.cpp)
├── readme.md                # Documentazione
├── LICENSE                  # Licenza MIT
├── .gitignore               # Esclusioni Git (solo locale, non versionato)
│
├── ffmpeg.exe               # ⚠️ Solo Windows, opzionale (vedi nota sotto)
│
└── venv/                    # Virtual environment (generato, non versionato)
```

> ⚠️ **ffmpeg.exe**: su Windows puoi copiare `ffmpeg.exe` direttamente nella cartella del progetto come alternativa all'installazione globale nel PATH. Su macOS e Linux, FFmpeg va installato a livello di sistema (vedi [Prerequisiti](#-prerequisiti)). Il file non va versionato.

> 💡 **venv/** e i modelli Whisper (salvati in `~/.cache/huggingface/`) non sono inclusi nel repository. Vengono generati localmente durante l'installazione e il primo avvio.

### Output generato a runtime

Quando elabori un singolo video, l'applicazione crea una cartella dedicata accanto al file sorgente:

```
cartella_video/
├── intervista.mp4
└── intervista/                               # Cartella output (nome = nome video)
    ├── intervista_segmento_1.wav             # Eliminati automaticamente dopo
    ├── intervista_segmento_2.wav             # la trascrizione completata
    ├── intervista_segmento_3.wav             #
    ├── intervista_trascrizione.txt           # Trascrizione
    ├── intervista_nota.md                    # Nota LLM (se la spunta è attiva)
    └── _progresso.json                       # Checkpoint (rimosso a completamento)
```

> 💡 Le impostazioni (modelli, parametri, cartella della knowledge base) vengono ricordate in `~/.from_video_to_wiki/settings.json`.

> 💡 I file audio segmentati vengono **eliminati automaticamente** al termine di una trascrizione completata con successo. In caso di interruzione, i file restano e `_progresso.json` tiene traccia di quanto già trascritto per permettere la ripresa.

---

## 💻 Utilizzo

### Interfaccia grafica (GUI)

```bash
python main.py
```

L'interfaccia è organizzata in tre blocchi, dall'alto verso il basso:

**1 │ Impostazioni** — da configurare per prime, valgono per tutto il resto:
- *Trascrizione*: modello Whisper, lingua, formato (`txt`/`dettagliato`), device (`auto`/`cpu`/`cuda`), MB per segmento, **thread CPU**
- *Sintesi LLM*: modello (menu a tendina con catalogo Ollama/HuggingFace, modelli già installati in Ollama e **✏️ Personalizzato…**), **tipo di contenuto** (Auto, Standup/Daily, Riunione/Call, Lezione/Studio, Webinar/Talk, Intervista, Brainstorming, Riassunto generico), lingua della nota, **contesto**, **max output**, **temperatura**, **thread LLM** e **accelerazione** (GPU o solo CPU). La riga informativa mostra stato del modello (installato / da scaricare), dimensione, contesto massimo, stima della RAM per la KV cache e un avviso se il modello non sta in memoria

> 💡 Accanto a ogni parametro c'è un'icona **ⓘ**: passandoci sopra con il mouse (o cliccandola) compare una spiegazione breve di cosa fa.

**2 │ Singolo contenuto** — estrai → trascrivi → nota:
1. Seleziona il video con **📁**
2. (Opzionale) scegli una **Knowledge base**: la nota verrà anche archiviata nella Wiki, che cresce a ogni contenuto
3. Lascia attiva **🧠 Genera nota .md con LLM** e clicca **⚡ Elabora + Trascrivi + Nota**
4. Altre azioni: **▶ Solo estrai audio**, **🎙 Trascrivi segmenti**, **📂 Da cartella audio…** (anche queste generano la nota se la spunta è attiva) e **📝 Nota da trascrizione…** per creare la nota da un `.txt` già esistente

**3 │ Knowledge Base** — cartella → LLM Wiki (vedi [Batch & LLM Wiki](#-batch--llm-wiki)), con un **Cervello Wiki** selezionabile (anche diverso dal modello delle note)

Durante l'esecuzione compaiono **⏸ Pausa** e **⏹ Stop** nel pannello log. Allo stop viene generato comunque il file di trascrizione con quanto elaborato fino a quel momento; durante la fase LLM lo stop interrompe la generazione in corso (la trascrizione resta salvata e la nota si può rigenerare con **📝 Nota da trascrizione…**).

A fine elaborazione il log riporta il report end-to-end, le statistiche LLM (token in/out, tok/s, eventuali avvisi di troncamento) e i percorsi di trascrizione, nota e pagina Wiki.

### Riga di comando (CLI)

```bash
# Trascrivi una cartella di file audio
python transcriber.py /percorso/cartella_audio/

# Forza italiano e formato dettagliato con timestamp
python transcriber.py /percorso/cartella_audio/ --lingua it --formato dettagliato

# Usa la GPU
python transcriber.py /percorso/cartella_audio/ --device cuda

# Usa un modello più leggero se la RAM scarseggia
python transcriber.py /percorso/cartella_audio/ --modello medium

# Specifica un file di output personalizzato
python transcriber.py /percorso/cartella_audio/ --output risultato.txt
```

**Opzioni CLI disponibili:**

| Opzione | Default | Descrizione |
|---|---|---|
| `--modello` | `large-v3-turbo` | Modello Whisper da usare |
| `--lingua` | auto-detect | Codice lingua (`it`, `en`, ecc.) |
| `--device` | `auto` | `cpu`, `cuda` o `auto` |
| `--beam-size` | `5` | Beam search size (1 = veloce, 5 = accurato) |
| `--formato` | `txt` | `txt` o `dettagliato` |
| `--output` | `trascrizione.txt` | Percorso file di output |
| `--no-vad` | disattivato | Disabilita il filtro anti-allucinazioni |
| `--cpu-threads` | `0` (= 4) | Thread CPU usati da faster-whisper |

---

## 🧠 Batch & LLM Wiki

Oltre al singolo video, l'app può elaborare **un'intera cartella** e produrre lo scaffold di una **LLM Wiki** — una base di conoscenza in markdown che un coding agent (Claude Code, OpenCode, ecc.) può poi popolare e mantenere, navigabile in [Obsidian](https://obsidian.md/).

L'idea segue il pattern [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): la conoscenza non viene riscoperta a ogni domanda, ma **compilata una volta e tenuta aggiornata**. L'app trascrive in batch e, se lo desideri, usa un **LLM locale come "cervello"** per fare un primo ingest della Wiki; un coding agent può poi rifinirla. Tutto resta 100% offline e senza costi ricorrenti.

### Come si usa

1. Nella sezione **3 │ Batch Cartella → LLM Wiki** clicca **📁 Sfoglia** e seleziona la cartella che contiene i video
2. Scegli il **Cervello Wiki** (di default lo stesso modello delle Impostazioni) e il **tipo di contenuto** nelle Impostazioni (es. *Lezione / Studio* per una cartella di lezioni)
3. Lascia attive **Genera struttura Wiki** e **🧠 Ingest automatico con LLM**
4. Clicca **🚀 Elabora Cartella + Wiki**

L'app elabora in sequenza tutti i video riconosciuti (MP4, AVI, MOV, MKV, ecc.): per ciascuno estrae l'audio, segmenta e trascrive. Il modello Whisper viene **caricato una sola volta** e riusato per tutti i file, poi viene liberato prima di caricare l'LLM (le due fasi non occupano la RAM insieme). Infine il cervello fa l'**ingest** di ogni sorgente: una pagina di sintesi, pagine concetto/entità collegate, aggiornamento di `index.md` e `log.md`.

Il progresso è salvato sia per i video sia per l'ingest: un batch interrotto **riparte** da dove si era fermato. Altri due pulsanti lavorano su una Wiki esistente:

- **🧠 Ingest Wiki esistente…** — sintetizza le sorgenti in `raw/` non ancora ingerite (utile dopo aver cambiato modello o aggiunto contenuti)
- **💬 Chiedi alla Wiki…** — il cervello seleziona le pagine più pertinenti che stanno nel contesto, risponde citando le fonti con wikilink e **archivia la risposta** in `wiki/query_<data>_<domanda>.md`

### Struttura generata

Accanto ai video viene creata una cartella di progetto:

```
_Wiki_<nome_cartella>/
│
├── raw/                    # Trascrizioni: una per video (.md, con frontmatter YAML)
│   ├── 01_Riunione_Q2.md   #   → FONTI IMMUTABILI: l'agente le legge, non le modifica
│   └── 02_Training_OBS.md
│
├── wiki/                   # Pagine generate dal cervello LLM e/o dall'agente
│   ├── index.md            #   → catalogo (Sorgenti, Entità, Concetti, Domande)
│   ├── log.md              #   → registro cronologico (append-only)
│   ├── sorgente_01_Riunione_Q2.md   # sintesi di una fonte
│   ├── concetto_<nome>.md  #   → pagine concetto, con riferimenti alle fonti
│   ├── entita_<nome>.md    #   → persone, progetti, strumenti, organizzazioni
│   └── query_<data>_<domanda>.md    # risposte archiviate
│
├── _segmenti/              # Cartelle di lavoro temporanee (ignorabili/eliminabili)
│
├── CLAUDE.md               # Schema/istruzioni per l'agente (Claude Code)
├── AGENTS.md               # Stesso schema (Codex, OpenCode, Aider, ...)
└── README_WIKI.md          # Guida rapida al workflow
```

> 💡 Le trascrizioni vengono scritte in `raw/` come `.md` solo se la spunta **Genera struttura LLM Wiki** è attiva. `CLAUDE.md` e `AGENTS.md` hanno contenuto identico: cambia solo quale agente li legge automaticamente.

### Workflow con un coding agent

1. Apri la cartella `_Wiki_<nome>/` come **vault in Obsidian**
2. Punta il tuo coding agent (es. Claude Code) alla **stessa cartella** — leggerà da sé `CLAUDE.md` / `AGENTS.md`
3. Chiedigli di **ingerire le fonti** in `raw/`: l'agente costruisce pagine, riferimenti incrociati e sintesi, navigabili in tempo reale nel grafo di Obsidian

> 🔒 **Vuoi restare interamente offline anche per l'agente?** Usa un modello locale via [Ollama](https://ollama.com/) con un agente che supporta `AGENTS.md` (OpenCode, Aider). Scegli un modello con supporto al *tool use* (es. Qwen 2.5 Coder / Qwen3) e alza la finestra di contesto (`num_ctx`) per la manutenzione multi-file della wiki.

---

## 🤖 Sintesi con LLM locale

### Modelli

Il menu **Modello** propone un catalogo di modelli open source adatti a laptop e workstation:

| Modello | Backend | Download | Contesto max | RAM consigliata |
|---|---|---|---|---|
| Qwen3 4B | Ollama / HF | ~2.6 GB | 32K | 8 GB |
| **Qwen3 8B** (default) | Ollama / HF | ~5.2 GB | 32K | 16 GB |
| Qwen3 14B | Ollama | ~9.3 GB | 32K | 32 GB |
| Qwen2.5 7B Instruct | Ollama / HF | ~4.7 GB | 32K | 16 GB |
| Llama 3.1 8B | Ollama / HF | ~4.9 GB | 128K | 16 GB |
| Llama 3.2 3B | Ollama | ~2.0 GB | 128K | 8 GB |
| Gemma 3 4B / 12B | Ollama / HF | ~3.3 / 8.1 GB | 128K | 8 / 24 GB |
| Mistral Nemo 12B | Ollama | ~7.1 GB | 128K | 24 GB |
| Phi-4 mini 3.8B | Ollama | ~2.5 GB | 128K | 8 GB |
| gpt-oss 20B | Ollama | ~14 GB | 128K | 32 GB |
| Mistral Small 3.2 24B | Ollama / HF | ~15 GB | 128K | 32 GB |
| Gemma 3 27B | Ollama / HF | ~17 GB | 128K | 32 GB |
| Qwen3 30B-A3B (MoE) | Ollama / HF | ~19 GB | 32K | 32 GB |
| Qwen3 32B · Qwen2.5 32B · DeepSeek-R1 32B | Ollama (Qwen3 anche HF) | ~20 GB | 32K / 128K | 48 GB |
| Llama 3.3 70B | Ollama | ~43 GB | 128K | 64 GB |
| gpt-oss 120B | Ollama | ~65 GB | 128K | 96 GB+ |

> 💡 **Qwen3 30B-A3B** è un *Mixture of Experts*: qualità vicina a un 30B ma attiva solo ~3B parametri per token, quindi genera a velocità simili a un modello piccolo (serve comunque la RAM per tutti i pesi). La riga informativa sotto il menu avvisa quando un modello rischia di non stare in RAM; su Apple Silicon segnala anche quando supera il limite GPU di macOS (~70% della memoria), oltre il quale una parte del modello gira su CPU.

Con **✏️ Personalizzato…** puoi usare qualsiasi modello: un nome Ollama (es. `qwen3:30b-a3b`) oppure un GGUF HuggingFace nel formato `hf:<owner>/<repo>[:<pattern>]` (es. `hf:bartowski/Qwen_Qwen3-14B-GGUF:*Q4_K_M.gguf`). I modelli già installati in Ollama compaiono automaticamente nel menu (pulsante **↻** per aggiornare).

### Parametri e finestra di contesto

| Parametro | Default | Significato |
|---|---|---|
| **Contesto** | `Auto` | Token che il modello "vede" in un colpo (prompt + testo + risposta). In *Auto* l'app sceglie la finestra più piccola che contiene l'intera trascrizione (max 32K per contenere la RAM). Valori manuali vengono limitati al massimo del modello |
| **Max output** | `4096` | Lunghezza massima della nota generata (token). Se viene raggiunto, il log lo segnala |
| **Temperatura** | `0.3` | Bassa = più fedele al testo; alzala solo per output più "creativi" |
| **Lingua nota** | Italiano | Italiano, English o la stessa lingua della registrazione |

Come viene gestito il contesto:

- Ollama di default usa un contesto piccolo e **tronca in silenzio** i testi lunghi: l'app imposta sempre `num_ctx` in modo esplicito e avvisa se l'input ha riempito la finestra.
- Il testo viene misurato in token (esatto con llama.cpp, stima conservativa con Ollama).
- Se la trascrizione non sta nel contesto scatta il **map-reduce**: la trascrizione viene divisa in blocchi sui confini di frase (con sovrapposizione), da ogni blocco si estraggono appunti dettagliati e gli appunti vengono fusi nella nota finale (fusione ricorsiva se ancora troppo lunghi).
- Il contesto resta fisso per tutta la sessione, per non far ricaricare il modello a ogni chiamata.
- Regola pratica per la RAM: la KV cache di un modello 4-8B occupa circa **0.14 MB per token** (≈ 4.5 GB a 32K), in aggiunta al peso del modello.

> 📏 Una registrazione di 1 ora corrisponde a circa 9-10 mila parole ≈ 13-16 mila token: con *Auto* sta in un solo passaggio a 32K; registrazioni più lunghe passano automaticamente al map-reduce.

### Thread, accelerazione e test di performance

| Impostazione | Dove | Default | Effetto |
|---|---|---|---|
| **Thread CPU** (Whisper) | Impostazioni → Trascrizione | `Auto (4)` | Thread usati da faster-whisper/CTranslate2. Il default della libreria è **4 thread** indipendentemente dai core disponibili: alzarlo velocizza la trascrizione su CPU |
| **Thread LLM** | Impostazioni → Sintesi | `Auto` | Thread CPU dell'LLM (`num_thread` di Ollama, `n_threads` di llama.cpp). In *Auto* Ollama usa i core ad alte prestazioni, llama.cpp metà dei core |
| **Accelerazione** | Impostazioni → Sintesi | `Auto (GPU se disponibile)` | *Solo CPU* forza l'esecuzione su CPU (`num_gpu=0` / `n_gpu_layers=0`): utile per misurare l'effetto dei thread, perché su Apple Silicon e con GPU NVIDIA l'LLM gira di norma sulla GPU |

Ogni elaborazione scrive nel log:

- un'**intestazione di sistema** (OS, CPU, core, RAM, GPU, configurazione Whisper e LLM);
- per Whisper, la **velocità di ogni segmento** (`66.8s audio in 10.5s → 6.3x tempo reale`) e nel report finale l'esecuzione (`CPU · int8 · 12 thread`) e i secondi per minuto di audio;
- per l'LLM, per ogni chiamata i token e la velocità di **lettura del prompt** e di **generazione** (tok/s), la quota del modello **su GPU** e, a fine sessione, un riepilogo;
- una riga in **`logs/performance.csv`** per ogni fase (`estrazione_audio`, `trascrizione`, `llm_nota`, `llm_ingest`, `llm_query`) con modello, thread, accelerazione, contesto, tempi, x tempo reale, tok/s e hardware: basta ripetere lo stesso contenuto cambiando un parametro e confrontare le righe.

Esempio misurato su Apple M5 Pro (15 core, 24 GB), 2.2 minuti di parlato italiano:

| Test | Configurazione | Risultato |
|---|---|---|
| Whisper `large-v3-turbo` | CPU int8, 4 thread (default) | 3.7x tempo reale (16.0 s per minuto di audio) |
| Whisper `large-v3-turbo` | CPU int8, 12 thread | **6.2x** tempo reale (9.6 s per minuto di audio) |
| LLM `qwen3:0.6b` | GPU Metal | generazione 183 tok/s |
| LLM `qwen3:0.6b` | solo CPU, 4 thread | generazione 55 tok/s |
| LLM `qwen3:0.6b` | solo CPU, 12 thread | generazione 97 tok/s |

> 💡 Oltre il numero di core fisici di solito non si guadagna; sui chip ibridi (Apple Silicon, Intel 12ª gen+) i core efficienti sono più lenti, quindi conviene provare valori diversi e confrontare il CSV.

### Tipi di nota

| Tipo | Struttura della nota |
|---|---|
| Auto | Il modello riconosce il tipo di registrazione e sceglie le sezioni |
| Standup / Daily | Aggiornamenti per persona (fatto / prossimo / blocchi), rischi, decisioni, action item |
| Riunione / Call di lavoro | Contesto, partecipanti, punti discussi, decisioni, action item (Cosa/Chi/Entro), domande aperte |
| Lezione / Studio | Concetti chiave, spiegazione, esempi ed esercizi, glossario, **domande di ripasso**, da approfondire |
| Webinar / Talk | Tesi, argomenti, dati e citazioni, key takeaway, risorse citate |
| Intervista | Profilo, temi, citazioni salienti, insight |
| Brainstorming / Idee | Obiettivo, idee per tema, valutazione pro/contro, idee scelte, parcheggio |
| Riassunto generico | Sintesi, punti principali, dettagli, conclusioni |

Ogni nota ha un frontmatter YAML (`title`, `categoria`, `source`, `date`, `model`, `tags`) pronto per Obsidian.

---

## 🖥 Compatibilità macOS / Windows

Lo stesso codice gira su entrambi i sistemi; le differenze sono gestite automaticamente:

| Aspetto | macOS | Windows |
|---|---|---|
| Trascrizione Whisper | CPU (faster-whisper non supporta Metal); il menu Device non mostra `cuda` | CPU o **GPU NVIDIA CUDA** (`requirements-gpu.txt`), fallback automatico su CPU |
| LLM | Ollama/llama.cpp usano la **GPU Metal** (memoria unificata) | Ollama usa GPU NVIDIA/AMD se presente, altrimenti CPU |
| FFmpeg | `brew install ffmpeg` (cercato anche in `/opt/homebrew/bin` e `/usr/local/bin`, utile se l'app non parte da terminale) | `ffmpeg.exe` nella cartella dell'app oppure nel PATH |
| Rilevamento hardware | `sysctl` (core Super/Performance o P/E, RAM) | Registro + API Win32 (nome CPU, core fisici, RAM), `nvidia-smi` per la GPU |
| Finestra | dimensionata sullo schermo | tiene conto dello scaling DPI (125%/150%); su schermi bassi i blocchi superiori diventano scorrevoli |
| Processi esterni | — | ffmpeg/nvidia-smi/Ollama avviati senza finestre di console |
| Log e impostazioni | `logs/` accanto all'app, `~/.from_video_to_wiki/` | `logs\` accanto all'app (o all'eseguibile PyInstaller), `%USERPROFILE%\.from_video_to_wiki\` |

> ℹ️ `torch` non è più necessario: il rilevamento CUDA usa `ctranslate2` (già installato con faster-whisper).

---

## 📁 Struttura output

Ogni video genera una cartella dedicata con il proprio nome:

```
cartella_video/
├── intervista.mp4
├── intervista/
│   ├── intervista_trascrizione.txt
│   └── intervista_nota.md
├── riunione.mp4
└── riunione/
    ├── riunione_trascrizione.txt
    └── riunione_nota.md
```

---

## ⚡ Performance reali e proiezioni

### Benchmark misurati

Due dati misurati **end-to-end** (caricamento modello + estrazione audio + segmentazione + trascrizione + generazione file) con `large-v3-turbo`:

- **CPU (int8)**: circa **20 sec/MB** di video sorgente
- **GPU NVIDIA RTX 4060 (float16)**: circa **0.09 sec/MB** — ovvero **~1.5 minuti per GB** di video

Sulla GPU il fattore di velocità rispetto alla CPU supera le **~200x** grazie al batching del modello turbo. La stessa cartella di video che la CPU macina in una notte, la GPU la chiude in pochi minuti.

> ⚠️ **Nota sull'unità di misura**: `sec/MB` dipende dal bitrate del video — un GB di registrazione ad alta risoluzione contiene meno minuti di audio (e quindi si trascrive più in fretta "per MB") di un GB a basso bitrate. I valori sopra sono indicativi e variano in base a CPU/GPU, RAM e contenuto.

### Rapporti di velocità stimati per modello (CPU, int8)

| Modello | sec/MB stimati | Rapporto vs turbo |
|---|---|---|
| `tiny` | ~3 | ~7x più veloce |
| `small` | ~7 | ~3x più veloce |
| `medium` | ~14 | ~1.4x più veloce |
| `large-v3-turbo` | **~20** | **baseline (misurato)** |
| `large-v3` | ~50 | ~2.5x più lento |

### Proiezioni tempi E2E per dimensione video e modello (CPU)

| Dimensione video | `tiny` (~3 s/MB) | `small` (~7 s/MB) | `medium` (~14 s/MB) | `large-v3-turbo` (~20 s/MB) ⭐ | `large-v3` (~50 s/MB) |
|---|---|---|---|---|---|
| **50 MB** | ~2 min | ~6 min | ~12 min | **~17 min** | ~42 min |
| **100 MB** | ~5 min | ~12 min | ~23 min | **~33 min** | ~1h 23min |
| **200 MB** | ~10 min | ~23 min | ~47 min | **~1h 7min** | ~2h 47min |
| **500 MB** | ~25 min | ~58 min | ~1h 57min | **~2h 47min** | ~6h 57min |
| **1 GB** | ~51 min | ~2h | ~3h 59min | **~5h 41min** | ~14h 13min |
| **2 GB** | ~1h 42min | ~3h 59min | ~7h 57min | **~11h 22min** | ~28h 26min |
| **5 GB** | ~4h 16min | ~9h 58min | ~19h 53min | **~28h 24min** | ~71h (≈3 giorni) |

> ⭐ La colonna `large-v3-turbo` su CPU è basata su benchmark reali. Le altre sono **proiezioni stimate** in base ai rapporti di velocità tipici tra modelli su CPU int8.

> ⚠️ **Nota qualità**: `tiny` e `small` sono molto più veloci ma la qualità in italiano degrada significativamente. `medium` è un buon compromesso se la RAM è limitata. `large-v3` offre qualità identica al turbo ma impiega 2.5x più tempo — **il turbo resta la scelta migliore** in quasi tutti gli scenari.

> 💡 **Consiglio**: per video grandi su CPU, lancia la trascrizione di notte. Con una **GPU NVIDIA** la differenza è enorme (vedi benchmark misurato sopra). Se lavori solo su CPU, il batch notturno resta la strategia migliore.

### Confronto CPU vs GPU vs Apple Silicon

| Setup | sec/MB | 200 MB | 1 GB |
|---|---|---|---|
| CPU (i7/Ryzen 7, int8) | ~20 | ~1h 7min | ~5h 41min |
| CPU (i9/Ryzen 9, int8) | ~14 | ~47 min | ~4h |
| MacBook Pro M3 Pro (int8) | ~12 | ~40 min | ~3h 25min |
| **MacBook M5 Pro 24 GB (int8, misurato)** | **~7** | **~23 min** | **~2h** |
| **GPU (RTX 4060, float16 — misurato)** | **~0.09** | **~18 sec** | **~1.5 min** |

> ⭐ La riga **RTX 4060** è un **benchmark reale end-to-end** con `large-v3-turbo`. GPU di fascia superiore (4070/4090) sono attese ancora più rapide. Le righe CPU/Apple sono indicative.

> 💡 **Nota su macOS**: faster-whisper su Apple Silicon gira su CPU (non sfrutta Metal/GPU nativamente), ma le performance dei chip M-series sono eccellenti grazie alla bandwidth di memoria unificata. Per sfruttare appieno la GPU Apple, valuta alternative come [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) che supportano l'accelerazione Metal nativa.

---

## 🛠 Scelta del modello

| Modello | RAM (int8) | VRAM (float16) | Velocità | Qualità IT/EN |
|---|---|---|---|---|
| `tiny` | ~0.5 GB | ~1 GB | ⚡⚡⚡⚡⚡ | ⭐⭐ |
| `small` | ~1 GB | ~2 GB | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| `medium` | ~2 GB | ~4 GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `large-v3-turbo` | ~2.5 GB | ~5 GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| `large-v3` | ~4 GB | ~8 GB | ⚡ | ⭐⭐⭐⭐⭐ |

> **Consiglio**: usa `large-v3-turbo` come default. Scala a `medium` se la RAM/VRAM non basta. Evita `tiny` e `base` per l'italiano.

---

## 🔧 Troubleshooting

| Problema | Soluzione |
|---|---|
| `FFmpeg non trovato` | Verifica che `ffmpeg -version` funzioni nel terminale. Se no, reinstallalo e riavvia il terminale. |
| `faster-whisper non installato` | Esegui `pip install faster-whisper` con il virtual environment attivo |
| Trascrizione lenta su CPU | Prova un modello più piccolo (`medium`, `small`) oppure passa a GPU con `--device cuda` (vedi sotto) |
| Il modello non si scarica | Verifica la connessione internet. La cache è in `~/.cache/huggingface/` |
| `Could not load library cudnn_ops_infer64_8.dll` o `cublas64_12.dll` non trovata | Installa le librerie GPU: `pip install --timeout 300 -r requirements-gpu.txt`. L'app rileva automaticamente i percorsi DLL e li aggiunge al PATH. |
| `CUDA out of memory` o errori CUDA generici | Scegli un modello più piccolo (es. `medium` invece di `large-v3-turbo`). Se la VRAM è scarsa (<5 GB), passa a `cpu`. L'app fa comunque **fallback automatico su CPU** se CUDA fallisce al caricamento del modello. |
| Driver NVIDIA mancanti | Anche con i pacchetti pip installati, serve il driver NVIDIA a livello di sistema. Aggiornalo da [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx). |
| Pip si blocca su `nvidia-cublas-cu12` (timeout) | Sono pacchetti grandi (~500 MB ciascuno). Alza il timeout: `pip install --timeout 300 -r requirements-gpu.txt` |
| L'app si chiudeva dopo la trascrizione GPU | Risolto: ora c'è un cleanup deterministico della cache CUDA prima del messagebox finale, l'app resta aperta. |
| Allucinazioni nel testo | Il filtro VAD è attivo di default. Se persistono, prova `--lingua it` per forzare la lingua |
| La trascrizione non riprende dal punto giusto | Verifica che `_progresso.json` sia presente nella cartella dei segmenti. Se corrotto, eliminalo per ripartire da zero |
| **macOS**: la finestra dell'app è bianca / vuota | Stai usando un Python con Tk 8.5 (tipicamente quello di Apple in `/Library/Developer/CommandLineTools`). Verifica con `python -c "import tkinter; print(tkinter.TkVersion)"`: serve ≥ 8.6. Ricrea il venv con Python da Homebrew (`python@3.12` + `python-tk@3.12`) o da python.org. |
| **macOS**: `FFmpeg non trovato` | `ffmpeg.exe` è per Windows. Installa `brew install ffmpeg` e riavvia il terminale. |
| `open() got an unexpected keyword argument 'metadata_errors'` su ogni segmento | Incompatibilità tra `faster-whisper` e una versione recente di `av` (PyAV 19). Esegui `pip install "av<17"`. Fino alla correzione l'app può mostrare "Completato" con trascrizione vuota (0 segmenti trascritti). |
| I segmenti `.wav` non vengono eliminati a fine trascrizione | La pulizia avviene solo se almeno un segmento è stato trascritto con successo e la trascrizione non è stata interrotta. Se nel report vedi "Segmenti trascritti: 0/N" c'è un errore a monte (controlla il file in `logs/`): i `.wav` restano apposta, così puoi ritrascrivere con **📂 Trascrivi da Cartella...** senza rifare lo splitting. |
| Il batch non rigenera i `.md` in `raw/` | Assicurati che la spunta **Genera struttura Wiki** sia attiva prima di avviare il batch |
| `Ollama non raggiungibile` | Installa Ollama da [ollama.com/download](https://ollama.com/download) e avvialo (l'app prova anche a lanciare `ollama serve`). Se usi un host diverso imposta la variabile `OLLAMA_HOST`. |
| La nota è incompleta o il log segnala "Risposta troncata" | Aumenta **Max output** (es. 8192). |
| Il log segnala "L'input ha riempito il contesto" | Aumenta **Contesto** o scegli un modello con contesto più ampio (Llama 3.1, Gemma 3, Mistral Nemo: 128K). |
| La sintesi LLM è lenta o la RAM non basta | Usa un modello più piccolo (Qwen3 4B, Llama 3.2 3B, Phi-4 mini) e/o un contesto più piccolo: la RAM cresce con il contesto (≈ 0.14 MB/token di KV cache). |
| Modelli `· HuggingFace` non disponibili | Esegui `pip install -r requirements-hf.txt`. Per repo "gated" serve `huggingface-cli login`. |
| Note con nomi, numeri o riferimenti inventati | Abbassa la **Temperatura** (0.1-0.3) e usa un modello più grande: i modelli sotto i 3B tendono a copiare le istruzioni del template e ad allucinare. |

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza [MIT](LICENSE).
