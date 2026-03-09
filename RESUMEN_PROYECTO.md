# 📄 RAG Document Intelligence System — Resumen del Proyecto

## ¿Qué es este proyecto?

Un sistema de **pregunta-respuesta sobre documentos** que funciona 100% en local. El usuario sube PDFs o DOCX, hace preguntas en lenguaje natural, y recibe respuestas fundamentadas con citas de los documentos originales.

Todo se levanta con un solo comando: `docker compose up --build -d`

---

## Arquitectura General

```
┌─────────────────────┐
│   Usuario/Browser   │
│   (localhost:8501)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐         ┌──────────────────┐
│   Streamlit UI      │────────▶│   Flask API      │
│   (frontend/)       │         │   (flask_app/)   │
│   - Upload docs     │         │   Puerto 5000    │
│   - Chat Q&A        │         │                  │
│   - Listar/borrar   │         │   5 endpoints:   │
└─────────────────────┘         │   POST /documents│
                                │   GET  /documents│
                                │   DEL  /documents│
                                │   POST /query    │
                                │   GET  /health   │
                                └──┬───┬───┬───────┘
                                   │   │   │
                    ┌──────────────┘   │   └────────────────┐
                    ▼                  ▼                    ▼
           ┌──────────────┐  ┌──────────────┐    ┌──────────────┐
           │    MinIO      │  │   ChromaDB   │    │  llama.cpp   │
           │  Puerto 9000  │  │  Puerto 8000 │    │  Puerto 8080 │
           │               │  │              │    │              │
           │  Guarda los   │  │  Almacena    │    │  Genera      │
           │  archivos     │  │  embeddings  │    │  respuestas  │
           │  originales   │  │  vectoriales │    │  con el LLM  │
           └──────────────┘  └──────────────┘    └──────────────┘
```

---

## El Pipeline RAG (5 fases)

Cuando un usuario **sube un documento**:

```
PDF/DOCX ──▶ 1.PARSEAR ──▶ 2.CHUNKEAR ──▶ 3.INDEXAR
              (pdfplumber)   (512 palabras)  (embeddings →
              (python-docx)  (con overlap)    ChromaDB)
```

Cuando un usuario **hace una pregunta**:

```
Pregunta ──▶ 4.RECUPERAR ──▶ 5.GENERAR
              (buscar chunks   (montar prompt
               similares en     con contexto →
               ChromaDB)        llama.cpp →
                                respuesta citada)
```

---

## Estructura de Archivos

```
Assigment 3/
│
├── docker-compose.yml          ← Orquesta los 5 servicios Docker
│
├── flask_app/                  ← EL CORAZÓN DEL SISTEMA
│   ├── Dockerfile              ← Imagen Docker del API
│   ├── requirements.txt        ← Dependencias Python
│   ├── app.py                  ← 🎯 Flask API (5 endpoints)
│   ├── config.py               ← Variables de entorno
│   ├── ingestion.py            ← Parseo PDF/DOCX + subida a MinIO
│   ├── chunking.py             ← Dividir texto en trozos (2 estrategias)
│   ├── retrieval.py            ← Embeddings + búsqueda en ChromaDB
│   └── llm_client.py           ← Cliente HTTP para llama.cpp
│
├── frontend/                   ← BONUS +5%
│   ├── Dockerfile
│   └── app.py                  ← Streamlit (upload, lista, chat)
│
├── eval/                       ← EVALUACIÓN
│   ├── eval_dataset.json       ← 15 preguntas anotadas
│   ├── run_eval.py             ← Calcula Hit Rate, MRR, Precision
│   └── create_eval_docs.py     ← Genera documentos de prueba
│
├── models/                     ← Modelo LLM (NO se sube a Git)
│   └── .gitkeep
│
├── results/                    ← Resultados de evaluación
├── report/                     ← Technical report (PDF)
├── README.md                   ← Documentación completa
└── .gitignore                  ← Excluye modelos y datos
```

---

## Los 5 Servicios Docker

| Servicio | Imagen | Puerto | Función |
|----------|--------|--------|---------|
| **MinIO** | `minio/minio` | 9000 | Almacén de archivos originales (como un S3 local) |
| **ChromaDB** | `chromadb/chroma` | 8000 | Base de datos vectorial para los chunks |
| **llama.cpp** | `ghcr.io/ggml-org/llama.cpp:server` | 8080 | Servidor del modelo LLM cuantizado |
| **Flask API** | Build local | 5000 | API REST que orquesta todo el pipeline |
| **Streamlit** | Build local | 8501 | Interfaz web para el usuario |

Todos usan `network_mode: "host"` (comparten la red del PC).  
Los datos se guardan en `/home/alumno/Desktop/datos/NLP/docker_assigment3/`.

---

## Los 5 Endpoints del API

| Método | Ruta | Qué hace |
|--------|------|----------|
| `POST /documents` | Sube un PDF/DOCX → parsea → chunkea → indexa → devuelve ID |
| `GET /documents` | Lista todos los documentos subidos |
| `DELETE /documents/{id}` | Borra documento de MinIO + ChromaDB + metadata |
| `POST /query` | Recibe pregunta → recupera chunks → genera respuesta con LLM |
| `GET /health` | Verifica que MinIO, ChromaDB, LLM y SQLite están OK |

---

## Tecnologías Clave

| Componente | Tecnología | Por qué |
|-----------|-----------|---------|
| LLM | TinyLlama 1.1B (GGUF Q4) | Ligero (~670MB), corre en CPU |
| Embeddings | all-MiniLM-L6-v2 | Modelo estándar para embeddings, 80MB |
| Vector DB | ChromaDB | Fácil de usar, búsqueda por coseno |
| Object Store | MinIO | Compatible con S3, para archivos raw |
| PDF parser | pdfplumber | Extrae texto de PDFs (detecta scanned) |
| DOCX parser | python-docx | Extrae texto de Word |
| Chunking | Fixed-size + Recursive | 2 estrategias implementadas |
| Metadata | SQLite | Almacena info de documentos (ligero) |

---

## Cómo Levantar el Proyecto

```bash
# 1. Descargar modelo (solo la primera vez, ~670MB)
wget -O /ruta/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 2. Arrancar todo
docker compose up --build -d

# 3. Verificar
curl http://localhost:5000/health

# 4. Abrir frontend
# → http://localhost:8501
```

---

## Flujo de Datos Detallado

### Al subir un documento:
1. El archivo llega a Flask vía `POST /documents`
2. Se guarda el archivo raw en **MinIO** (bucket `documents`)
3. Se parsea el texto con `pdfplumber` (PDF) o `python-docx` (DOCX)
4. Si es un PDF escaneado (sin texto), devuelve error 422
5. El texto se divide en **chunks** de ~512 palabras con 50 de overlap
6. Cada chunk se convierte en un **vector** con `all-MiniLM-L6-v2`
7. Los vectores se guardan en **ChromaDB**
8. Los metadatos del documento se guardan en **SQLite**

### Al hacer una pregunta:
1. La pregunta llega a Flask vía `POST /query`
2. Se convierte la pregunta en un **vector** con el mismo modelo
3. Se buscan los **top-k chunks más similares** en ChromaDB (coseno)
4. Se construye un **prompt** con el contexto de los chunks
5. Se envía al **LLM** (llama.cpp) que genera una respuesta citada
6. Se devuelve la respuesta + los chunks fuente al usuario
