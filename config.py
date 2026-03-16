
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────
USE_LOCAL_EMBEDDINGS = os.getenv('USE_LOCAL_EMBEDDINGS', 'True') == 'True'

# Storage / Container
STORAGE_CONN_STR          = str(os.getenv('STORAGE_CONN_STR', 'DefaultEndpointsProtocol=...'))
CONTAINER_NAME            = str(os.getenv('CONTAINER_NAME', 'knowledge-base'))

# FAISS index paths
LOCAL_FAISS_INDEX_PATH    = str(os.getenv('LOCAL_FAISS_INDEX_PATH', 'pipeline_demo_faiss.index'))
LOCAL_FAISS_METADATA_PATH = str(os.getenv('LOCAL_FAISS_METADATA_PATH', 'pipeline_demo_faiss_metadata.json'))

# Chunking
CHUNK_SIZE                = int(os.getenv('CHUNK_SIZE', '1200'))
CHUNK_OVERLAP             = int(os.getenv('CHUNK_OVERLAP', '100'))
MIN_CHUNK_SIZE            = int(os.getenv('MIN_CHUNK_SIZE', '200'))

# Search
INDEX_NAME                = str(os.getenv('INDEX_NAME', 'knowledge-base-demo'))
RRF_K                     = int(os.getenv('RRF_K', '60'))

# OCR / Tesseract
TESSERACT_LANGUAGE        = str(os.getenv('TESSERACT_LANGUAGE', 'eng'))
OCR_THRESHOLD             = int(os.getenv('OCR_THRESHOLD', '100'))

# Logging
VIEW_INTERNAL_LOG         = os.getenv('VIEW_INTERNAL_LOG', 'False')

# Embedding / OpenAI
BATCH_SIZE                = int(os.getenv('BATCH_SIZE', '64'))
RETRY_DELAY_SECONDS       = float(os.getenv('RETRY_DELAY_SECONDS', '1.0'))
OPENAI_ENDPOINT           = os.getenv('OPENAI_ENDPOINT', '')
OPENAI_API_KEY            = os.getenv('OPENAI_API_KEY', '')
OPENAI_DEPLOYMENT         = os.getenv('OPENAI_DEPLOYMENT', 'text-embedding-ada-002')
DEPLOYMENT_NAME           = os.getenv('DEPLOYMENT_NAME', '')
