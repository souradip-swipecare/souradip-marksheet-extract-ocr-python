# Souradip Marksheet Extraction API

> A simple API that extracts data from marksheets using OCR and AI. Just upload image/pdf and get structured JSON!

Made by: **Souradip Biswas**

---

## Folder Structure

```
python_api_extract/
│
├── app/                          # Main application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   │
│   ├── api/                      # API routes and security
│   │   ├── __init__.py
│   │   ├── routes.py             # Main router
│   │   ├── security.py           # API key authentication
│   │   └── v1/
│   │       └── routes.py         # v1 API endpoints (/extract, /batch, /health)
│   │
│   ├── core/                     # Configuration stuff
│   │   ├── __init__.py
│   │   ├── config.py             # Settings from .env file
│   │   └── logging.py            # Logger setup
│   │
│   ├── models/                   # Pydantic models
│   │   ├── __init__.py
│   │   └── schemas.py            # Request/Response schemas
│   │
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   ├── extraction.py         # LLM extraction (Gemini/OpenAI)
│   │   ├── ocr_service.py        # Tesseract OCR wrapper
│   │   └── prompts.py            # AI prompts for extraction
│   │
│   ├── static/
│   │   └── demo.html             # Frontend demo page
│   │
│   └── utils/                    # Helper functions
│       ├── __init__.py
│       ├── exceptions.py         # Custom exceptions
│       └── file_processor.py     # PDF/Image processing
│
├── extract/                      # Output folder (auto created)
│   ├── ocr_*.txt                 # Saved OCR text
│   └── output_*.json             # Saved extraction results
│
├── logs/                         # Log files
│   └── app.log
│
├── testdata/                     # Sample marksheets for testing
│   ├── marks_sheet_1.webp
│   ├── marks_sheet_2.webp
│   ├── shashank.jpg
│   └── ... more test files
│
├── tests/                        # Unit tests
│   └── test_api.py
│
├── .env                          # Environment variables (API keys etc)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker config
├── docker-compose.yml            # Docker compose
└── README.md                     # This file!
```

---

## How It Works (Workflow)

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER UPLOADS FILE                        │
│                    (image: jpg/png/webp or PDF)                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FILE PROCESSOR                              │
│  • Check file type and size                                      │
│  • If PDF with text → extract text directly (fast!)              │
│  • If scanned PDF/image → convert to images                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TESSERACT OCR                               │
│  • Run 17 OCR passes in PARALLEL (8 workers)                     │
│  • Pick best result based on text quality + confidence           │
│  • Calculate weighted confidence score                           │
│  • ~4-5 seconds (vs 15 seconds sequential)                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │  OCR Confidence >= 60% │(this can be set by the user in setting folder)
                    └───────────────────────┘
                          │           │
                         YES          NO
                          │           │
         ▼ ---------------|           ▼
┌─────────────────────┐   ┌─────────────────────┐
│   OCR + LLM Mode    │   │  Direct Vision Mode │
│                     │   │                     │
│ Send OCR text to    │   │ Send raw image to   │
│ Gemini for parsing  │   │ Gemini Vision API   │
│ (faster & cheaper)  │   │ (better accuracy)   │
└─────────────────────┘   └─────────────────────┘
           │                  │
           └─────────---------┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GEMINI AI / OpenAI                          │
│  • Understand the marksheet structure                            │
│  • Extract all fields with confidence scores                     │
│  • Return structured JSON                                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RESPONSE                                    │
│  • Structured JSON with all extracted data                       │
│  • Confidence scores for each field                              │
│  • Processing time and method used                               │
│  • Save output to extract/ folder                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Multi-Pass OCR (How It Works)

We run Tesseract OCR **17 times in PARALLEL** with different image preprocessing techniques and pick the best result. This handles various image qualities, lighting, and backgrounds.

### Parallel Processing (8 Workers)

All 17 OCR passes run **simultaneously** using Python's `ThreadPoolExecutor` with 8 workers:

```
BEFORE (Sequential):
Pass 1 → Pass 2 → Pass 3 → ... → Pass 17
[============= 15 seconds =============]

AFTER (Parallel with 8 workers):
Pass 1  ──┐
Pass 2  ──┤
Pass 3  ──┤
Pass 4  ──┼──→ All run at the same time!
...       │
Pass 17 ──┘
[=== 4-5 seconds ===]
```

**Result: ~3-4x faster OCR processing!**

### Preprocessing Passes

| Pass | Technique | Purpose |
|------|-----------|---------|
| 1-2 | Grayscale + CLAHE + Denoise | Basic enhancement |
| 3-4 | Scaled image (2x or 1.5x) | For small/low-res images |
| 5 | Bilateral filter | Edge-preserving smoothing |
| 6 | Otsu's binarization | Automatic threshold |
| 7-9 | Adaptive threshold (blocks: 11, 15, 21) | Handle uneven lighting |
| 10-11 | Morphological close + open | Clean up text edges |
| 12 | Sharpened (unsharp masking) | Improve text clarity |
| 13-16 | Different PSM modes (4, 6, 11, 12) | Different text layouts |
| 17 | Inverted image | For dark backgrounds |

### PSM Modes Explained

| Mode | Name | When Used |
|------|------|-----------|
| PSM 3 | Auto page segmentation | Default, works for most |
| PSM 4 | Single column | Vertical text layouts |
| PSM 6 | Uniform block | Tables, structured text |
| PSM 11 | Sparse text | Find scattered text |
| PSM 12 | Sparse text + OSD | With orientation detection |

### Best Result Selection

After all passes, we score each result:

```
Score = (alpha_count * 0.4) + (confidence * 100 * 0.4) + (word_count * 0.2)
```

| Factor | Weight | Why |
|--------|--------|-----|
| Alpha count | 40% | More real characters = less garbage |
| Confidence | 40% | Higher confidence = better quality |
| Word count | 20% | More words = more complete extraction |

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/souradip-swipecare/souradip-marksheet-extract-ocr-python.git
cd python_api_extract

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

**Mac:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Setup environment variables

Create `.env` file:
```env
# App Settings
APP_NAME="Marksheet Extraction API"
DEBUG=false

# Gemini API Key (get from https://aistudio.google.com/apikey)
GOOGLE_API_KEY=your_gemini_api_key_here

# Model settings
DEFAULT_LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash

# OCR settings
OCR_CONFIDENCE_THRESHOLD=0.60
SAVE_OCR_TEXT=true
```

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open in browser

- **Demo Page:** http://localhost:8000/
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/v1/health

---

## API Endpoints

### Extract Single Marksheet

```
POST /api/v1/extract
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | Marksheet image or PDF |
| model | string | No | LLM model (default: "gemini") |
| apikey | string | No | Your Gemini API key (optional) |

**Example with curl:**
```bash
# Using server's default API key
curl -X POST "http://localhost:8000/api/v1/extract?model=gemini" \
  -F "file=@marksheet.jpg"

# Using your own API key
curl -X POST "http://localhost:8000/api/v1/extract?model=gemini&apikey=YOUR_KEY" \
  -F "file=@marksheet.jpg"
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "candidate": {
      "name": {"value": "Rahul Kumar", "confidence": 0.95},
      "roll_number": {"value": "12345678", "confidence": 0.92},
      "father_name": {"value": "Suresh Kumar", "confidence": 0.88}
    },
    "subjects": [
      {
        "subject_name": {"value": "Mathematics", "confidence": 0.94},
        "obtained_marks": {"value": 85, "confidence": 0.91},
        "max_marks": {"value": 100, "confidence": 0.95}
      }
    ],
    "result": {
      "total_marks": {"value": 425, "confidence": 0.90},
      "percentage": {"value": 85.0, "confidence": 0.88},
      "result_status": {"value": "PASS", "confidence": 0.95}
    },
    "extraction_confidence": 0.89
  },
  "processing_time_ms": 3245.67,
  "extraction_method": "ocr_llm"
}
```

### Batch Extract (Multiple Files)

```
POST /api/v1/extract/batch
```

Upload up to 10 files at once.

### Health Check

```
GET /api/v1/health
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_api.py::TestMarksheetExtraction -v

# Run with output
pytest tests/ -v -s
```

---

## Docker

```bash
# Build and run
docker-compose up --build

# Or just build
docker build -t marksheet-api .
docker run -p 8000:8000 --env-file .env marksheet-api
```

---

## Extraction Methods

The API automatically picks the best method:

| Method | When Used | Speed | Accuracy |
|--------|-----------|-------|----------|
| `text_pdf_llm` | PDF has selectable text | Fastest | High |
| `ocr_llm` | OCR confidence >= 60% | Fast | Good |
| `direct_llm_vision` | OCR confidence < 60% | Slower | Best |

---

## Processing Time Breakdown

Here's where the ~30-40 seconds goes:

| Step | Time | Description |
|------|------|-------------|
| File Upload & Validation | ~100ms | Read file, check type/size |
| Image Preprocessing | ~500ms | Convert PDF to images, resize |
| **Multi-pass OCR** | **5-15 sec** | Run 15+ OCR passes with different preprocessing |
| **LLM API Call** | **15-30 sec** | Send to Gemini, wait for response |
| JSON Parsing | ~10ms | Parse LLM response |
| Response Building | ~50ms | Build final response |

**Why so slow?**
- LLM Vision models are slow (network + processing)
- Multi-pass OCR ensures best quality but takes time
- Gemini processes each image pixel-by-pixel

**How to speed up:**
1. Use `text_pdf_llm` method (upload text-based PDFs, not scanned)
2. Upload smaller/lower resolution images
3. Use OCR mode (confidence >= 60%) instead of direct vision

---

## Configuration Options

All settings in `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | - | Your Gemini API key |
| `GEMINI_MODEL` | gemini-2.5-flash | Gemini model to use |
| `OCR_CONFIDENCE_THRESHOLD` | 0.60 | Min confidence for OCR mode |
| `MAX_FILE_SIZE_MB` | 10 | Max upload size |
| `SAVE_OCR_TEXT` | true | Save OCR output to files |
| `DEBUG` | false | Enable debug mode |
| `RATE_LIMIT_REQUESTS` | 100 | Max requests per minute |

---

## How Confidence Score is Calculated

The API returns confidence scores at multiple levels to help you understand the reliability of extracted data.

### 1. OCR Confidence (Tesseract)

Tesseract OCR gives a confidence score (0-100) for each word it detects. We calculate the **average OCR confidence** like this:

```
Formula:
- For each word, Tesseract gives confidence (0-100%)
- High confidence words (>=70%) get weight = 1.5
- Low confidence words (<70%) get weight = 1.0
- Average = sum(confidence * weight) / sum(weights)
- If >70% words are high confidence, we give 10% bonus (max 1.0)
```

**Example:**
```
Words detected: ["RAHUL" (conf: 95%), "KUMAR" (conf: 88%), "x7z" (conf: 30%)]

Calculation:
- RAHUL: 0.95 * 1.5 = 1.425 (high conf, weight 1.5)
- KUMAR: 0.88 * 1.5 = 1.32  (high conf, weight 1.5)
- x7z:   0.30 * 1.0 = 0.30  (low conf, weight 1.0)

Average = (1.425 + 1.32 + 0.30) / (1.5 + 1.5 + 1.0) = 0.76
```

### 2. Field-Level Confidence (LLM)

Gemini AI returns confidence for each extracted field based on:
- How clearly visible the text is
- How well it matches expected patterns (dates, numbers etc)
- Consistency with other fields

| Confidence Range | Meaning |
|-----------------|---------|
| 0.8 - 1.0 | Very High - Field is perfectly clear |
| 0.6 - 0.8 | High - Field is readable, minor ambiguity |
| 0.3 - 0.6 | Medium - Some uncertainty exists |
| 0.0 - 0.3 | Low - Field partially visible or inferred |

### 3. Overall Extraction Confidence

The `extraction_confidence` in response is calculated as weighted average:

```
Important fields (higher weight = 2.0):
- name, roll_number, result_status

Medium importance (weight = 1.5):
- obtained_marks, total_marks, subject_name

Other fields (weight = 1.0):
- All other extracted fields
```

### Why OCR Confidence Matters

| OCR Confidence | What Happens |
|---------------|--------------|
| >= 60% | Use OCR text + send to Gemini (fast) |
| < 60% | Send raw image to Gemini Vision (accurate but slower) |

This threshold (60%) can be changed via `OCR_CONFIDENCE_THRESHOLD` in `.env`

---

## Troubleshooting

### "Tesseract not found"
Make sure tesseract is installed and in PATH:
```bash
tesseract --version
```

### "Google API key not configured"
Either:
1. Set `GOOGLE_API_KEY` in `.env` file
2. Or pass `apikey` parameter in request

### "Empty subjects array"
- Make sure image is clear and readable
- Try uploading higher resolution image
- Check if OCR is working: look at `extract/ocr_*.txt` files

---

## Can be improved

- [ ] Add support for Openai GPT-4 Vision
- [ ] Add more test cases
- [ ] Support for handwritten marksheets
- [ ] Using sinario background job
- [ ] Add caching for repeated requests
- [x] Rate limiting (implemented via slowapi)

---
