# AI Code Plagiarism Detector

An AI-powered full-stack web application to detect code plagiarism. Upload or paste code files and get instant similarity analysis using Cosine Similarity, Token Matching, and Longest Common Subsequence (LCS) algorithms.

## 🏗️ Project Structure

```
ai-plagiarism-detector/
│
├── client/                          # React Frontend (Vite)
│   ├── src/
│   │   ├── components/              # Reusable UI components
│   │   │   ├── Navbar.jsx           # Top navigation bar
│   │   │   ├── FileUploader.jsx     # Drag & drop file upload
│   │   │   ├── CodeEditor.jsx       # Code paste text areas
│   │   │   ├── SimilarityCard.jsx   # Result card with score bars
│   │   │   └── MatchViewer.jsx      # Matched code sections viewer
│   │   ├── pages/                   # Route-level page components
│   │   │   ├── UploadPage.jsx       # Upload/paste files for comparison
│   │   │   ├── ResultsPage.jsx      # View comparison results
│   │   │   └── DashboardPage.jsx    # View past submissions
│   │   ├── services/
│   │   │   └── api.js               # Axios API communication layer
│   │   ├── App.jsx                  # Router setup
│   │   ├── main.jsx                 # Entry point
│   │   └── index.css                # Global styles (dark theme)
│   ├── package.json
│   └── vite.config.js               # Vite config with API proxy
│
├── server/                          # Node.js + Express Backend
│   ├── src/
│   │   ├── config/
│   │   │   └── db.js                # MongoDB connection
│   │   ├── controllers/
│   │   │   └── compareController.js # Request handlers
│   │   ├── middleware/
│   │   │   ├── upload.js            # Multer file upload config
│   │   │   └── errorHandler.js      # Global error handler
│   │   ├── models/
│   │   │   ├── Submission.js        # Schema for uploaded files
│   │   │   └── Result.js            # Schema for comparison results
│   │   ├── routes/
│   │   │   └── compareRoutes.js     # REST API routes
│   │   ├── services/
│   │   │   ├── similarityService.js # Cosine, Token, LCS algorithms
│   │   │   └── comparisonService.js # Comparison pipeline orchestrator
│   │   ├── utils/
│   │   │   ├── preprocessor.js      # Code cleaning & normalization
│   │   │   └── tokenizer.js         # Code tokenization
│   │   └── app.js                   # Express app configuration
│   ├── server.js                    # Entry point
│   ├── .env                         # Environment variables
│   └── package.json
│
├── ai-service/                      # Python Flask Microservice
│   ├── app.py                       # Flask app with /analyze endpoint
│   └── requirements.txt             # Python dependencies
│
├── .env.example                     # Environment template
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** v18+ and npm
- **MongoDB** running locally (or MongoDB Atlas URI)
- **Python** 3.9+ (optional, for AI microservice)

### 1. Clone and navigate

```bash
git clone <your-repo-url>
cd ai-plagiarism-detector
```

### 2. Start the Backend

```bash
cd server
npm install
# Edit .env if needed (default: MongoDB on localhost:27017)
npm run dev
```
Server runs on http://localhost:5000

### 3. Start the Frontend

```bash
cd client
npm install
npm run dev
```
Frontend runs on http://localhost:3000

### 4. Start the Python Microservice (Optional)

```bash
cd ai-service
pip install -r requirements.txt
python app.py
```
Python service runs on http://localhost:8000

## 📡 API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| POST   | `/api/compare`    | Upload/paste files and compare     |
| GET    | `/api/results`    | Get all past comparison results    |
| GET    | `/api/results/:id`| Get a specific result by ID        |
| GET    | `/api/health`     | Health check                       |

### POST /api/compare

**Option A: File Upload (multipart/form-data)**
```bash
curl -X POST http://localhost:5000/api/compare \
  -F "files=@file1.js" \
  -F "files=@file2.js"
```

**Option B: Paste Code (JSON)**
```json
{
  "files": [
    { "name": "file1.js", "content": "function add(a, b) { return a + b; }" },
    { "name": "file2.js", "content": "function sum(x, y) { return x + y; }" }
  ]
}
```

## 🧠 Algorithms

| Algorithm         | Description                                    | Weight |
|-------------------|------------------------------------------------|--------|
| Cosine Similarity | TF-IDF vector dot product                      | 40%    |
| Token Matching    | Jaccard similarity on unique token sets         | 25%    |
| LCS               | Longest Common Subsequence on code lines        | 35%    |

### Verdict Thresholds
- **Low**: < 30% overall similarity
- **Medium**: 30% – 60% overall similarity
- **High**: > 60% overall similarity

## ⚙️ Environment Variables

```env
PORT=5000
MONGO_URI=mongodb://localhost:27017/plagiarism_detector
PYTHON_SERVICE_URL=http://localhost:8000
```

## 🛠️ Tech Stack

- **Frontend**: React 19, Vite, React Router, Axios, react-dropzone
- **Backend**: Node.js, Express, Mongoose, Multer
- **Database**: MongoDB
- **AI Service**: Python, Flask, scikit-learn, difflib
