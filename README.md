# Syntherview 🎙️💻

**A Voice-First AI Mock Interview Platform**

Syntherview is a full-stack application designed to simulate technical interviews. It accepts a candidate's resume and job description, generates a tailored interview plan, provides a live coding environment, and delivers detailed feedback using AI.

The project is built as a monorepo using **Next.js** for the frontend and **FastAPI** on **Modal** for the serverless backend.

## ✨ Features

  * **📄 Resume & Context Parsing:** Upload a PDF resume and job description to generate a context-aware interview loop.
  * **tj Interview Planner:** AI agents (powered by Claude 3 Haiku via OpenRouter) generate a structured 3-4 stage interview plan.
  * **👨‍💻 Live Coding Canvas:** A split-screen interface featuring a fully functional **Monaco Editor** (Python) and a live transcript.
  * **🔄 Real-Time State Sync:** The code written in the browser is synchronized in real-time with the backend, allowing the AI to "read" the candidate's code.
  * **📝 Intelligent Feedback:** Generates actionable highlights and recommendations based on the code written and the interview plan.
  * **☁️ Serverless Deployment:** Entirely deployable via Modal, serving the static frontend directly alongside the API.

## 🏗️ Architecture & Tech Stack

### Frontend (`/frontend`)

  * **Framework:** Next.js 16 (App Router)
  * **Styling:** Tailwind CSS v4
  * **Editor:** `@monaco-editor/react`
  * **State:** React Hooks & local state management

### Backend (`/backend` / `fastapi_app.py`)

  * **Runtime:** Python 3.x
  * **Framework:** FastAPI
  * **Infrastructure:** [Modal](https://modal.com) (Serverless GPU/CPU compute)
  * **AI/LLM:** OpenRouter API (Anthropic Claude 3 Haiku)
  * **Validation:** Pydantic

-----

## 🚀 Getting Started

### Prerequisites

  * Node.js (v18+) and `pnpm` or `npm`
  * Python 3.10+
  * A [Modal](https://modal.com) account
  * An [OpenRouter](https://openrouter.ai) API Key

### 1\. Clone the Repository

```bash
git clone https://github.com/babusid/Nova_Hackathon_CMU.git
cd Nova_Hackathon_CMU
```

### 2\. Backend Setup

The backend runs on Modal. You need to set up your secrets first.

```bash
# Install Modal client
pip install modal

# Authenticate
modal setup

# Create the secret for OpenRouter
modal secret create openrouter-secret OPENROUTER_API_KEY=your_key_here
```

### 3\. Frontend Setup

Navigate to the frontend directory to install dependencies.

```bash
cd frontend
npm install
# or
pnpm install
```

-----

## 🛠️ Development Workflow

### Running Locally

To run the frontend locally (communicating with the deployed backend or a local backend instance):

1.  **Start the Frontend:**

    ```bash
    cd frontend
    npm run dev
    ```

    Access the app at `http://localhost:3000`.

2.  **Start the Backend (Dev Mode):**

    ```bash
    # From the root directory
    modal serve fastapi_app.py
    ```

    *Note: You may need to adjust the `PLANNER_URL` or CORS settings in the frontend `page.tsx` to point to your local or Modal dev URL.*

### 📦 Production Deployment

The application is designed to be served entirely from Modal. This involves building the Next.js app as a static site and mounting it into the Modal container.

1.  **Build Frontend:**
    Inside the `/frontend` directory, run:

    ```bash
    npm run build:static
    ```

    *This will generate an `out/` directory containing the static HTML/CSS/JS.*

2.  **Sync & Deploy:**
    Ensure the static output is located where the Modal script expects it (defined as `static_frontend` in `fastapi_app.py`).

    ```bash
    # From the root directory
    modal deploy fastapi_app.py
    ```

    Modal will return a URL (e.g., `https://your-username--syntherview-fastapi-app.modal.run`) where the full full-stack app is live.

-----

## 📂 Project Structure

```text
├── fastapi_app.py        # Main Backend Entrypoint (Modal + FastAPI)
├── frontend/             # Next.js Frontend Application
│   ├── app/              # App Router pages (page.tsx, layout.tsx)
│   ├── components/       # React components
│   ├── public/           # Static assets
│   ├── package.json      # Frontend dependencies
│   └── ...
└── static_frontend/      # (Generated) Static build output for deployment
```

## 🧩 Key Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/generate_plan` | Accepts PDF resume & Job Desc. Returns JSON interview plan. |
| `POST` | `/editor-state` | Syncs the Monaco editor content to the backend memory. |
| `POST` | `/generate_feedback` | Analyzes the final code state and generates a report. |
| `WS` | `/voice_ws` | (Experimental) WebSocket for voice-to-text pipeline. |

## 🚧 Roadmap

  * [ ] **Voice Integration:** Complete the `inference_loop` in `VoiceWebSocketSession` to support real-time speech-to-text and text-to-speech.
  * [ ] **Prompt Engineering:** Refine Claude 3 prompts for deeper technical scrutiny.
  * [ ] **Session Persistence:** Save interview transcripts and reports to a database.

## 🤝 Contributing

Contributions are welcome\! Please open an issue or submit a pull request for any bugs or feature enhancements.

## 📄 License

Distributed under the MIT License.
