# 🤖 Jugaru Agent

An AI-powered automation and project-generation tool for Windows/Linux.  
This script can **generate projects**, **edit existing code**, **run projects**, and even **chat with AI** using either [Ollama](https://ollama.ai) or [Hugging Face Transformers](https://huggingface.co/).

---

## ✨ Features
- **Project Generator**
  - Create complete runnable projects in Python (Flask), PHP (Laravel/CodeIgniter), Node.js (Express), or React.
  - Auto-generates common files like `index.html`, `style.css`, `requirements.txt`, etc.

- **Code Editor**
  - Apply natural-language instructions to edit code across multiple files.

- **Run Projects**
  - Detects project type (Python, Node, PHP) and runs it automatically.

- **Chat Assistant**
  - Conversational mode with context memory.
  - Can integrate with custom modules inside the `features/` folder.
  - Supports web search (via DuckDuckGo) when real-time data is needed.

- **Dynamic Modules**
  - Load and run feature modules from `features/` folder.
  - Activate modules for interactive use in chat.

- **Package Installer**
  - Install Python packages directly from the CLI.

---

## 📦 Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
