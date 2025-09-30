#!/usr/bin/env python
"""
Jugaru-agent (Windows/Linux) - agent.py
"""

import os
import re
import sys
import subprocess
import importlib
import shutil
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = BASE_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

PREFER_OLLAMA = os.environ.get("PREFER_OLLAMA", "0") == "1"

# ---------- Utilities ----------
def run_cmd(cmd, cwd=None, shell=False, capture_output=True):
    try:
        p = subprocess.run(cmd, cwd=cwd, shell=shell, check=False,
                           stdout=subprocess.PIPE if capture_output else None,
                           stderr=subprocess.PIPE if capture_output else None,
                           text=True)
        return p.returncode, (p.stdout or ""), (p.stderr or "")
    except Exception as e:
        return 1, "", str(e)

def which_prog(name):
    return shutil.which(name) is not None

def read_all_files(root: Path, exts=None):
    files = []
    for p in root.rglob('*'):
        if p.is_file():
            if exts is None or p.suffix.lower() in exts:
                rel = p.relative_to(root)
                files.append((str(rel), p.read_text(encoding='utf-8', errors='ignore')))
    return files

def slow_print(msg, delay=0.02):
    for c in msg:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print("")

def ensure_package(pkg_name, import_name=None):
    try:
        return importlib.import_module(import_name or pkg_name)
    except ImportError:
        print(f"[AI] Missing package '{pkg_name}', installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        return importlib.import_module(import_name or pkg_name)

# ---------- Model backends ----------
OLLAMA_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

TRANSFORMERS_AVAILABLE = True
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    TRANSFORMERS_AVAILABLE = False

def ensure_transformers_stack():
    global TRANSFORMERS_AVAILABLE, AutoTokenizer, AutoModelForCausalLM
    if TRANSFORMERS_AVAILABLE:
        return
    ensure_package("transformers")
    ensure_package("torch")
    ensure_package("accelerate")
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True

MODEL_BACKEND = {"type": None, "model": None, "hf_model_obj": None, "hf_tokenizer": None}

def load_model_backend(hf_model_name="codellama/CodeLlama-7b-Instruct-hf", ollama_model="deepseek-coder:6.7b"):
    if (PREFER_OLLAMA or OLLAMA_AVAILABLE):
        rc, _, _ = run_cmd(["ollama", "ls"], capture_output=True)
        if rc == 0:
            MODEL_BACKEND.update({"type": "ollama", "model": ollama_model})
            print(f"[AI] Using Ollama backend with model {ollama_model}")
            return
    ensure_transformers_stack()
    MODEL_BACKEND.update({"type": "hf", "model": hf_model_name})
    print(f"[AI] Using HuggingFace backend with model {hf_model_name}")

def unload_hf_model():
    m = MODEL_BACKEND.get("hf_model_obj")
    t = MODEL_BACKEND.get("hf_tokenizer")
    if m or t:
        MODEL_BACKEND["hf_model_obj"] = None
        MODEL_BACKEND["hf_tokenizer"] = None
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def generate_with_ollama(prompt, model="deepseek-coder:6.7b"):
    try:
        resp = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
        print(resp.get("message", {}).get("content", ""))
        return resp.get("message", {}).get("content", "")
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed: {e}")

def generate_with_hf(prompt, model_name, max_tokens=2000, temperature=0.2):
    import torch
    if MODEL_BACKEND.get("hf_tokenizer") is None:
        print("[AI] Loading HF tokenizer & model (may take a while)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
        try:
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        except Exception:
            print("[AI] Model load with device_map failed, retrying...")
            model_obj = AutoModelForCausalLM.from_pretrained(model_name)
        MODEL_BACKEND["hf_tokenizer"] = tokenizer
        MODEL_BACKEND["hf_model_obj"] = model_obj
    else:
        tokenizer = MODEL_BACKEND["hf_tokenizer"]
        model_obj = MODEL_BACKEND["hf_model_obj"]

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cpu")
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    device = next(model_obj.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_obj.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=4000,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=pad_id,
            eos_token_id=getattr(tokenizer, "eos_token_id", None)
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def ai_generate_text(prompt):
    if MODEL_BACKEND["type"] is None:
        load_model_backend()
    if MODEL_BACKEND["type"] == "ollama":
        return generate_with_ollama(prompt, model=MODEL_BACKEND["model"])
    return generate_with_hf(prompt, model_name=MODEL_BACKEND["model"])

# ---------- File parsing ----------
def parse_files_from_output(output_text: str):
    files = {}
    # Strict regex for "### file: some/path.ext" then code block
    pattern = re.compile(
        r"###\s*file:\s*([^\n]+?)\s*\n```[a-zA-Z0-9]*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(output_text)

    for fn, content in matches:
        clean_fn = fn.strip().replace("\r", "").replace("\n", "")

        # Skip invalid or placeholder paths
        if any(
            bad in clean_fn.lower()
            for bad in ["<path>", "relative/path", "example", "file contents"]
        ):
            continue
        if clean_fn.startswith(("relative/path", "<")) or clean_fn == "":
            continue

        # Normalize Windows/Unix paths
        safe_path = Path(clean_fn).as_posix()

        # Only keep non-empty code blocks
        clean_content = content.strip()
        if clean_content:
            files[safe_path] = clean_content + "\n"

    # If nothing matched, try single fenced block fallback
    if not files:
        fence = re.search(r"```(?:php|python|html|css|js|json|txt|bat|sh)?\n(.*?)```", output_text, re.DOTALL)
        if fence:
            code = fence.group(1).rstrip() + "\n"

            # detect project type
            lower_out = output_text.lower()
            if "php" in lower_out:
                default_file = "index.php"
            elif "html" in lower_out:
                default_file = "index.html"
            elif "css" in lower_out:
                default_file = "style.css"
            elif "javascript" in lower_out or "js" in lower_out:
                default_file = "script.js"
            elif "json" in lower_out:
                default_file = "data.json"
            elif "bat" in lower_out:
                default_file = "run.bat"
            elif "sh" in lower_out:
                default_file = "run.sh"
            else:
                default_file = "main.py"  # python fallback

            files[default_file] = code

    # Final fallback
    if not files:
        files["main.py"] = "print('Hello World')\n"

    return files

# ---------- Project creation ----------
def ai_generate_project(name, ptype, purpose="auto-generated"):
    feature_path = FEATURES_DIR / name
    feature_path.mkdir(parents=True, exist_ok=True)
    print(f"[AI] Generating project '{name}' ({ptype}) ...")

    required_files = []
    if "flask" in ptype.lower() or "python" in ptype.lower():
        required_files = [
            "main.py",
            "templates/index.html",
            "static/style.css",
            "static/script.js",
            "requirements.txt",
            "run.bat",
            "run.sh"
        ]
    elif "laravel" in ptype.lower():
        required_files = [
            "routes/web.php",
            "resources/views/welcome.blade.php",
            "public/css/style.css",
            "public/js/app.js",
            "composer.json",
            "run.bat",
            "run.sh"
        ]
    elif "codeigniter" in ptype.lower() or "php" in ptype.lower():
        required_files = [
            "index.php",
            "application/config/routes.php",
            "application/views/welcome_message.php",
            "assets/style.css",
            "assets/app.js",
            "run.bat",
            "run.sh"
        ]
    elif "node" in ptype.lower() or "express" in ptype.lower():
        required_files = [
            "server.js",
            "views/index.ejs",
            "public/style.css",
            "public/app.js",
            "package.json",
            "run.bat",
            "run.sh"
        ]
    elif "react" in ptype.lower():
        required_files = [
            "src/App.js",
            "src/index.js",
            "public/index.html",
            "src/App.css",
            "package.json",
            "run.bat",
            "run.sh"
        ]
    else:
        required_files = ["index.html", "style.css", "script.js"]

    prompt = f"""
        You are a code generator. Create a complete runnable **{ptype}** project for: "{purpose}".

        ⚠️ RULES (follow strictly):
        - Output ONLY code blocks for files.
        - For each file:
        - First line: "### file: relative/path/to/file"
        - Then a fenced code block with correct language.
        - Example:

            ### file: app/main.py
            ```python
            print("hello")
            ```

        - Do NOT add explanations, steps, or lists outside of files.
        - Required files at minimum:
        {json.dumps(required_files, indent=2)}
        """

    try:
        out_text = ai_generate_text(prompt)
    except Exception as e:
        print(f"[AI] Generation failed: {e}")
        (feature_path / "README.md").write_text("Project failed.\n")
        return True, feature_path

    files = parse_files_from_output(out_text)

    # write files
    for relpath, content in files.items():
        relpath = relpath.strip().lstrip("/\\")
        target = feature_path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        slow_print(f"[AI] writing {relpath} ...")
        target.write_text(content, encoding="utf-8")
        time.sleep(0.2)

    print(f"[AI] ✅ Project created at {feature_path}")
    return True, feature_path


def run_project(project_path: str):
    project = Path(project_path)
    if not project.exists():
        print("[AI] Project path not found.")
        return False

    # detect type
    if (project / "main.py").exists():
        cmd = ["python", "main.py"]
    elif (project / "server.js").exists():
        cmd = ["node", "server.js"]
    elif (project / "artisan").exists():
        cmd = ["php", "artisan", "serve"]
    elif (project / "index.php").exists():
        cmd = ["php", "-S", "localhost:8000", "index.php"]
    else:
        print("[AI] Could not detect project type.")
        return False

    print(f"[AI] Running project: {' '.join(cmd)} in {project}")
    try:
        subprocess.Popen(cmd, cwd=project)
        return True
    except Exception as e:
        print(f"[AI] Run failed: {e}")
        return False


# add to CLI
def cli():
    load_model_backend()
    print("[jugaru-agent] Ready. Type 'help'")
    while True:
        try:
            raw = input("> ").strip()
            if not raw:
                continue
            if raw in ("exit", "quit"):
                break
            if raw == "help":
                print("Commands:\n  auto_project <name> <ptype> \"<purpose>\"\n"
                      "  auto_edit <project_path> \"<instructions>\"\n"
                      "  run_project <project_path>\n"
                      "  reload_model\n  exit")
                continue
            if raw.startswith("auto_project"):
                parts = raw.split(" ", 3)
                if len(parts) < 4:
                    print("Usage: auto_project <name> <ptype> \"<purpose>\"")
                    continue
                _, name, ptype, purpose = parts
                ai_generate_project(name, ptype, purpose)
                continue
            if raw.startswith("auto_edit"):
                parts = raw.split(" ", 2)
                if len(parts) < 3:
                    print('Usage: auto_edit <project_path> "instructions"')
                    continue
                _, path, instr = parts
                ai_edit_project(path, instr)
                continue
            if raw.startswith("run_project"):
                parts = raw.split(" ", 1)
                if len(parts) < 2:
                    print("Usage: run_project <project_path>")
                    continue
                _, path = parts
                run_project(path)
                continue
            if raw == "reload_model":
                print("[AI] Reloading...")
                unload_hf_model()
                MODEL_BACKEND["type"] = None
                load_model_backend()
                continue
            print("unknown command. help")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[AI] CLI error: {e}")



# ---------- Project editing ----------
def ai_edit_project(project_path: str, edit_instructions: str):
    project = Path(project_path)
    if not project.exists():
        print("[AI] Project path does not exist.")
        return False, project

    # Read all project files
    files = read_all_files(project)

    # Build prompt more carefully
    parts = [
        "You are an assistant that edits code.",
        "Rules:",
        "- Only return modified files.",
        "- Keep unrelated code unchanged.",
        "- Do not regenerate whole project.",
        "- Always preserve indentation and style.",
        "- Apply edits exactly as instructed.",
        "- Apply edits on multiple files if given",
        """- For file or multiple files:
        - First line: "### file: relative/path/to/file"
        - Then a fenced code block with correct language.
        - Example:

            ### file: app/main.py
            ```python
            print("hello")
            ```

        - Do NOT add explanations, steps, or lists outside of files.""",
    ]
    for rel, content in files:
        parts.append(f"--- {rel} ---\n{content}\n")
    parts.append(f"Now apply this edit:\n{edit_instructions}\n")
    prompt = "\n".join(parts)

    try:
        out_text = ai_generate_text(prompt)
    except Exception as e:
        print(f"[AI] Edit generation failed: {e}")
        return False, project

    files_out = parse_files_from_output(out_text)
    if not files_out:
        print("[AI] No modified files returned.")
        return False, project

    for rel, content in files_out.items():
        safe_rel = Path(rel)
        if safe_rel.is_absolute() or ".." in safe_rel.parts:
            continue

        target = project / safe_rel

        # If file already exists, overwrite with updated version
        if target.exists():
            slow_print(f"[AI] updating {rel} ...")
        else:
            slow_print(f"[AI] creating {rel} ...")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        time.sleep(0.2)

    print(f"[AI] ✅ Edited {list(files_out.keys())}")
    return True, project



CHAT_FILE = BASE_DIR / "chat_history.json"

def load_chat_history():
    if CHAT_FILE.exists():
        try:
            return json.loads(CHAT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_chat_history(history):
    CHAT_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

def chat_loop():
    print("[AI] Chat mode started. Type 'exit' to quit chat.")
    history = load_chat_history()

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("[AI] Chat ended.")
                break

            # add user message
            history.append({"role": "user", "content": user_input})

            # build prompt (past + current)
            prompt_parts = ["You are an AI assistant. Use chat history and continue the conversation.\n"]
            for msg in history[-10:]:  # last 10 msgs for context
                role = msg["role"]
                content = msg["content"]
                prompt_parts.append(f"{role.upper()}: {content}")
            prompt_parts.append("ASSISTANT:")

            prompt = "\n".join(prompt_parts)

            # generate
            response = ai_generate_text(prompt)

            # clean response
            response = response.strip()
            print(f"AI: {response}")

            # save response
            history.append({"role": "assistant", "content": response})
            save_chat_history(history)

        except KeyboardInterrupt:
            print("\n[AI] Chat interrupted.")
            break
        except Exception as e:
            print(f"[AI] Chat error: {e}")
            break
# ---------- CLI ----------
def cli():
    load_model_backend()
    print("[jugaru-agent] Ready. Type 'help'")
    while True:
        try:
            raw = input("> ").strip()
            if not raw:
                continue
            if raw in ("exit", "quit"):
                break
            if raw == "help":
                print("Commands:\n  auto_project <name> <ptype> \"<purpose>\"\n"
                      "  auto_edit <project_path> \"<instructions>\"\n"
                      "  reload_model\n  exit")
                continue
            if raw.startswith("auto_project"):
                parts = raw.split(" ", 3)
                if len(parts) < 4:
                    print("Usage: auto_project <name> <ptype> \"<purpose>\"")
                    continue
                _, name, ptype, purpose = parts
                ai_generate_project(name, ptype, purpose)
                continue
            if raw.startswith("auto_edit"):
                parts = raw.split(" ", 2)
                if len(parts) < 3:
                    print('Usage: auto_edit <project_path> "instructions"')
                    continue
                _, path, instr = parts
                ai_edit_project(path, instr)
                continue
            if raw == "reload_model":
                print("[AI] Reloading...")
                unload_hf_model()
                MODEL_BACKEND["type"] = None
                load_model_backend()
                continue
            if raw == "chat":
                chat_loop()
                continue
            print("unknown command. help")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[AI] CLI error: {e}")

if __name__ == "__main__":
    cli()
