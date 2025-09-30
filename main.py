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
from ddgs import DDGS

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

ACTIVE_MODULES = {}  # e.g. {"weather": module, "voice_chat": module}

def set_active_module(name: str):
    if name in LOADED_MODULES:
        ACTIVE_MODULES[name] = LOADED_MODULES[name]
        print(f"[AI] 🎯 Module '{name}' activated in chat.")
    else:
        print(f"[AI] ⚠️ Module '{name}' not loaded.")

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




def web_search(query, max_results=5):
    """Generic web search that returns clean titles + snippets."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            snippets = []
            for r in results:
                title = r.get("title", "No Title")
                body = r.get("body", "")
                link = r.get("href", "")
                snippets.append(f"- {title}: {body} (link: {link})")
            return "\n".join(snippets)
    except Exception as e:
        return f"[Web search failed: {e}]"


def needs_web_search(user_input, response):
    # If user is asking about real-time info (weather, news, time, price, etc.)
    realtime_keywords = ["weather", "today", "news", "price", "stock", "current", "latest", "time", "score"]
    if any(word in user_input.lower() for word in realtime_keywords):
        return True

    # If response looks like refusal / weak
    if len(response) < 30 or re.search(r"\b(i don't know|not sure|can't answer|sorry)\b", response, re.I):
        return True

    return False


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

            # ------------------------
            # STEP 1: Check active modules (smart matching)
            # ------------------------
            module_response = None
            if "ACTIVE_MODULES" in globals():
                user_text = user_input.lower()
                print("[AI] checking active modules...")

                for name, mod in ACTIVE_MODULES.items():
                    # normalize name for easier matching
                    simple_name = name.lower().replace("_module", "").replace("module", "")
                    print(simple_name)
                    # agar module ka naam user ke input mein aata hai
                    if simple_name in user_text or name.lower() in user_text:
                        print(f"[AI] matched module: {name}")
                        try:
                            if hasattr(mod, "process"):
                                r = mod.process(user_input)
                            elif hasattr(mod, "run"):
                                r = mod.run(user_input)
                            else:
                                r = None

                            if r and "could not understand" not in str(r).lower():
                                module_response = f"[{name}] {r}"
                                print(f"[AI] module response: {module_response}")
                                break
                        except Exception as me:
                            print(f"[AI] ⚠️ Module '{name}' failed: {me}")

            # ------------------------
            # STEP 2: Build AI prompt
            # ------------------------
            prompt_parts = ["You are an AI assistant. Use chat history and continue the conversation.\n"]
            for msg in history[-10:]:
                role = msg["role"]
                content = msg["content"]
                prompt_parts.append(f"{role.upper()}: {content}")

            if module_response:
                prompt_parts.append(f"MODULE: {module_response}")

            prompt_parts.append("ASSISTANT:")
            prompt = "\n".join(prompt_parts)

            ai_response = ai_generate_text(prompt).strip()

            # ------------------------
            # STEP 2b: Retry AI if refusal detected
            # ------------------------
            if re.search(r"real-time|cannot|sorry|apologies|no access", ai_response, re.I):
                print("[AI] ⚠️ Model tried to refuse, retrying with stricter instructions...")
                force_prompt = f"""
                User asked: {user_input}

                The module '{name}' responded with:
                {module_response}

                Your job: Combine this module response into a natural conversational answer. 
                Do NOT refuse. Do NOT say you lack real-time capability. Always include the module output.
                Rules:
                - Do NOT refuse.
                - Do NOT mention real-time limitations.
                - Give a helpful and direct answer.
                - If you don't know, still attempt to construct a useful explanation.

                Now rewrite your answer without any refusal:
                """
                ai_response = ai_generate_text(force_prompt).strip()

            # ------------------------
            # STEP 3: Web search if needed
            # ------------------------
            web_response = None
            if needs_web_search(user_input, ai_response) and module_response is None:
                print("[AI] 🌍 Searching the web for better answer...")
                web_results = web_search(user_input, max_results=5)

                web_prompt = f"""
                You are an AI assistant with extended capabilities.
                The user has asked: "{user_input}"

                Here are the web search results you MUST use:
                {web_results}

                ⚠️ Very Important Instructions:
                - You MUST use the web results to answer.
                - DO NOT say "I cannot provide real-time info".
                - DO NOT refuse.
                - Your ONLY source is the web results above.
                - Summarize the relevant information from them.
                - If multiple results overlap, combine into one clear answer.
                - If nothing relevant is found, explicitly say:
                "No relevant information was found in search results."

                Now provide the final helpful answer:
                """
                web_response = ai_generate_text(web_prompt).strip()

                # retry if refusal detected in web response
                if re.search(r"real-time|cannot|sorry|apologies|no access", web_response, re.I):
                    print("[AI] ⚠️ Model tried to refuse web response, retrying...")
                    force_prompt = f"""
                    User asked: {user_input}

                    Web results:
                    {web_results}

                    Your job: Summarize ONLY from web results. 
                    If relevant info exists → give it directly. 
                    If not → just say: "No relevant information was found."
                    Do NOT say anything about lacking internet or real-time access.
                    """
                    web_response = ai_generate_text(force_prompt).strip()

            # ------------------------
            # STEP 4: Combine responses
            # ------------------------
            final_parts = []
            if module_response:
                final_parts.append(module_response)
            if ai_response:
                final_parts.append(f"[ai] {ai_response}")
            if web_response:
                final_parts.append(f"[web] {web_response}")

            final_answer = "\n".join(final_parts)

            # ------------------------
            # STEP 5: Print + save
            # ------------------------
            print(f"AI: {final_answer}")
            history.append({"role": "assistant", "content": final_answer})
            save_chat_history(history)

        except KeyboardInterrupt:
            print("\n[AI] Chat interrupted.")
            break
        except Exception as e:
            print(f"[AI] Chat error: {e}")
            break



def install_package(package_name: str):
    """
    Install a Python package using pip programmatically.
    Example: install_package("requests")
    """
    try:
        print(f"[AI] Installing package: {package_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"[AI] ✅ Package '{package_name}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[AI] ❌ Failed to install '{package_name}'. Error: {e}")
    except Exception as e:
        print(f"[AI] ⚠️ Unexpected error: {e}")


LOADED_MODULES = {}

def load_module(module_name: str):
    """
    Dynamically load a project/module from 'features/' folder.
    """
    try:
        module_path = f"features.{module_name}"
        
        if module_path in sys.modules:
            # reload if already loaded
            module = importlib.reload(sys.modules[module_path])
        else:
            module = importlib.import_module(module_path)

        # agar module ke pass run() ya register() hai to call karo
        if hasattr(module, "run"):
            LOADED_MODULES[module_name] = module
            print(f"[AI] ✅ Module '{module_name}' loaded. You can run it with: run_module {module_name}")
            print(LOADED_MODULES)
        else:
            print(f"[AI] ⚠️ Module '{module_name}' loaded but no 'run()' function found.")

    except Exception as e:
        print(f"[AI] ❌ Failed to load module '{module_name}': {e}")


def run_module(module_name: str):
    """
    Run a loaded module's `run()` function.
    If 'all' is passed, runs all loaded modules.
    """
    try:
        if module_name.lower() == "all":
            if not LOADED_MODULES:
                print("[AI] ⚠️ No modules are loaded.")
                return

            print("[AI] ▶ Running all loaded modules...")
            for name, module in LOADED_MODULES.items():
                if hasattr(module, "run"):
                    try:
                        print(f"\n[AI] ▶ Running module '{name}' ...")
                        set_active_module(name)
                        result = module.run()
                        if result:
                            print(f"[{name}] {result}")
                    except Exception as e:
                        print(f"[AI] ❌ Error running '{name}': {e}")
                else:
                    print(f"[AI] ⚠️ Module '{name}' has no run() function.")
            return

        # -----------------------
        # Single module execution
        # -----------------------
        module = LOADED_MODULES.get(module_name)
        if not module:
            print(f"[AI] ⚠️ Module '{module_name}' is not loaded. Use 'load_module {module_name}' first.")
            return

        if hasattr(module, "run"):
            print(f"[AI] ▶ Running module '{module_name}' ...")
            set_active_module(module_name)
            result = module.run()
            if result:
                print(f"[{module_name}] {result}")
        else:
            print(f"[AI] ⚠️ Module '{module_name}' has no run() function.")

    except Exception as e:
        print(f"[AI] ❌ Error running module '{module_name}': {e}")
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
            if raw.startswith("pip_install"):
                parts = raw.split(" ", 1)
                if len(parts) < 2:
                    print("Usage: pip_install <package>")
                    continue
                _, package = parts
                install_package(package)
                continue
            if raw.startswith("load_module"):
                parts = raw.split(" ", 1)
                if len(parts) < 2:
                    print("Usage: load_module <name>")
                    continue
                _, name = parts
                load_module(name)
                continue

            if raw.startswith("run_module"):
                parts = raw.split(" ", 1)
                if len(parts) < 2:
                    print("Usage: run_module <name>")
                    continue
                _, name = parts
                run_module(name)
                continue
            print("unknown command. help")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[AI] CLI error: {e}")

if __name__ == "__main__":
    cli()
