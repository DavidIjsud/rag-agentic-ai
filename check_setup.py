"""Week 0 setup check for the RAG & Agentic AI course.

Run it from anywhere inside your repo:

    uv run scripts/check_setup.py

It reads .env, checks your tools, makes one tiny model call (free on Gemini's
free tier), checks LangSmith and your Supabase database, and prints one line
per check. Secrets are never printed in full. Paste the output into our chat.

Version 2 (15 Nov 2026): LLM_MODEL can name any of these providers:
openai, anthropic, google_genai (Gemini) and ollama (local models, no key).
The last two are called through LangChain's init_chat_model(), the same
function the course code uses, so changing models means changing one line.

Version 3 (16 Nov 2026): the Gemini test call gives up after about a minute
(25 s timeout, 2 attempts) and says why, instead of retrying for minutes.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

PASS, WARN, FAIL = "pass", "warn", "fail"
ICONS = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌"}
DEFAULT_MODEL = "google_genai:gemini-3.7-flash"
# Provider -> the environment variables that can hold its key (the first one set wins).
KEY_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "ollama": (),  # runs on your Mac, so no key
}
# Providers called through LangChain, and the integration package each one needs.
LANGCHAIN_PACKAGES = {"google_genai": "langchain-google-genai", "ollama": "langchain-ollama"}
# Limits for the test call, so a busy or unreachable API fails in about a minute instead
# of hanging. Gemini's defaults are no timeout and 6 attempts with growing waits between them.
CALL_OPTIONS: dict[str, dict[str, object]] = {"google_genai": {"timeout": 25, "max_retries": 2}}
TRACE_WAIT_SECONDS = 20
AGENT_ENVIRONMENTS = {"local", "development", "staging", "production"}
HTTP_HINTS = {
    401: "the key is wrong or was revoked",
    403: "this key's project can't use that model",
    404: "model not found: check LLM_MODEL",
    429: "no credit left (insufficient_quota) or too many requests: check Billing",
}
URL_RESERVED = "#?/@[]"  # must be percent-encoded inside a password in a URL
ENCODE_HINT = "percent-encode them (# → %23, ? → %3F, / → %2F, @ → %40) or reset the password to letters and digits"

results: list[str] = []


def report(status: str, name: str, detail: str = "") -> None:
    results.append(status)
    print(f"{ICONS[status]} {name}" + (f": {detail}" if detail else ""))


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    for folder in (here, *here.parents):
        if (folder / "pyproject.toml").exists():
            return folder
    return Path.cwd()


def mask(value: str) -> str:
    return f"{value[:7]}… ({len(value)} chars)"


def first_line(error: Exception, limit: int = 180) -> str:
    lines = str(error).strip().splitlines()
    text = lines[0] if lines else ""
    if len(text) > limit:
        text = text[:limit] + "…"
    return f"{type(error).__name__}: {text}"


def check_python() -> None:
    major, minor, micro = sys.version_info[:3]
    version = f"{major}.{minor}.{micro}"
    if (major, minor) in {(3, 12), (3, 13)}:
        report(PASS, "Python", version)
    elif (major, minor) >= (3, 14):
        report(FAIL, "Python", f"{version}. CrewAI (Week 11) needs 3.13 or older. Run: uv python pin 3.12")
    elif (major, minor) >= (3, 10):
        report(WARN, "Python", f"{version} works, but the course uses 3.12. Run: uv python pin 3.12")
    else:
        report(FAIL, "Python", f"{version} is too old. Run: uv python pin 3.12")


def check_project(root: Path) -> None:
    uv = shutil.which("uv")
    if uv:
        report(PASS, "uv", run([uv, "--version"], root).stdout.strip())
    else:
        report(FAIL, "uv", "not on your PATH. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh")

    if Path(sys.prefix).resolve() == (root / ".venv").resolve():
        report(PASS, "Project environment", ".venv")
    else:
        report(FAIL, "Project environment", f"running from {sys.prefix}. Run it with: uv run scripts/check_setup.py")

    if (root / "uv.lock").exists():
        report(PASS, "uv.lock", "found (commit it)")
    else:
        report(FAIL, "uv.lock", "missing. Run: uv lock")


def check_git(root: Path) -> None:
    git = shutil.which("git")
    if not git or run([git, "rev-parse", "--is-inside-work-tree"], root).returncode != 0:
        report(FAIL, "Git repo", "not found. Run: git init")
        return

    if (root / ".env").exists():
        if run([git, "ls-files", "--error-unmatch", ".env"], root).returncode == 0:
            report(FAIL, ".env kept out of git", "it IS committed. Run: git rm --cached .env, then rotate every key in it")
        elif run([git, "check-ignore", "-q", ".env"], root).returncode == 0:
            report(PASS, ".env kept out of git", "ignored by .gitignore")
        else:
            report(FAIL, ".env kept out of git", "not ignored. Add .env to .gitignore before you commit")

    remote = run([git, "remote", "get-url", "origin"], root)
    if remote.returncode == 0:
        report(PASS, "GitHub remote", re.sub(r"//[^@/]+@", "//", remote.stdout.strip()))
    else:
        report(WARN, "GitHub remote", "none yet. Create the repo on GitHub, then: git remote add origin <url> && git push -u origin HEAD")


def load_env(root: Path) -> bool:
    env_file = root / ".env"
    if not env_file.exists():
        report(FAIL, ".env", "missing. Run: cp .env.example .env, then fill in your values")
        return False
    try:
        from dotenv import dotenv_values
    except ImportError:
        report(FAIL, ".env", "python-dotenv is not installed. Run: uv add python-dotenv")
        return False

    values = dotenv_values(env_file)
    overridden = sorted(k for k, v in values.items() if v and os.environ.get(k) not in (None, v))
    for key, value in values.items():  # same rule as load_dotenv(): the shell wins
        if value is not None and key not in os.environ:
            os.environ[key] = value

    detail = f"{sum(1 for v in values.values() if v)} values set"
    if overridden:
        report(WARN, ".env", f"{detail}, but your shell overrides {', '.join(overridden)}. Remove those exports from ~/.zshrc")
    else:
        report(PASS, ".env", detail)
    return True


def check_langsmith() -> bool:
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        report(FAIL, "LangSmith", "set LANGSMITH_TRACING=true")
        return False
    key = os.getenv("LANGSMITH_API_KEY", "")
    if not key:
        report(FAIL, "LangSmith", "LANGSMITH_API_KEY is missing from .env")
        return False

    project = os.getenv("LANGSMITH_PROJECT", "")
    agent_id = os.getenv("LANGSMITH_AGENT_ID", "")
    agent_env = os.getenv("LANGSMITH_AGENT_ENVIRONMENT", "")
    if project and (agent_id or agent_env):
        report(FAIL, "LangSmith", "use LANGSMITH_PROJECT or the two LANGSMITH_AGENT_* values, not both: LangSmith rejects runs that have both")
        return False
    if bool(agent_id) != bool(agent_env):
        report(FAIL, "LangSmith", "set both LANGSMITH_AGENT_ID and LANGSMITH_AGENT_ENVIRONMENT, or neither")
        return False
    if agent_env and agent_env not in AGENT_ENVIRONMENTS:
        report(FAIL, "LangSmith", f"LANGSMITH_AGENT_ENVIRONMENT must be one of: {', '.join(sorted(AGENT_ENVIRONMENTS))}")
        return False

    try:
        from langsmith import Client

        next(iter(Client().list_projects(limit=1)), None)
    except ImportError:
        report(FAIL, "LangSmith", "langsmith is not installed. Run: uv add langsmith")
        return False
    except Exception as error:
        if "Auth" in type(error).__name__ or "Forbidden" in type(error).__name__:
            hint = " (check the key; for an EU account also set LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com)"
        elif "Connection" in type(error).__name__:
            hint = " (no connection to LangSmith: check your network or VPN)"
        else:
            hint = ""
        report(FAIL, "LangSmith", first_line(error) + hint)
        return False

    destination = f"agent {agent_id} ({agent_env})" if agent_id else f"project {project or 'default'}"
    report(PASS, "LangSmith", f"key accepted {mask(key)} · traces go to {destination}")
    return True


# --- Model call ---------------------------------------------------------------


def message_text(message: object) -> str:
    """The text of a LangChain message, whether its content is a string or a list of blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def call_openai(model: str, prompt: str, trace: bool) -> tuple[str, object, object]:
    from openai import OpenAI

    client = OpenAI(timeout=25.0, max_retries=1)
    if trace:
        from langsmith.wrappers import wrap_openai

        client = wrap_openai(client)
    response = client.responses.create(model=model, input=prompt, max_output_tokens=300)
    return response.output_text, response.usage.input_tokens, response.usage.output_tokens


def call_anthropic(model: str, prompt: str, trace: bool) -> tuple[str, object, object]:
    import anthropic

    client = anthropic.Anthropic(timeout=25.0, max_retries=1)
    if trace:
        from langsmith.wrappers import wrap_anthropic

        client = wrap_anthropic(client)
    response = client.messages.create(model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}])
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens


def call_langchain(provider: str, model: str, prompt: str, trace: bool) -> tuple[str, object, object]:
    if not trace:
        os.environ["LANGSMITH_TRACING"] = "false"  # LangSmith isn't working, so don't try to send traces
    from langchain.chat_models import init_chat_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chat = init_chat_model(model, model_provider=provider, **CALL_OPTIONS.get(provider, {}))
        message = chat.invoke(prompt)
    usage = getattr(message, "usage_metadata", None) or {}
    return message_text(message), usage.get("input_tokens", "?"), usage.get("output_tokens", "?")


def flush_traces() -> bool:
    """LangChain uploads traces in the background. Wait for them, but not forever."""
    from langchain_core.tracers.langchain import wait_for_all_tracers

    waiter = threading.Thread(target=wait_for_all_tracers, daemon=True)
    waiter.start()
    waiter.join(TRACE_WAIT_SECONDS)
    return not waiter.is_alive()


def model_hint(error: Exception, provider: str, model: str) -> str:
    text, name = str(error), type(error).__name__
    status = getattr(error, "status_code", None)
    if provider in ("openai", "anthropic") and status in HTTP_HINTS:
        return HTTP_HINTS[status]
    if provider == "google_genai":
        if "limit: 0" in text:
            return "this model has no free-tier quota for your project: set LLM_MODEL to another model listed in AI Studio"
        if "RESOURCE_EXHAUSTED" in text or "429" in text:
            return "free-tier limit reached: wait a minute and run the check again"
        if "UNAVAILABLE" in text or "503" in text or "overloaded" in text.lower():
            return "Google's servers are busy right now: try again in a few minutes, or use another model"
        if "API_KEY_INVALID" in text or "API key not valid" in text or "UNAUTHENTICATED" in text or "401" in text:
            return "Google rejected the key: check GOOGLE_API_KEY, or create a new key in AI Studio"
        if "location is not supported" in text:
            return "the Gemini API isn't available from your location"
        if "NOT_FOUND" in text or "404" in text:
            return "model not found: check LLM_MODEL against the models listed in AI Studio"
        if "Timeout" in name or "timed out" in text.lower() or "DEADLINE_EXCEEDED" in text:
            return f"no answer within {CALL_OPTIONS['google_genai']['timeout']} s: check your network or VPN, or try again later"
    if provider == "ollama":
        if "not found" in text.lower():
            return f"the model isn't downloaded yet. Run: ollama pull {model}"
        if "connect" in text.lower() or "Connect" in name:
            return f"Ollama isn't running. Open the Ollama app, then run: ollama pull {model}"
    if "Connect" in name or "Timeout" in name:
        return "no connection to the API: check your network or VPN"
    return ""


def check_model(trace: bool) -> None:
    spec = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    provider, _, model = spec.partition(":")
    if provider not in KEY_VARS or not model:
        report(FAIL, "LLM_MODEL", f"{spec!r}. Use provider:model with one of: {', '.join(KEY_VARS)}. Example: {DEFAULT_MODEL}")
        return
    names = KEY_VARS[provider]
    if names:
        found = next((name for name in names if os.getenv(name)), None)
        if not found:
            report(FAIL, names[0], f"missing from .env (LLM_MODEL={spec} needs it)")
            return
        report(PASS, found, mask(os.environ[found]))

    prompt = "Reply with exactly one word: listo"
    print(f"⏳ Model call: waiting for {spec} (up to a minute)…", end="", flush=True)
    started = time.perf_counter()
    try:
        if provider == "openai":
            text, tokens_in, tokens_out = call_openai(model, prompt, trace)
        elif provider == "anthropic":
            text, tokens_in, tokens_out = call_anthropic(model, prompt, trace)
        else:
            text, tokens_in, tokens_out = call_langchain(provider, model, prompt, trace)
    except ImportError as error:
        print("\r\033[K", end="")  # clear the waiting line
        if provider in LANGCHAIN_PACKAGES:
            fix = f"uv add langchain {LANGCHAIN_PACKAGES[provider]}"
        else:
            fix = f"uv add {(error.name or provider).split('.')[0]}"
        report(FAIL, "Model call", f"a package is missing ({first_line(error, 90)}). Run: {fix}")
        return
    except Exception as error:
        print("\r\033[K", end="")
        hint = model_hint(error, provider, model)
        report(FAIL, "Model call", f"{spec}: {first_line(error)}" + (f" ({hint})" if hint else ""))
        return

    seconds = time.perf_counter() - started
    traced = ""
    if trace:
        sent = flush_traces() if provider in LANGCHAIN_PACKAGES else True
        traced = " · trace sent to LangSmith" if sent else " · trace still uploading to LangSmith"
    print("\r\033[K", end="")
    reply = repr(text.strip()[:40]) if text.strip() else "no text (the output went to thinking)"
    report(PASS, "Model call", f"{spec} replied {reply} · {tokens_in} in / {tokens_out} out tokens · {seconds:.1f} s{traced}")


# --- Supabase -------------------------------------------------------------------


def split_db_url(url: str) -> tuple[str, str, str]:
    """(user, password, host) of a postgresql:// URL. The host follows the last @."""
    rest = url.split("://", 1)[-1]
    userinfo, _, hostpart = rest.rpartition("@")
    user, _, password = userinfo.partition(":")
    host = hostpart.split("/", 1)[0].split("?", 1)[0].rsplit(":", 1)[0]
    return user, password, host or "?"


def check_supabase() -> None:
    url = os.getenv("SUPABASE_DB_URL", "")
    if not url:
        report(FAIL, "Supabase", "SUPABASE_DB_URL is missing from .env (Dashboard > Connect > Session pooler)")
        return
    try:
        import psycopg
    except ImportError:
        report(FAIL, "Supabase", 'psycopg is not installed. Run: uv add "psycopg[binary]"')
        return

    user, password, host = split_db_url(url)
    unsafe = " ".join(sorted({char for char in password if char in URL_RESERVED}))
    breaks_libpq = "/" in unsafe or "@" in unsafe  # psycopg can't read the URL then
    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            server = conn.execute("show server_version").fetchone()[0]
            row = conn.execute("select extversion from pg_extension where extname = 'vector'").fetchone()
    except Exception as error:
        message = first_line(error)
        if password:
            message = message.replace(password, "****")
        if breaks_libpq:
            hint = f"the password contains {unsafe}, which break the URL: {ENCODE_HINT}"
        elif "password authentication failed" in str(error):
            hint = "check the database password, or reset it to letters and digits"
        elif host.startswith("db.") and host.endswith(".supabase.co"):
            hint = "use the Session pooler string: the direct host only works on IPv6 networks"
        else:
            hint = "check the string from Connect > Session pooler; if the project is paused, restore it in the dashboard"
        report(FAIL, "Supabase", f"{message} ({hint})")
    else:
        project = user.removeprefix("postgres.") or "?"
        if row:
            report(PASS, "Supabase", f"project {project} · {host} · Postgres {server.split()[0]} · vector {row[0]} enabled")
        else:
            report(FAIL, "Supabase", "connected, but pgvector is off. In the SQL editor run: create extension if not exists vector with schema extensions;")
    if unsafe and not breaks_libpq:
        report(WARN, "Database password", f"it contains {unsafe} without percent-encoding. psycopg copes, but other tools misread the URL: {ENCODE_HINT}")


def main() -> int:
    logging.getLogger("langsmith").setLevel(logging.ERROR)  # keep the report readable
    root = find_root()
    print(f"Week 0 setup check v3 · {datetime.now():%Y-%m-%d %H:%M} · {root.name}\n")
    check_python()
    check_project(root)
    check_git(root)
    if load_env(root):
        langsmith_ok = check_langsmith()
        check_model(trace=langsmith_ok)
        check_supabase()

    failed, warned = results.count(FAIL), results.count(WARN)
    print(f"\n{results.count(PASS)} passed · {warned} warning{'s' if warned != 1 else ''} · {failed} failed")
    print("Week 0 setup is ready. Paste this output into our chat." if not failed else "Fix the ❌ lines, then run the check again.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())