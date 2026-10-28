"""
Ежедневный бот-аналитик новостей ресторанов.

Шаги (это и есть «агенты»):
  1. Сборщик   – тянет новости из Google News (RSS) и публичных Telegram-каналов.
  2. Аналитик  – Claude отсеивает мусор, определяет компанию, категорию, важность и пишет краткую суть.
  3. Отправщик – шлёт отчёт в Telegram и/или на почту.
"""
import hashlib
import html
import json
import os
import re
import smtplib
import ssl
import sys
from email.mime.text import MIMEText
from urllib.parse import quote

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEN_PATH = os.path.join(ROOT, "seen.json")
UA = {"User-Agent": "Mozilla/5.0 (compatible; RestoNewsBot/1.0)"}


# ---------- конфиг и память о том, что уже отправляли ----------

def load_config():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seen():
    if os.path.exists(SEEN_PATH):
        with open(SEEN_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_seen(seen):
    # храним последние 3000 записей, чтобы файл не рос бесконечно
    with open(SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(seen)[-3000:], f)


def item_id(item):
    return hashlib.sha1(item["url"].encode("utf-8")).hexdigest()[:16]


# ---------- агент 1: сборщик ----------

def fetch_google_news(query, lang="ru", country="UZ"):
    url = (
        "https://news.google.com/rss/search?q="
        f"{quote(query)}+when:2d&hl={lang}&gl={country}&ceid={country}:{lang}"
    )
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:15]:
        summary = BeautifulSoup(e.get("summary", ""), "html.parser").get_text(" ")
        items.append(
            {
                "source": "Google News",
                "title": e.get("title", ""),
                "url": e.get("link", ""),
                "text": summary[:500],
            }
        )
    return items


def fetch_telegram_channel(handle):
    r = requests.get(f"https://t.me/s/{handle}", headers=UA, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    items = []
    for msg in soup.select(".tgme_widget_message")[-15:]:
        text_el = msg.select_one(".tgme_widget_message_text")
        link_el = msg.select_one(".tgme_widget_message_date")
        if not text_el or not link_el:
            continue
        text = text_el.get_text(" ")
        items.append(
            {
                "source": f"t.me/{handle}",
                "title": text[:120],
                "url": link_el["href"],
                "text": text[:800],
            }
        )
    return items


def collect(cfg):
    items = []
    queries = list(cfg.get("general_queries", []))
    for company in cfg.get("companies", []):
        queries.append(f'"{company}" Ташкент')
    for q in queries:
        try:
            items += fetch_google_news(q)
        except Exception as e:
            print(f"[warn] Google News '{q}': {e}", file=sys.stderr)
    for handle in cfg.get("telegram_channels", []) or []:
        try:
            items += fetch_telegram_channel(handle)
        except Exception as e:
            print(f"[warn] Telegram '{handle}': {e}", file=sys.stderr)

    # убираем дубли по ссылке внутри одного запуска
    unique = {}
    for it in items:
        if it["url"]:
            unique.setdefault(item_id(it), it)
    return unique


# ---------- агент 2: аналитик (Claude) ----------

SYSTEM_PROMPT = """Ты аналитик рынка общепита и доставки еды в Узбекистане (Ташкент).
Тебе дают список новостей и постов. Для каждого реши, важен ли он для владельца или менеджера,
который следит за конкурентами и рынком.
Важное: новые акции и программы лояльности, изменения цен и меню, открытие/закрытие точек,
конфликты и скандалы (сотрудники, курьеры, клиенты), проверки и штрафы, изменения условий доставки,
партнёрства, кадровые новости, проблемы с качеством.
Неважное: общая реклама без новости, повторы, посторонние темы.
Ответь ТОЛЬКО JSON-массивом без пояснений и без markdown:
[{"id": <номер>, "relevant": true/false, "company": "<название или 'рынок'>",
  "category": "<лояльность|акции|цены|точки|конфликт|кадры|доставка|качество|другое>",
  "importance": <1-5>, "summary": "<1-2 предложения по-русски, только факты из текста>"}]
Не выдумывай факты, которых нет в тексте."""


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_gemini_model_cache = {}


def _gemini_headers():
    return {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}


def _gemini_generate(model, prompt):
    r = requests.post(
        f"{GEMINI_BASE}/models/{model}:generateContent",
        headers=_gemini_headers(),
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 4000,
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _gemini_candidates():
    """Спрашиваем у Google, какие модели доступны этому ключу, и берём быстрые (flash)."""
    r = requests.get(f"{GEMINI_BASE}/models?pageSize=200", headers=_gemini_headers(), timeout=30)
    r.raise_for_status()
    bad = ["image", "tts", "live", "audio", "embedding", "vision", "robotics", "computer"]
    names = []
    for m in r.json().get("models", []):
        name = m.get("name", "").replace("models/", "")
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        if "flash" not in name or any(w in name for w in bad):
            continue
        names.append(name)
    # сначала стабильные, без preview/exp; внутри группы — новые версии выше, lite в конце
    def rank(n):
        return (("preview" in n or "exp" in n), "lite" in n, [-ord(c) for c in n])
    return sorted(names, key=rank)


def call_llm(cfg, prompt):
    """Возвращает текст ответа модели или None, если ключа нет."""
    provider = cfg.get("provider", "gemini")
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        if "model" in _gemini_model_cache:
            return _gemini_generate(_gemini_model_cache["model"], prompt)
        tried = []
        first = cfg.get("gemini_model")
        if first:
            try:
                text = _gemini_generate(first, prompt)
                _gemini_model_cache["model"] = first
                return text
            except requests.HTTPError as e:
                tried.append(f"{first}: {e.response.status_code}")
        last_err = None
        for name in _gemini_candidates()[:4]:
            if name == first:
                continue
            try:
                text = _gemini_generate(name, prompt)
                _gemini_model_cache["model"] = name
                print(f"[info] Использую модель Gemini: {name}")
                return text
            except requests.HTTPError as e:
                tried.append(f"{name}: {e.response.status_code}")
                last_err = e
        raise RuntimeError("ни одна модель Gemini не ответила (" + "; ".join(tried) + ")")
    if provider == "claude" and os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic  # платный вариант, нужен pip install anthropic

        resp = Anthropic().messages.create(
            model=cfg.get("claude_model", "claude-sonnet-5-5"),
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    return None


# Запасной режим без ИИ: простые ключевые слова
CATEGORY_WORDS = {
    "лояльность": ["лояльн", "кешбэк", "кэшбэк", "бонус", "накопител", "кэшбек", "баллы"],
    "акции": ["акци", "скидк", "промокод", "подарок", "бесплатн"],
    "цены": ["цен", "подорож", "подешев", "стоимост"],
    "точки": ["откры", "закры", "новый филиал", "новая точка", "филиал"],
    "конфликт": ["скандал", "конфликт", "поссор", "драка", "жалоб", "отравлен"],
    "кадры": ["сотрудник", "увольн", "директор", "назначен", "вакансии"],
    "доставка": ["курьер", "доставк"],
    "качество": ["проверк", "штраф", "санэпид", "качество"],
}


GEO_WORDS = ["ташкент", "узбекистан", "uzbekistan", "tashkent", "toshkent", "o'zbekiston"]
FOOD_WORDS = [
    "ресторан", "кафе", "еда", "еды", "общепит", "фастфуд", "пицц", "суши", "ролл",
    "бургер", "лаваш", "шаурм", "курьер", "доставк", "кофейн",
]


def _has(low, words):
    # совпадение только с начала слова, чтобы «цен» не находилось внутри «Автоцентра»
    return any(re.search(r"\b" + re.escape(w), low) for w in words)


def analyze_keywords(cfg, items_chunk):
    out = []
    companies = cfg.get("companies", [])
    for it in items_chunk:
        low = (it["title"] + " " + it["text"]).lower()
        company = next((c for c in companies if c.lower() in low), None)
        # без названия компании берём только местные новости про еду и доставку
        if not company and not (_has(low, GEO_WORDS) and _has(low, FOOD_WORDS)):
            continue
        cats = [c for c, words in CATEGORY_WORDS.items() if _has(low, words)]
        if not cats:
            continue
        out.append(
            {
                "company": company or "рынок",
                "category": cats[0],
                "importance": min(5, 1 + len(cats) + (2 if company else 0)),
                "summary": it["title"][:200],
                "url": it["url"],
                "source": it["source"],
            }
        )
    return out


def analyze(cfg, items):
    if not items:
        return []
    ids = list(items.keys())
    results = []
    batch_size = 30
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        chunk_items = [items[i] for i in chunk]
        lines = [
            f"[{n}] ({it['source']}) {it['title']}\n{it['text']}"
            for n, it in enumerate(chunk_items)
        ]
        parsed = None
        try:
            raw = call_llm(cfg, "\n\n".join(lines))
            if raw:
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
        except Exception as e:
            print(f"[warn] ИИ недоступен ({e}), использую ключевые слова", file=sys.stderr)
        if parsed is None:
            results += analyze_keywords(cfg, chunk_items)
            continue
        for r in parsed:
            try:
                if r.get("relevant"):
                    it = chunk_items[int(r["id"])]
                    results.append({**r, "url": it["url"], "source": it["source"]})
            except (KeyError, IndexError, ValueError, TypeError):
                continue
    results.sort(key=lambda r: -int(r.get("importance", 1)))
    return results[: cfg.get("max_items_in_report", 15)]


# ---------- отчёт ----------

def build_report(results):
    if not results:
        return "Ресторанные новости: за сутки ничего важного не нашёл."
    lines = ["Ресторанные новости за сутки", ""]
    for r in results:
        stars = "★" * int(r.get("importance", 1))
        lines.append(f"{stars} {r.get('company', '')} · {r.get('category', '')}")
        lines.append(r.get("summary", ""))
        lines.append(f"{r.get('source', '')}: {r.get('url', '')}")
        lines.append("")
    return "\n".join(lines).strip()


# ---------- агент 3: отправщик ----------

def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return False
    # лимит Telegram — 4096 символов, режем по абзацам
    parts, cur = [], ""
    for block in text.split("\n\n"):
        if len(cur) + len(block) + 2 > 3800:
            parts.append(cur)
            cur = ""
        cur += block + "\n\n"
    parts.append(cur)
    for part in parts:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": part.strip(),
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        r.raise_for_status()
    return True


def send_email(text):
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    to = os.getenv("EMAIL_TO")
    if not (host and user and password and to):
        return False
    msg = MIMEText(text, "plain", "utf-8")
    msg["Subject"] = "Ресторанные новости за сутки"
    msg["From"] = user
    msg["To"] = to
    with smtplib.SMTP_SSL(host, 465, context=ssl.create_default_context()) as s:
        s.login(user, password)
        s.send_message(msg)
    return True


# ---------- запуск ----------

def main():
    cfg = load_config()
    seen = load_seen()
    items = collect(cfg)
    new_items = {k: v for k, v in items.items() if k not in seen}
    print(f"Собрано: {len(items)}, новых: {len(new_items)}")

    results = analyze(cfg, new_items)
    report = build_report(results)
    print(report)

    sent_tg = send_telegram(report)
    sent_mail = send_email(report)
    if not (sent_tg or sent_mail):
        print("[warn] Ни Telegram, ни почта не настроены, отчёт только в логе.")

    # помечаем как просмотренные только после попытки отправки
    save_seen(seen | set(new_items.keys()))


if __name__ == "__main__":
    main()
