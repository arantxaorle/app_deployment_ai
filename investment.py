
"""
Investment Assistant 
------------------------------------------------
Features:
- RAG with embeddings, chunking, FAISS vectorstore
- 3+ domain functions: fetch_price_yf, fetch_financial_news, analyze_portfolio, get_historical_return, suggest_investments
- Input validation, error handling, logging
- Rate limiting and API key management via Streamlit secrets or environment variable
- Streamlit UI: shows prices, analysis, RAG answers, sources, and progress indicators
Setup:
1. Put your OpenAI API key into Streamlit secrets as OPENAI_API_KEY (see .streamlit/secrets.toml)
2. Install dependencies:
   pip install streamlit yfinance requests beautifulsoup4 langchain openai faiss-cpu
3. Run:
   streamlit run investment.py
"""

import os
import re
import time
import logging
import json
from io import StringIO
import streamlit as st
from dotenv import load_dotenv
import yfinance as yf
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
from bs4 import BeautifulSoup
from functools import wraps
from typing import Dict, List, Tuple, Optional
from streamlit_autorefresh import st_autorefresh
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------------------
# Configuration & Secrets
# ---------------------------
st.set_page_config(page_title="Investment Assistant", layout="wide")
st_autorefresh(interval=60 * 1000, key="data_refresh")

load_dotenv()

# ---------------------------
# Get API key
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key missing. Add it to Streamlit secrets or environment variable.")
    st.stop()

KB_PATH = Path("finance_guide.txt")  
FAISS_INDEX_DIR = Path("faiss_index")
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# ---------------------------
# Streamlit UI
# ---------------------------


with st.sidebar:
    st.header("Settings")
    st.write(f"OpenAI key: {'present' if OPENAI_API_KEY else 'missing'}")
    top_n = st.number_input("Number of RAG sources to display", min_value=1, max_value=10, value=4)
    tracked_input = st.text_input("Symbols to track (comma separated)", value="AAPL,MSFT,GOOGL,AMZN,TSLA,SPY")

    if "language" not in st.session_state:
        st.session_state.language = "English"

    st.session_state.language = st.selectbox(
        "🌐 Select Language / Seleccione idioma",
        ["English", "Español"],
        index=["English", "Español"].index(st.session_state.language)
    )

language = st.session_state.language

TEXTS = {
    "English": {
        "title": "📈 Investment Assistant",
        "intro": "An intelligent assistant for stock prices, portfolio analysis, and RAG-based answers. *Not financial advice.*",
        "market_prices": "📊 Market Prices",
        "portfolio": "💼 Portfolio Analysis",
        "ask_kb": "💬 Ask the Knowledge Base (RAG)",
        "export": "💾 Export Conversation",
        "fetch_news": "📰 Financial News",
        "chart": "📉 Price History Visualization"
    },
    "Español": {
        "title": "📈 Asistente de Inversiones",
        "intro": "Un asistente inteligente para precios de acciones, análisis de portafolio y respuestas basadas en RAG. *No es asesoramiento financiero.*",
        "market_prices": "📊 Precios del Mercado",
        "portfolio": "💼 Análisis de Portafolio",
        "ask_kb": "💬 Preguntar a la Base de Conocimientos (RAG)",
        "export": "💾 Exportar Conversación",
        "fetch_news": "📰 Noticias Financieras",
        "chart": "📉 Visualización del Historial de Precios"
    }
}

#st.title("📈 Investment Assistant")
st.title(TEXTS[language]["title"])
#st.write("An intelligent assistant for stock prices, portfolio analysis, and RAG-based answers. *Not financial advice.*")
st.write(TEXTS[language]["intro"])

# ---------------------------
# Logging setup
# ---------------------------


# ---------------------------
# Logging setup
# ---------------------------
LOGFILE = "investment_assistant.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOGFILE), logging.StreamHandler()]
)
logger = logging.getLogger("investment_assistant")

logger.info("Starting Investment Assistant app")

# ---------------------------
# Rate limiting 
# ---------------------------
def rate_limited(calls_per_minute: int = 30):
    """Simple in-memory rate limiter per function."""
    interval = 60.0 / calls_per_minute
    def decorator(func):
        last_time = {"t": 0.0}
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            elapsed = now - last_time["t"]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            result = func(*args, **kwargs)
            last_time["t"] = time.time()
            return result
        return wrapper
    return decorator

# ---------------------------
# Validation helpers
# ---------------------------
SYMBOL_RE = re.compile(r"^[A-Z\.]{1,6}$")

def validate_symbol(symbol: str) -> bool:
    return bool(SYMBOL_RE.match(symbol.strip().upper()))

def parse_portfolio_input(text: str) -> Dict[str, int]:
    """Parse user portfolio input like 'AAPL=10, MSFT=5'."""
    allocation = {}
    for entry in text.split(","):
        if not entry.strip():
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid entry (missing '='): {entry}")
        sym, shares = entry.split("=")
        sym = sym.strip().upper()
        if not validate_symbol(sym):
            raise ValueError(f"Invalid symbol format: {sym}")
        shares_int = int(shares.strip())
        if shares_int < 0:
            raise ValueError(f"Shares must be non-negative: {entry}")
        allocation[sym] = shares_int
    return allocation

# ---------------------------
# RAG setup: Load KB, chunk, embed, store in FAISS
# ---------------------------
@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(kb_path: str, openai_api_key: str) -> FAISS:
    try:
        logger.info("Initializing embeddings and vectorstore")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        index_file = FAISS_INDEX_DIR / "index.faiss"
        meta_file = FAISS_INDEX_DIR / "index.pkl"

        # Try to load existing FAISS index
        if index_file.exists() and meta_file.exists():
            try:
                vs = FAISS.load_local(str(FAISS_INDEX_DIR), embeddings)
                logger.info("Loaded existing FAISS index from disk.")
                return vs
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}, rebuilding...")

        # Otherwise build from KB file
        if not Path(kb_path).exists():
            with open(kb_path, "w", encoding="utf-8") as f:
                f.write("Investing basics:\n- Diversify assets\n- Dollar-cost averaging\n- Long-term horizon\n")

        loader = TextLoader(kb_path, encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} text chunks from KB")

        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(str(FAISS_INDEX_DIR))
        logger.info("FAISS index built and saved.")
        return vs
    except Exception as e:
        logger.exception("Error building/loading vectorstore")
        raise

vectorstore = build_or_load_vectorstore(str(KB_PATH), OPENAI_API_KEY)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0.0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

# ---------------------------
# Domain functions
# ---------------------------

@rate_limited(calls_per_minute=60)
def fetch_price_yf(symbol: str) -> Optional[float]:
    symbol = symbol.strip().upper()
    if not validate_symbol(symbol):
        return None
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        return None
    except Exception as e:
        logger.exception(f"fetch_price_yf error: {e}")
        return None

@rate_limited(calls_per_minute=30)
def get_historical_return(symbol: str, period: str = "1y") -> Optional[Dict[str, float]]:
    symbol = symbol.strip().upper()
    if not validate_symbol(symbol):
        return None
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty or len(data["Close"]) < 2:
            return None
        start = float(data["Close"].iloc[0])
        end = float(data["Close"].iloc[-1])
        pct = ((end - start) / start) * 100.0 if start != 0 else 0.0
        return {"symbol": symbol, "start": start, "end": end, "pct_return": pct}
    except Exception as e:
        logger.exception(f"get_historical_return error: {e}")
        return None

@rate_limited(calls_per_minute=10)
def fetch_financial_news(symbols: List[str], max_articles: int = 5) -> Dict[str, List[Tuple[str, str]]]:
    results = {}
    session = requests.Session()
    headers = {"User-Agent": "investment-assistant-bot/1.0 (+https://example.com)"}
    for sym in symbols:
        if not validate_symbol(sym):
            results[sym] = []
            continue
        try:
            url = f"https://finance.yahoo.com/quote/{sym}"
            r = session.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(r.text, "html.parser")
            items = []
            for a in soup.select("a")[:200]:
                text = a.get_text(strip=True)
                href = a.get("href", "")
                if len(text) > 30 and ("news" in href.lower() or "article" in href.lower()):
                    full = href if href.startswith("http") else f"https://finance.yahoo.com{href}"
                    items.append((text, full))
                if len(items) >= max_articles:
                    break
            results[sym] = items
        except Exception as e:
            logger.exception(f"Error fetching news for {sym}: {e}")
            results[sym] = []
    return results

def suggest_investments(prices: Dict[str, Optional[float]]) -> List[str]:
    suggestions = []
    for sym, price in prices.items():
        if price is None:
            suggestions.append(f"{sym}: Price unavailable — check symbol.")
            continue
        if price < 50:
            suggestions.append(f"{sym}: Low price (${price:.2f}). Might be worth a long-term look.")
        elif price < 200:
            suggestions.append(f"{sym}: Moderate price (${price:.2f}). Possibly good opportunity.")
        elif price > 500:
            suggestions.append(f"{sym}: High price (${price:.2f}). Consider reviewing valuation.")
        else:
            suggestions.append(f"{sym}: Price ${price:.2f}. Continue monitoring.")
    return suggestions

def analyze_portfolio(prices: Dict[str, Optional[float]], allocation: Dict[str, int]) -> Dict[str, float]:
    breakdown = {}
    total = 0.0
    for sym, shares in allocation.items():
        p = prices.get(sym)
        if p is None:
            breakdown[sym] = 0.0
            continue
        val = p * shares
        breakdown[sym] = val
        total += val
    return {"total_value": total, "breakdown": breakdown}

# ---------------------------
# RAG query wrapper
# ---------------------------
#def query_knowledge_base(query: str) -> Tuple[str, List[dict]]:
#    try:
#        result = qa_chain({"query": query})
#        answer = result.get("result") or result.get("answer") or ""
#        source_docs = result.get("source_documents") or []
#        sources = [{"content": getattr(d, "page_content", str(d)), "metadata": getattr(d, "metadata", {})} for d in source_docs]
#        return answer, sources
#    except Exception as e:
#        logger.exception("RAG query failed")
#        return f"Error answering question: {e}", []

def query_knowledge_base(query: str, language: str = "English") -> Tuple[str, List[dict]]:
    try:
        lang_instruction = (
            "Please answer in English." if language == "English"
            else "Por favor, responde en español con lenguaje financiero claro y natural."
        )
        full_query = f"{query}\n\n{lang_instruction}"
        result = qa_chain({"query": full_query})
        answer = result.get("result") or result.get("answer") or ""
        source_docs = result.get("source_documents") or []
        sources = [
            {"content": getattr(d, "page_content", str(d)), "metadata": getattr(d, "metadata", {})}
            for d in source_docs
        ]
        return answer, sources
    except Exception as e:
        logger.exception("RAG query failed")
        return f"Error answering question: {e}", []





# ---------------------------
# Interactive Help Chatbot
# ---------------------------
with st.sidebar.expander("🧠 Help & Guide"):
    st.markdown("""
    **Welcome to the Investment Assistant!**  
    Here's what you can do:
    - **📊 Prices:** View live stock prices.
    - **💼 Portfolio:** Analyze your holdings.
    - **📰 News:** Get the latest financial news.
    - **💬 Ask KB:** Query the investment knowledge base.
    - **💾 Export:** Save your conversation.
    """)
    
    help_question = st.text_input("Ask for help (e.g. 'How do I analyze my portfolio?')")
    if st.button("Ask Help"):
        if "portfolio" in help_question.lower():
            st.info("Go to **💼 Portfolio Analysis** and enter your stocks in the format: SYMBOL=SHARES.")
        elif "price" in help_question.lower():
            st.info("Use the **📊 Market Prices** section to see live data from Yahoo Finance.")
        elif "rag" in help_question.lower() or "kb" in help_question.lower():
            st.info("Type any investment-related question in the **💬 Ask the Knowledge Base** box.")
        else:
            st.info("Try asking about 'portfolio', 'prices', or 'knowledge base'.")



# ---------------------------
# Conversation History 
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  

def add_to_history(query: str, answer: str):
    st.session_state.history.append({"query": query, "answer": answer})

# ---------------------------
# Parse symbols
# ---------------------------
TRACKED_SYMBOLS = [s.strip().upper() for s in tracked_input.split(",") if validate_symbol(s.strip().upper())]

#st.subheader("📊 Market Prices")
st.subheader(TEXTS[language]["market_prices"])
prices = {}
progress = st.progress(0)
cols = st.columns(min(3, max(1, len(TRACKED_SYMBOLS))))

for i, sym in enumerate(TRACKED_SYMBOLS):
    price = fetch_price_yf(sym)
    prices[sym] = price
    with cols[i % len(cols)]:
        if price:
            st.metric(sym, f"${price:.2f}")
        else:
            st.write(f"{sym}: ❌ No data")
    progress.progress(int((i + 1) / len(TRACKED_SYMBOLS) * 100))
progress.empty()

st.subheader("📉 Price History Visualization")

col1, col2 = st.columns(2)
with col1:
    selected_sym = st.selectbox("Select a stock to visualize", TRACKED_SYMBOLS)
with col2:
    selected_period = st.selectbox("Select time period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)

if selected_sym:
    try:
        data = yf.Ticker(selected_sym).history(period=selected_period)
        if not data.empty:
            st.line_chart(data["Close"], use_container_width=True)
        else:
            st.info("No price history available for this period.")
    except Exception as e:
        st.error(f"Error loading chart: {e}")

# ---------------------------
# Export Stock Prices
# ---------------------------
st.subheader("💾 Export Stock Prices")


df_prices = pd.DataFrame(list(prices.items()), columns=["Symbol", "Price"])
csv_data = df_prices.to_csv(index=False)
st.download_button("📂 Download CSV", csv_data, "stock_prices.csv", "text/csv")


if st.button("📄 Generate PDF Report"):
    pdf_path = "stock_prices.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Stock Prices Report")
    y -= 30

    c.setFont("Helvetica", 12)
    for sym, price in prices.items():
        c.drawString(50, y, f"{sym}: ${price:.2f}" if price else f"{sym}: No data")
        y -= 20
        if y < 50:  # add page break
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50

    c.save()
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF", f, "stock_prices.pdf", "application/pdf")


st.subheader("💡 Quick Suggestions")
for s in suggest_investments(prices):
    st.write("- " + s)



st.subheader("📰 Financial News")
if st.button("Fetch Latest News"):
    with st.spinner("Fetching news..."):
        news = fetch_financial_news(TRACKED_SYMBOLS, max_articles=3)
        for sym, items in news.items():
            st.write(f"**{sym}**")
            if not items:
                st.write("_No news found._")
            for title, url in items:
                st.write(f"- [{title}]({url})")

st.subheader("📈 Historical Returns")
hist_sym = st.selectbox("Select symbol", TRACKED_SYMBOLS)
period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
if st.button("Calculate Return"):
    with st.spinner("Calculating..."):
        hr = get_historical_return(hist_sym, period)
        if hr:
            st.write(f"{hist_sym} ({period}): ${hr['start']:.2f} → ${hr['end']:.2f} ({hr['pct_return']:.2f}%)")
        else:
            st.write("Not enough data.")

st.subheader("💼 Portfolio Analysis")
portfolio_input = st.text_area("Enter your portfolio (format: SYMBOL=SHARES, e.g., AAPL=10, MSFT=5)", value="AAPL=10, MSFT=5")
if st.button("Analyze Portfolio"):
    try:
        allocation = parse_portfolio_input(portfolio_input)
    except Exception as e:
        st.error(f"Invalid input: {e}")
        allocation = {}
    if allocation:
        with st.spinner("Analyzing..."):
            for sym in allocation.keys():
                if prices.get(sym) is None:
                    prices[sym] = fetch_price_yf(sym)
            analysis = analyze_portfolio(prices, allocation)
            st.success(f"Total Portfolio Value: ${analysis['total_value']:.2f}")
            st.table({k: f"${v:.2f}" for k, v in analysis["breakdown"].items()})

st.subheader("💬 Ask the Knowledge Base (RAG)")
user_query = st.text_input("Ask any investment-related question:")

if st.button("Ask KB"):
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Searching knowledge base..."):
            answer, sources = query_knowledge_base(user_query, language)
            st.markdown("**Answer:**")
            st.write(answer)
            st.markdown("---")
            st.markdown("**Relevant Sources:**")
            for i, sdoc in enumerate(sources[:top_n]):
                excerpt = sdoc["content"][:600].strip().replace("\n", " ")
                st.write(f"**Source #{i+1}:** {excerpt}...")
                #if sdoc.get("metadata"):
                #    st.write(f"_metadata:_ {sdoc['metadata']}
            add_to_history(user_query, answer)
        
        with st.expander("🔍 View RAG Process Explanation"):
            st.markdown("""
            **Retrieval-Augmented Generation (RAG)** steps used here:
            1. **Retrieve**: Search your knowledge base for the most relevant text chunks.
            2. **Augment**: Feed those chunks into GPT-4 as context.
            3. **Generate**: Produce a final answer grounded in those sources.
            
            This ensures factual, domain-specific answers rather than purely generative text.
            """)

st.subheader("💾 Export Conversation")

if st.session_state.history:
    export_format = st.selectbox("Select export format", ["JSON", "Text"])
    if st.button("Export"):
        if export_format == "JSON":
            buffer = StringIO()
            json.dump(st.session_state.history, buffer, indent=2)
            st.download_button("Download JSON", buffer.getvalue(), "conversation.json", "application/json")
        else:
            text_content = "\n\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in st.session_state.history])
            st.download_button("Download Text", text_content, "conversation.txt", "text/plain")
else:
    st.info("No conversation yet — ask a question in the RAG section to start.")

with st.expander("View Knowledge Base Content (first 2k chars)"):
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            st.text(f.read(2000))
    except Exception as e:
        st.write("KB not found or read error.")

st.caption(f"Last updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
st.caption("Made with care — and a dash of humor. This is not financial advice.")

logger.info("App rendered successfully")
