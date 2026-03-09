"""
Streamlit Frontend — Document upload, listing, and RAG querying.
Bonus +5% component.
"""
import streamlit as st
import requests
import os
from datetime import datetime

API_URL = os.getenv("FLASK_API_URL", "http://localhost:5000")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 RAG Document Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📄 RAG Document Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload documents, ask questions, get grounded answers with citations</div>', unsafe_allow_html=True)


# ── Sidebar: System Health ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")
    if st.button("🔄 Check Health"):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=10)
            health = resp.json()
            status = health.get("status", "unknown")
            if status == "healthy":
                st.success("✅ All services healthy")
            else:
                st.warning(f"⚠️ Status: {status}")
            for svc, s in health.get("services", {}).items():
                icon = "✅" if s == "ok" else "❌"
                st.write(f"{icon} **{svc}**: {s}")
        except Exception as e:
            st.error(f"❌ Cannot reach API: {e}")

    st.divider()
    st.header("📋 Settings")
    top_k = st.slider("Number of sources (top_k)", 1, 10, 5)
    strategy = st.selectbox("Chunking strategy", ["fixed_size", "recursive"])


# ── Tab Layout ───────────────────────────────────────────────────────────────
tab_upload, tab_docs, tab_query = st.tabs(["📤 Upload", "📚 Documents", "💬 Ask Questions"])

# ── Tab 1: Upload ────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
        help="Maximum supported formats: PDF and DOCX"
    )

    if uploaded_file and st.button("🚀 Upload & Index", type="primary"):
        with st.spinner("Uploading, parsing, chunking, and indexing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {"strategy": strategy}
                resp = requests.post(f"{API_URL}/documents", files=files, data=data, timeout=120)

                if resp.status_code == 201:
                    result = resp.json()
                    st.success(f"✅ **{result['filename']}** uploaded successfully!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Document ID", result["id"][:8] + "...")
                    col2.metric("Chunks Created", result["chunks"])
                    col3.metric("Strategy", result["strategy"])
                elif resp.status_code == 422:
                    st.error(f"⚠️ {resp.json().get('error', 'Processing error')}")
                else:
                    st.error(f"❌ Error: {resp.json().get('error', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API. Is the Flask server running?")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ── Tab 2: Documents ────────────────────────────────────────────────────────
with tab_docs:
    st.subheader("Uploaded Documents")
    if st.button("🔄 Refresh List"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/documents", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            docs = data.get("documents", [])
            if not docs:
                st.info("📭 No documents uploaded yet. Go to the Upload tab to add documents.")
            else:
                st.write(f"**{len(docs)} document(s) indexed**")
                for doc in docs:
                    with st.expander(f"📄 {doc['filename']}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        col1.write(f"**ID:** `{doc['id'][:12]}...`")
                        col2.write(f"**Chunks:** {doc['chunk_count']}")
                        col3.write(f"**Strategy:** {doc['chunk_strategy']}")
                        st.write(f"**Uploaded:** {doc['upload_date']}")
                        st.write(f"**Size:** {doc.get('file_size', 0) / 1024:.1f} KB")

                        if st.button(f"🗑️ Delete", key=f"del_{doc['id']}"):
                            try:
                                del_resp = requests.delete(
                                    f"{API_URL}/documents/{doc['id']}", timeout=10
                                )
                                if del_resp.status_code == 200:
                                    st.success("Document deleted!")
                                    st.rerun()
                                else:
                                    st.error(f"Delete failed: {del_resp.text}")
                            except Exception as e:
                                st.error(f"Error: {e}")
        else:
            st.error("Failed to fetch documents")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Cannot connect to API. Is the server running?")
    except Exception as e:
        st.error(f"Error: {e}")

# ── Tab 3: Query ─────────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Ask a Question")

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**[Source {i}]** `{src['filename']}` "
                            f"(chunk {src['chunk_index']}, score: {src['score']:.4f})"
                        )
                        st.markdown(f"> {src['text'][:300]}...")

    # Chat input
    if question := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/query",
                        json={"question": question, "top_k": top_k},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        answer = result.get("answer", "No answer generated.")
                        sources = result.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"📎 {len(sources)} Source(s)", expanded=True):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(
                                        f"**[Source {i}]** `{src['filename']}` "
                                        f"(chunk {src['chunk_index']}, "
                                        f"relevance: {src['score']:.4f})"
                                    )
                                    st.markdown(f"> {src['text'][:300]}...")
                                    st.divider()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    else:
                        error_msg = resp.json().get("error", "Unknown error")
                        st.error(f"❌ {error_msg}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Is the server running?")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
