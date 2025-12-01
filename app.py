import streamlit as st
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain.agents import AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import datetime, os, requests, re, hashlib
from dotenv import load_dotenv
from io import StringIO
import pandas as pd
import time

load_dotenv()

# ================== SUPER ADMIN CONFIGURATION ==================
SUPER_ADMIN_PASSWORD = "admin123"  # Change this to your desired password
SUPER_ADMIN_PASSWORD_HASH = hashlib.sha256(SUPER_ADMIN_PASSWORD.encode()).hexdigest()

# ================== SCHEMA BACKUP SYSTEM ==================
def backup_table_schema(db_path, table_name):
    """Backup current table schema before modifications"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get CREATE TABLE statement
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        result = cursor.fetchone()
        
        if result:
            schema_sql = result[0]
            
            # Get all data from table
            cursor.execute(f"SELECT * FROM {table_name}")
            data = cursor.fetchall()
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            conn.close()
            
            backup_entry = {
                'table_name': table_name,
                'schema_sql': schema_sql,
                'data': data,
                'columns': columns,
                'timestamp': datetime.datetime.now().isoformat(),
                'db_path': db_path
            }
            
            return backup_entry
    except Exception as e:
        return None
    
    return None

def restore_table_schema(backup_entry):
    """Restore table to previous schema state"""
    try:
        conn = sqlite3.connect(backup_entry['db_path'])
        cursor = conn.cursor()
        
        table_name = backup_entry['table_name']
        
        # Drop current table
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}_temp_backup")
        cursor.execute(f"CREATE TABLE {table_name}_temp_backup AS SELECT * FROM {table_name}")
        
        # Drop and recreate original table
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(backup_entry['schema_sql'])
        
        # Restore data if possible
        if backup_entry['data']:
            column_names = [col[1] for col in backup_entry['columns']]
            placeholders = ','.join(['?'] * len(column_names))
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
            
            try:
                cursor.executemany(insert_sql, backup_entry['data'])
            except:
                pass  # If data doesn't fit new schema, skip
        
        # Drop temp backup
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}_temp_backup")
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        return False

def get_added_columns(db_path, table_name, original_columns):
    """Get list of columns added after original schema"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        current_columns = [col[1] for col in cursor.fetchall()]
        conn.close()
        
        original_col_names = [col[1] for col in original_columns]
        added = [col for col in current_columns if col not in original_col_names]
        
        return added
    except:
        return []

# ================== DATABASE SELECTION ==================
if "selected_db" not in st.session_state:
    st.session_state.selected_db = None

if st.session_state.selected_db is None:
    st.set_page_config(page_title="Select Database", page_icon="🗄️")
    st.title("🗄️ Database Selection")
    st.write("Please select a database to work with for this session.")
    
    # Define available databases - You should ensure these files exist
    databases = {
        "Task Manager": "task-manager.db",
        "Hospital": "hospital.db",
        "Sports": "sports.db"
    }
    
    available_dbs = {}
    for name, filepath in databases.items():
        # Check if file exists, or just allow selection for demo purposes
        # if os.path.exists(filepath):
        available_dbs[name] = filepath
    
    st.subheader("Available Databases:")
    
    cols = st.columns(len(available_dbs))
    for idx, (db_name, db_file) in enumerate(available_dbs.items()):
        with cols[idx]:
            st.info(f"**{db_name}**")
            st.caption(f"File: {db_file}")
            if st.button(f"Select {db_name}", key=f"select_{db_name}", use_container_width=True, type="primary"):
                st.session_state.selected_db = db_file
                st.session_state.selected_db_name = db_name
                st.rerun()
    
    st.stop()

# ================== STREAMLIT UI (After DB Selection) ==================
st.set_page_config(page_title="AI Company Chat (DB + RAG)", page_icon="🤖")
st.title(f"🤖 AI Assistant - {st.session_state.selected_db_name}")

st.sidebar.success(f"📁 Database: **{st.session_state.selected_db_name}**")
st.sidebar.caption(f"File: {st.session_state.selected_db}")
st.sidebar.divider()

# ================== SUPER ADMIN STATUS ==================
if "is_super_admin" not in st.session_state:
    st.session_state.is_super_admin = False
if "admin_auth_pending" not in st.session_state:
    st.session_state.admin_auth_pending = False

# Show admin status in sidebar
if st.session_state.is_super_admin:
    st.sidebar.success("🔐 Super Admin Mode Active")
    if st.sidebar.button("🚪 Exit Admin Mode", use_container_width=True):
        st.session_state.is_super_admin = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Super Admin mode deactivated. You now have standard user permissions."
        })
        st.rerun()
else:
    st.sidebar.info("👤 Standard User Mode")

st.sidebar.divider()

# Load API keys
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# ================== PREDEFINED QUICK ACTIONS BY DATABASE ==================
PREDEFINED_QUERIES = {
    "task-manager.db": {
        "📋 Show All Tasks": "SELECT * FROM task",
        "👥 Show All Employees": "SELECT * FROM employee",
        "🏢 Show All Clients": "SELECT * FROM client",
        "⏳ Pending Tasks": "SELECT * FROM task WHERE LOWER(status) = 'pending'",
        "✅ Completed Tasks": "SELECT * FROM task WHERE LOWER(status) = 'completed'",
        "🔥 High Priority Tasks": "SELECT * FROM task WHERE LOWER(priority) = 'high'",
        "🚀 In Progress Tasks": "SELECT * FROM task WHERE LOWER(status) = 'in progress'",
        "📊 Tasks by Employee": "SELECT e.employee_name, COUNT(t.task_id) as task_count FROM employee e LEFT JOIN task t ON e.employee_id = t.employee_id GROUP BY e.employee_id, e.employee_name"
    },
    "hospital.db": {
        "👨‍⚕️ Show All Doctors": "SELECT * FROM doctor",
        "🏥 Show All Patients": "SELECT * FROM patient",
        "📅 Show All Appointments": "SELECT * FROM appointment",
        "⏳ Pending Appointments": "SELECT * FROM appointment WHERE LOWER(status) = 'pending'",
        "✅ Completed Appointments": "SELECT * FROM appointment WHERE LOWER(status) = 'completed'",
        "🔄 In Progress Appointments": "SELECT * FROM appointment WHERE LOWER(status) = 'in progress'",
        "📊 Appointments by Doctor": "SELECT d.doctor_name, COUNT(a.appointment_id) as appointment_count FROM doctor d LEFT JOIN appointment a ON d.doctor_id = a.doctor_id GROUP BY d.doctor_id, d.doctor_name",
        "🩺 Today's Appointments": "SELECT * FROM appointment WHERE appointment_date = date('now')"
    },
    "sports.db": {
        "⚽ Show All Teams": "SELECT * FROM team",
        "🏃 Show All Players": "SELECT * FROM player",
        "🏆 Show All Matches": "SELECT * FROM match",
        "⏳ Pending Matches": "SELECT * FROM match WHERE LOWER(status) = 'pending'",
        "✅ Completed Matches": "SELECT * FROM match WHERE LOWER(status) = 'completed'",
        "🔄 In Progress Matches": "SELECT * FROM match WHERE LOWER(status) = 'in progress'",
        "📊 Players by Team": "SELECT t.team_name, COUNT(p.player_id) as player_count FROM team t LEFT JOIN player p ON t.team_id = p.team_id GROUP BY t.team_id, t.team_name",
        "🎯 Recent Match Results": "SELECT * FROM match WHERE status = 'Completed' ORDER BY match_date DESC LIMIT 5"
    }
}

# ================== SIDEBAR QUICK ACTIONS ==================
st.sidebar.header("⚡ Quick Actions")

current_db_queries = PREDEFINED_QUERIES.get(st.session_state.selected_db, {})

if current_db_queries:
    st.sidebar.subheader("📊 Quick Queries")
    
    for button_label, sql_query in current_db_queries.items():
        if st.sidebar.button(button_label, use_container_width=True):
            st.session_state.pending_query = sql_query
            st.session_state.is_direct_sql = True

st.sidebar.divider()

st.sidebar.subheader("🧹 Chat Management")
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Chat cleared! Ask me about data in **{st.session_state.selected_db_name}** or PDFs.\n\n"
                    "💡 Use the Quick Actions in the sidebar for common queries!"
    }]
    if "store" in st.session_state:
        st.session_state.store = {}
    if "sql_chat_history" in st.session_state:
        st.session_state.sql_chat_history = []
    st.rerun()

# ================== INITIALIZE SQL CHAT HISTORY ==================
if "sql_chat_history" not in st.session_state:
    st.session_state.sql_chat_history = []

# ================== INITIALIZE SCHEMA BACKUP HISTORY ==================
if "schema_backups" not in st.session_state:
    st.session_state.schema_backups = {}  # Format: {table_name: [backup_entry1, backup_entry2, ...]}

if "show_schema_manager" not in st.session_state:
    st.session_state.show_schema_manager = False

def get_sql_context():
    """Build context string from recent SQL chat history with resolved IDs"""
    if not st.session_state.sql_chat_history:
        return ""
    
    # Get last 5 interactions for context
    recent_history = st.session_state.sql_chat_history[-5:]
    context_parts = []
    
    for item in recent_history:
        context_parts.append(f"User asked: {item['query']}")
        if 'result_summary' in item:
            context_parts.append(f"Result: {item['result_summary']}")
        # ADD THIS: Track resolved entity IDs
        if 'resolved_ids' in item:
            context_parts.append(f"Resolved IDs: {item['resolved_ids']}")
    
    return "\n".join(context_parts)

# ================== HELPER: FIND ENTITY BY PARTIAL NAME ==================
def find_entity_by_name(table_name, name_column, search_name):
    """Find entities by partial name match and return options if multiple found"""
    try:
        conn = sqlite3.connect(st.session_state.selected_db)
        cursor = conn.cursor()
        
        # Get the ID column name (usually table_name + _id)
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        id_column = columns[0][1]  # First column is usually the ID
        
        # Search for matches (case-insensitive, partial match)
        query = f"""
        SELECT {id_column}, {name_column} 
        FROM {table_name} 
        WHERE LOWER({name_column}) LIKE LOWER('%{search_name}%')
        """
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        return results
    except Exception as e:
        return []

# ================== LLM SETUP ==================
llm = ChatGroq(api_key=api_key, model="qwen/qwen3-32b", temperature=0, streaming=True)
 
def configure_db():
    """Configure database connection based on selected DB"""
    # Fix for path handling
    try:
        if os.path.isabs(st.session_state.selected_db):
             dbfilepath = st.session_state.selected_db
        else:
             dbfilepath = (Path(__file__).parent / st.session_state.selected_db).absolute()
        
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=rw", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None
 
db = configure_db()
if db:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
else:
    st.error("Could not initialize database toolkit")
    st.stop()
 
def get_sql_prefix():
    base_prefix = """You interact with a SQL database. Create SQLite queries and return answers.
Limit to 10 results unless specifically asked for more. Query only relevant columns.

**CRITICAL: SINGLE-QUERY OPTIMIZATION**
When user asks "show tasks for Bob":
- DO NOT make separate queries to find ID first
- USE A SINGLE JOIN QUERY IMMEDIATELY:
  SELECT t.* FROM task t 
  JOIN employee e ON t.employee_id = e.employee_id 
  WHERE LOWER(e.employee_name) LIKE LOWER('%bob%')
- This is faster and avoids redundant steps

IMPORTANT QUERY RULES:
- ALWAYS use LOWER() for case-insensitive text comparisons
- For partial matches, use: WHERE LOWER(column) LIKE LOWER('%search%')
- When searching across tables, ALWAYS use JOINs in a SINGLE query

**AVOID MULTI-STEP QUERIES:**
❌ WRONG (2 queries):
1. SELECT employee_id FROM employee WHERE name LIKE '%bob%' 
2. SELECT * FROM task WHERE employee_id = 2

✅ CORRECT (1 query):
SELECT t.* FROM task t 
JOIN employee e ON t.employee_id = e.employee_id 
WHERE LOWER(e.employee_name) LIKE LOWER('%bob%')

CONTEXT AWARENESS:
Recent conversation history:
{context}
"""
    
    if st.session_state.is_super_admin:
        admin_suffix = """
SUPER ADMIN MODE - EXTENDED PERMISSIONS:
- ALLOWED: SELECT, UPDATE, DELETE, ALTER (modify table structure, add/rename columns)
- COLUMN MANAGEMENT: Can drop columns that were added by super admin
- STRICTLY FORBIDDEN: DROP/TRUNCATE entire tables, INSERT directly
- You can modify table schemas (ALTER TABLE) but cannot delete entire tables
- When altering tables, ALWAYS create a backup first by calling: "backup schema for [table_name]"
- Before ALTER operations, remind user that backups are available

IMPORTANT: When a super admin adds a new column or modifies schema:
1. Automatically suggest creating a backup
2. Track schema changes for potential rollback
3. Allow dropping only newly added columns (not original schema columns)

INSERT operations still require the guided form interface.
"""
    else:
        admin_suffix = """
SECURITY:
- NEVER: DROP, TRUNCATE, ALTER
- ALLOW ONLY: SELECT (query data), UPDATE (modify existing records), DELETE (remove records)
- STRICTLY FORBIDDEN: INSERT operations - Users must use the guided form interface

When users ask to add/create/insert new records, respond: "Please use the 'add new [table]' command to open the guided form."
"""

    footer = """
Refuse destructive operations (DROP/TRUNCATE) and INSERT politely. Double-check queries before executing.

💡 IMPORTANT: When returning query results, ALWAYS format them as a proper Markdown table with this exact format:
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| Value1  | Value2  | Value3  |

Make sure to include the header separator line with dashes.
"""
    
    context = get_sql_context()
    return base_prefix.format(context=context) + admin_suffix + footer


def create_sql_agent_with_context():
    """Create SQL agent with current context"""
    return create_sql_agent(
        llm=llm, 
        toolkit=toolkit, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=get_sql_prefix(), 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=30, 
        max_execution_time=200, 
        early_stopping_method="generate"
    )

# ================== RAG SETUP ==================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
 
def build_rag_chain(selected_db):
    """Build RAG chain from PDFs in folder named after the selected DB"""
    base_folder = "pdfs"
    db_name = os.path.splitext(os.path.basename(selected_db))[0]
    folder_path = os.path.join(base_folder, db_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        # st.info(f"📂 Created folder `{folder_path}`. Please add PDFs there for RAG.")
        return None

    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()

    if not documents:
        # st.warning(f"⚠️ No PDFs found in `{folder_path}` for '{db_name}' database.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        return None
        
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Enhanced system prompt for strict RAG adherence
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, rephrase the question to be standalone if needed. Otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that STRICTLY answers questions based ONLY on the provided PDF context.

CRITICAL RULES:
1. ONLY use information from the PDF context provided below
2. If the answer is not in the context, say: "I cannot find this information in the provided PDFs."
3. NEVER use your general knowledge or training data
4. NEVER make assumptions beyond what's explicitly stated in the PDFs
5. Always cite which part of the document your answer comes from when possible

Context from PDF documents:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if "store" not in st.session_state:
            st.session_state.store = {}
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, 
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain

# ================== Load RAG for selected DB ==================
selected_db = st.session_state.selected_db

if "rag_chain" not in st.session_state or st.session_state.get("rag_for_db") != selected_db:
    with st.spinner(f"Loading RAG for {selected_db}..."):
        st.session_state.rag_chain = build_rag_chain(selected_db)
        st.session_state.rag_for_db = selected_db
        st.session_state.has_pdfs = st.session_state.rag_chain is not None

# ================== Chat History ==================
if "messages" not in st.session_state:
    pdf_note = "\n- Ask questions about PDFs in the './pdfs' folder" if st.session_state.get('has_pdfs', False) else ""
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Hello! I can help you with:\n"
                   f"- Querying data in **{st.session_state.selected_db_name}**{pdf_note}\n\n"
                   "💡 **Quick Commands:**\n"
                   "- 'create new task' - Create a new record\n"
                   "- 'enable super admin' - Activate admin mode (password required)\n"
                   "- Use Quick Actions in sidebar for common queries!"
    }]
 
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "is_direct_sql" not in st.session_state:
    st.session_state.is_direct_sql = False
 
for idx, msg in enumerate(st.session_state.messages[-10:]):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "dataframe" in msg:
            df = pd.DataFrame(msg["dataframe"])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"query_results_{idx}.csv",
                mime="text/csv",
                key=f"download_{idx}_{len(st.session_state.messages)}"
            )
 
# ========== Enhanced Table Parsing Function ==========
def try_render_table(response_text):
    """Parse response and extract table data"""
    try:
        lines = response_text.strip().split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line:
                in_table = True
                cleaned = line.strip()
                if cleaned and not all(c in '|-: ' for c in cleaned.replace('|', '')):
                    table_lines.append(cleaned)
            elif in_table and line.strip() == '':
                break
        
        if len(table_lines) >= 2:
            table_lines = [line for line in table_lines if not all(c in '|-: ' for c in line.replace('|', ''))]
            
            if len(table_lines) >= 1:
                table_str = '\n'.join(table_lines)
                df = pd.read_csv(StringIO(table_str), sep='|', engine='python')
                
                df = df.dropna(axis=1, how='all')
                df.columns = df.columns.str.strip()
                
                if len(df.columns) > 0 and df.columns[0] == '':
                    df = df.iloc[:, 1:]
                if len(df.columns) > 0 and df.columns[-1] == '':
                    df = df.iloc[:, :-1]
                
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                
                if not df.empty and len(df.columns) > 0:
                    return df
    except Exception as e:
        pass
    
    return None

# ========== Execute Direct SQL ==========
def execute_direct_sql(sql_query):
    """Execute SQL query directly and return formatted result"""
    try:
        conn = sqlite3.connect(st.session_state.selected_db)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        if results:
            columns = [description[0] for description in cursor.description]
            
            result = "| " + " | ".join(columns) + " |\n"
            result += "|" + "|".join(["---"] * len(columns)) + "|\n"
            
            for row in results:
                result += "| " + " | ".join(str(v) if v is not None else "" for v in row) + " |\n"
            
            conn.close()
            return result
        else:
            conn.close()
            return "✅ Query executed successfully but returned no results."
            
    except Exception as e:
        return f"⚠️ Error executing query: {str(e)}"

# ========== Super Admin Authentication ==========
def check_super_admin_trigger(query):
    """Check if user is trying to activate super admin mode"""
    patterns = [
        r'enable super admin',
        r'activate admin mode',
        r'become super admin',
        r'super admin mode',
        r'i am super admin',
        r'admin access'
    ]
    
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in patterns)

def show_admin_auth():
    """Show super admin authentication dialog"""
    st.info("🔐 **Super Admin Authentication Required**")
    st.write("Please enter the super admin password to continue:")
    
    password = st.text_input("Password:", type="password", key="admin_password_input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Authenticate", type="primary", use_container_width=True):
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash == SUPER_ADMIN_PASSWORD_HASH:
                st.session_state.is_super_admin = True
                st.session_state.admin_auth_pending = False
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "✅ **Super Admin Mode Activated!**\n\n"
                               "You now have extended permissions:\n"
                               "- Can ALTER table structures\n"
                               "- Can UPDATE and DELETE records\n"
                               "- Cannot DROP or TRUNCATE tables\n"
                               "- Cannot INSERT directly (use forms)\n\n"
                               "⚠️ Use these permissions responsibly!"
                })
                st.rerun()
            else:
                st.error("❌ Incorrect password. Access denied.")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.admin_auth_pending = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Authentication cancelled. Remaining in standard user mode."
            })
            st.rerun()

# ========== Enhanced Name Resolution ==========
def resolve_entity_name(query):
    """Resolve partial names to specific entities with ID confirmation"""
    # Detect table and name patterns
    patterns = {
        'employee': r'employee\s+(?:named\s+)?["\']?(\w+)["\']?',
        'doctor': r'doctor\s+(?:named\s+)?["\']?(\w+)["\']?',
        'patient': r'patient\s+(?:named\s+)?["\']?(\w+)["\']?',
        'player': r'player\s+(?:named\s+)?["\']?(\w+)["\']?',
        'client': r'client\s+(?:named\s+)?["\']?(\w+)["\']?',
    }
    
    query_lower = query.lower()
    
    for table, pattern in patterns.items():
        match = re.search(pattern, query_lower)
        if match:
            name = match.group(1)
            
            # Determine name column based on table
            name_columns = {
                'employee': 'employee_name',
                'doctor': 'doctor_name',
                'patient': 'patient_name',
                'player': 'player_name',
                'client': 'name'
            }
            
            name_column = name_columns.get(table)
            if name_column:
                matches = find_entity_by_name(table, name_column, name)
                
                if len(matches) > 1:
                    result = f"⚠️ Found multiple {table}s matching '{name}':\n\n"
                    for match in matches:
                        result += f"- **ID {match[0]}**: {match[1]}\n"
                    result += f"\nPlease specify the {table} ID in your query."
                    return result, True
                elif len(matches) == 1:
                    # Auto-resolve to single match
                    return f"Found {table}: {matches[0][1]} (ID: {matches[0][0]})", False
    
    return None, False

# ========== Query Detection ==========
def should_use_rag(query, force_rag=False):
    """Decide whether to send the query to RAG"""
    query_lower = (query or "").lower()

    if force_rag:
        return True

    strong_pdf_patterns = [
        r"\bpdf\b", r"\bdocument\b", r"\bfile\b", r"\bsummar(y|ize|ise)\b",
        r"\bexplain\b", r"\bwhat does the (doc|document|pdf) say\b",
        r"\bwhat('?s| is) in the (doc|document|pdf)\b", r"\bfrom the pdf\b",
        r"\bin the document\b", r"\bread the (pdf|document)\b",
        r"\bcontent of\b", r"\bshow the (pdf|document)\b"
    ]

    for pat in strong_pdf_patterns:
        if re.search(pat, query_lower):
            return True

    if not st.session_state.get("has_pdfs", False):
        return False

    weak_question_words = [
        r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bwhen\b", 
        r"\bwho\b", r"\bdescribe\b", r"\btell me about\b"
    ]

    db_keywords = [
        r"\bselect\b", r"\bfrom\b", r"\bwhere\b", r"\bjoin\b", 
        r"\bshow\b", r"\blist\b", r"\bcount\b", r"\brows\b", 
        r"\brecords\b", r"\bdatabase\b", r"\bsql\b", r"\bquery\b"
    ]

    for pat in db_keywords:
        if re.search(pat, query_lower):
            return False

    for pat in weak_question_words:
        if re.search(pat, query_lower):
            return True

    return False

def check_for_resolved_id(query):
    """Check if we already have a resolved ID for this entity"""
    if not st.session_state.sql_chat_history:
        return None
    
    # Look for entity names in query
    name_match = re.search(r"(?:of|for|by)\s+['\"]?(\w+)['\"]?", query, re.IGNORECASE)
    if not name_match:
        return None
    
    entity_name = name_match.group(1).lower()
    
    # Check recent history for resolved IDs
    for item in reversed(st.session_state.sql_chat_history[-5:]):
        if 'resolved_ids' in item and entity_name in item['resolved_ids']:
            return item['resolved_ids'][entity_name]
    
    return None

# ========== Enhanced Process Query with History ==========
def process_query(query, is_direct_sql=False):
    """Unified query processor with context awareness"""
    resolved_id = check_for_resolved_id(query)
    if resolved_id and re.search(r"(task|appointment|match)", query, re.IGNORECASE):
        # We have a cached ID, inject it into the prompt
        query = f"{query} (Use ID: {resolved_id})"
    
    # Check for schema backup commands (super admin only)
    if st.session_state.is_super_admin:
        backup_match = re.search(r'backup\s+schema\s+(?:for\s+)?(\w+)', query.lower())
        if backup_match:
            table_name = backup_match.group(1)
            backup = backup_table_schema(st.session_state.selected_db, table_name)
            if backup:
                if table_name not in st.session_state.schema_backups:
                    st.session_state.schema_backups[table_name] = []
                st.session_state.schema_backups[table_name].append(backup)
                return f"✅ Schema backup created for **{table_name}** at {backup['timestamp']}\n\nYou can now safely make schema modifications. Use the Schema Management dashboard to restore if needed."
            else:
                return f"❌ Failed to create backup for {table_name}. Please check if the table exists."
    
    # Check for name resolution needs
    resolution_msg, needs_clarification = resolve_entity_name(query)
    if needs_clarification:
        return resolution_msg
    
    if is_direct_sql:
        result = execute_direct_sql(query)
        # Add to SQL history
        st.session_state.sql_chat_history.append({
            'query': query,
            'result_summary': 'Direct SQL execution',
            'timestamp': datetime.datetime.now()
        })
        return result

    try:
        use_rag = should_use_rag(query)

        # RAG Path
        if use_rag:
            rag_chain = st.session_state.get("rag_chain", None)
            if not rag_chain:
                return "📄 I detected you're asking about PDFs, but no PDFs are currently loaded. Please add PDFs to the ./pdfs folder and reload."

            try:
                with st.spinner("📄 Searching through PDFs..."):
                    time.sleep(3)
                    result = rag_chain.invoke(
                        {"input": query},
                        config={"configurable": {"session_id": "session1"}}
                    )
                answer = result.get("answer") if isinstance(result, dict) else str(result)
                if not answer:
                    return "📄 I cannot find this information in the provided PDFs. Please ensure your question relates to the PDF content."
                return "📄 " + answer
            except Exception as e:
                return f"⚠️ Error searching PDFs: {str(e)}"

        # SQL Agent Path with context
        if re.search(r"\b(pdf|document|file|summarize|summary|read the)\b", query.lower()):
            return "⚠️ This looks like a PDF/document question. Please upload PDFs to use the RAG system."

        try:
            with st.spinner("🤔 Thinking..."):
                time.sleep(3)

            # Create agent with current context
            sql_agent = create_sql_agent_with_context()

            # NEW: Use invoke with StreamlitCallbackHandler to capture intermediate steps
            response = sql_agent.run(query)
            
            # **ADD THIS: Extract and store resolved IDs**
            resolved_ids = {}

            # Look for ID resolution patterns in the response
            id_patterns = [
                r"employee_id[:\s=]+(\d+)",
                r"doctor_id[:\s=]+(\d+)",
                r"patient_id[:\s=]+(\d+)",
                r"client_id[:\s=]+(\d+)",
                r"patient_id[:\s=]+(\d+)",
                r"player_id[:\s=]+(\d+)",
                r"team_id[:\s=]+(\d+)",
                r"Found.*?ID[:\s]+(\d+)",
                r"ID:\s*(\d+)",
            ]

            for pattern in id_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    # Extract entity name from query
                    name_match = re.search(r"(?:named|called|for|of)\s+['\"]?(\w+)['\"]?", query, re.IGNORECASE)
                    if name_match:
                        entity_name = name_match.group(1)
                        resolved_ids[entity_name.lower()] = matches[0]

            # NEW: Try to extract table data directly from the database if query mentions SELECT
            if 'select' in query.lower() or 'show' in query.lower():
            # Try to execute the last SQL query to get raw data
                try:
                    # Extract SQL from response or recent SQL history
                    sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
                    if not sql_match:
                        sql_match = re.search(r'```\n(SELECT.*?)\n```', response, re.DOTALL | re.IGNORECASE)
                    if not sql_match:
                        # Look for SELECT statements in response
                        sql_match = re.search(r'(SELECT\s+.*?(?:FROM|;).*?)(?:\n|$)', response, re.DOTALL | re.IGNORECASE)
       
                    if sql_match:
                        sql_query = sql_match.group(1).strip()
                        # Clean up the query
                        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
                        if not sql_query.endswith(';'):
                            sql_query += ';'
           
                        # Execute to get raw data
                        conn = sqlite3.connect(st.session_state.selected_db)
                        cursor = conn.cursor()
                        cursor.execute(sql_query)
                        results = cursor.fetchall()
                        columns = [description[0] for description in cursor.description]
                        conn.close()
           
                        if results:
                            # Store as dataframe for rendering
                            import pandas as pd
                            df = pd.DataFrame(results, columns=columns)
               
                            # Add to SQL history WITH resolved IDs
                            st.session_state.sql_chat_history.append({
                                'query': query,
                                'result_summary': f"Found {len(results)} records",
                                'timestamp': datetime.datetime.now(),
                                'dataframe': df,
                                'resolved_ids': resolved_ids  # **NEW - Added here**
                            })
               
                            # Return response with embedded dataframe marker
                            return f"✅ **Complete!**\n\n{response}\n\n__DATAFRAME_RESULT__"
               
                except Exception as e:
                    # If extraction fails, just continue with normal response
                    pass

                # Add to SQL history WITH resolved IDs
            st.session_state.sql_chat_history.append({
            'query': query,
            'result_summary': response[:200] if len(response) > 200 else response,
            'timestamp': datetime.datetime.now(),
            'resolved_ids': resolved_ids  # **NEW - Added here too**
            })

            return f"✅ **Complete!**\n\n{response}"

        except Exception as e:
            err = str(e).lower()
            if any(keyword in err for keyword in ["drop", "truncate"]):
                return "🚫 DROP and TRUNCATE operations are not allowed for safety, even in Super Admin mode."
            elif "insert" in err:
                return "📝 Please use the guided 'Add New Record' form instead of typing INSERT manually."
            elif "permission" in err and not st.session_state.is_super_admin:
                return "🚫 Permission denied. This action requires Super Admin mode. Use 'enable super admin' to activate."
            elif "alter" in err and not st.session_state.is_super_admin:
                return "🚫 ALTER operations require Super Admin mode. Use 'enable super admin' to activate."
            elif "parse" in err or "output_parsing_error" in err:
                return "⚠️ The model couldn't understand that query format. Try rephrasing it."
            else:
                return f"⚠️ Something went wrong while processing your query: {str(e)}"

    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"
    
    
# ========== Smart Query Detection ==========
def detect_intent(query):
    """Detect if user wants to create/add records, delete records, OR drop columns"""
    query_lower = query.lower()
    
    conn = sqlite3.connect(st.session_state.selected_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # Check for column drop/removal intent (highest priority)
    column_drop_patterns = [
        r'(?:remove|drop|delete)\s+column\s+(?:name\s+)?(\w+)\s+(?:in|from)\s+(\w+)\s+table',
        r'(?:remove|drop|delete)\s+(?:the\s+)?column\s+(\w+)\s+(?:in|from)\s+(?:table\s+)?(\w+)',
        r'drop\s+(\w+)\s+column\s+(?:in|from)\s+(\w+)',
        r'alter\s+table\s+(\w+)\s+drop\s+column\s+(\w+)'
    ]
    
    for pattern in column_drop_patterns:
        match = re.search(pattern, query_lower)
        if match:
            # Pattern might have column and table in different orders
            groups = match.groups()
            # Try to identify which is table name
            for table in tables:
                if table.lower() in groups:
                    table_name = table
                    column_name = [g for g in groups if g.lower() != table.lower()][0]
                    return f"drop_column_{table_name}_{column_name}"
    
    # Check for delete record intent
    delete_patterns = [
        r'delete\s+(?:the\s+)?(\w+)\s+(?:with|having|where)',
        r'remove\s+(?:the\s+)?(\w+)\s+(?:with|having|where)',
        r'delete\s+(?:the\s+)?(\w+)\s+(?:named|called)'
    ]
    
    for pattern in delete_patterns:
        match = re.search(pattern, query_lower)
        if match:
            entity = match.group(1)
            # Check if it's a table name
            for table in tables:
                if table.lower() == entity.lower() or table.lower() in query_lower:
                    return f"delete_{table}"
    
    # Check for create intent
    create_patterns = {}
    for table in tables:
        patterns = [
            rf'create\s+(a\s+)?{table}',
            rf'add\s+(a\s+)?(new\s+)?{table}',
            rf'new\s+{table}',
            rf'insert\s+(a\s+)?{table}'
        ]
        create_patterns[table] = patterns
    
    for table, patterns in create_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return f"create_{table}"
    
    return "normal"

# ========== Handle Delete Operations ==========
def handle_delete_intent(table_name, query):
    """Handle delete operations with confirmation"""
    # Extract potential identifiers from query
    id_match = re.search(r'id[:\s]+(\d+)', query.lower())
    name_match = re.search(r'(?:named|called)\s+["\']?(\w+)["\']?', query.lower())
    
    if id_match:
        # Direct ID provided
        record_id = id_match.group(1)
        st.session_state.pending_delete = {
            'table': table_name,
            'id': record_id,
            'query': query
        }
        return True
    elif name_match:
        # Name provided - need to find matches
        name = name_match.group(1)
        name_columns = {
            'employee': 'employee_name',
            'doctor': 'doctor_name',
            'patient': 'patient_name',
            'player': 'player_name',
            'client': 'name',
            'task': 'task_description',
            'appointment': 'diagnosis',
            'team': 'team_name'
        }
        
        name_column = name_columns.get(table_name)
        if name_column:
            matches = find_entity_by_name(table_name, name_column, name)
            
            if len(matches) == 0:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ No {table_name} found matching '{name}'."
                })
                return False
            elif len(matches) > 1:
                result = f"⚠️ Found multiple {table_name}s matching '{name}':\n\n"
                for match in matches:
                    result += f"- **ID {match[0]}**: {match[1]}\n"
                result += f"\n❓ Please specify which {table_name} to delete by ID.\n"
                result += f"Example: 'delete {table_name} with id {matches[0][0]}'"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result
                })
                return False
            else:
                # Single match found
                st.session_state.pending_delete = {
                    'table': table_name,
                    'id': matches[0][0],
                    'name': matches[0][1],
                    'query': query
                }
                return True
    
    return False

def show_delete_confirmation():
    """Show delete confirmation dialog"""
    delete_info = st.session_state.pending_delete
    
    # Get record details
    conn = sqlite3.connect(st.session_state.selected_db)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {delete_info['table']} WHERE rowid = ?", (delete_info['id'],))
    record = cursor.fetchone()
    
    if record:
        cursor.execute(f"PRAGMA table_info({delete_info['table']})")
        columns = [col[1] for col in cursor.fetchall()]
        conn.close()
        
        st.warning(f"⚠️ **Confirm Deletion of {delete_info['table'].title()}**")
        st.write("You are about to delete the following record:")
        
        # Display record details
        record_df = pd.DataFrame([record], columns=columns)
        st.dataframe(record_df, use_container_width=True)
        
        st.error("⚠️ This action cannot be undone!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Delete", type="primary", use_container_width=True, key="confirm_delete"):
                try:
                    conn = sqlite3.connect(st.session_state.selected_db)
                    cursor = conn.cursor()
                    cursor.execute(f"DELETE FROM {delete_info['table']} WHERE rowid = ?", (delete_info['id'],))
                    conn.commit()
                    conn.close()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ {delete_info['table'].title()} (ID: {delete_info['id']}) deleted successfully!"
                    })
                    st.session_state.pending_delete = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting record: {str(e)}")
        
        with col2:
            if st.button("❌ Cancel", use_container_width=True, key="cancel_delete"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Deletion cancelled."
                })
                st.session_state.pending_delete = None
                st.rerun()
    else:
        conn.close()
        st.error(f"❌ Record not found in {delete_info['table']}")
        st.session_state.pending_delete = None

# ========== Handle Column Drop Operations ==========
def handle_column_drop_intent(table_name, column_name, query):
    """Handle column drop operations with super admin verification"""
    
    # Check if user is super admin
    if not st.session_state.is_super_admin:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🚫 **Access Denied**\n\nDropping columns requires Super Admin privileges.\n\nTo enable Super Admin mode, type: `enable super admin`"
        })
        return False
    
    # Check if column exists
    conn = sqlite3.connect(st.session_state.selected_db)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    current_columns = cursor.fetchall()
    conn.close()
    
    column_names = [col[1] for col in current_columns]
    
    if column_name not in column_names:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ **Column Not Found**\n\nThe column `{column_name}` does not exist in the `{table_name}` table.\n\n**Available columns:**\n" + 
                       "\n".join([f"- {col}" for col in column_names])
        })
        return False
    
    # Check if we have a backup to determine if this column was added
    if table_name in st.session_state.schema_backups and st.session_state.schema_backups[table_name]:
        original_backup = st.session_state.schema_backups[table_name][0]
        original_columns = [col[1] for col in original_backup['columns']]
        
        if column_name in original_columns:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🚫 **Cannot Drop Original Column**\n\n"
                           f"The column `{column_name}` is part of the original `{table_name}` table schema and cannot be dropped.\n\n"
                           f"**Super Admin can only drop columns that were added after the original schema.**\n\n"
                           f"Original columns: {', '.join(original_columns)}"
            })
            return False
    else:
        # No backup exists - warn and offer to proceed with caution
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ **No Backup Found**\n\n"
                       f"There's no backup for the `{table_name}` table, so I cannot verify if `{column_name}` is an added column or part of the original schema.\n\n"
                       f"**Recommendation:** Create a backup first using:\n"
                       f"`backup schema for {table_name}`\n\n"
                       f"Then use the **Schema Management Dashboard** to safely drop columns."
        })
        return False
    
    # Column can be dropped - set up confirmation
    st.session_state.pending_column_drop = {
        'table': table_name,
        'column': column_name,
        'query': query
    }
    return True

def show_column_drop_confirmation():
    """Show column drop confirmation dialog"""
    drop_info = st.session_state.pending_column_drop
    
    st.warning(f"⚠️ **Confirm Column Removal**")
    st.write(f"You are about to drop the column `{drop_info['column']}` from the `{drop_info['table']}` table.")
    
    # Show current table structure
    conn = sqlite3.connect(st.session_state.selected_db)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({drop_info['table']})")
    columns = cursor.fetchall()
    
    # Show sample data from this column
    try:
        cursor.execute(f"SELECT {drop_info['column']} FROM {drop_info['table']} LIMIT 5")
        sample_data = cursor.fetchall()
        conn.close()
        
        if sample_data:
            st.write("**Sample data that will be lost:**")
            for row in sample_data:
                st.text(f"  • {row[0]}")
    except:
        conn.close()
    
    st.error("⚠️ **This action cannot be undone!** All data in this column will be permanently lost.")
    st.info("💡 You can restore the table from a backup if needed using the Schema Management Dashboard.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Confirm Drop Column", type="primary", use_container_width=True, key="confirm_column_drop"):
            try:
                conn = sqlite3.connect(st.session_state.selected_db)
                cursor = conn.cursor()
                
                # Get all columns except the one to drop
                cursor.execute(f"PRAGMA table_info({drop_info['table']})")
                all_columns = cursor.fetchall()
                remaining_cols = [col[1] for col in all_columns if col[1] != drop_info['column']]
                
                # SQLite doesn't support DROP COLUMN directly, need to recreate table
                cursor.execute(f"CREATE TABLE {drop_info['table']}_temp AS SELECT {', '.join(remaining_cols)} FROM {drop_info['table']}")
                cursor.execute(f"DROP TABLE {drop_info['table']}")
                cursor.execute(f"ALTER TABLE {drop_info['table']}_temp RENAME TO {drop_info['table']}")
                
                conn.commit()
                conn.close()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ **Column Dropped Successfully**\n\n"
                               f"The column `{drop_info['column']}` has been removed from the `{drop_info['table']}` table.\n\n"
                               f"Remaining columns: {', '.join(remaining_cols)}"
                })
                st.session_state.pending_column_drop = None
                st.success("✅ Column dropped successfully!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error dropping column: {str(e)}")
    
    with col2:
        if st.button("📋 View Schema Manager", use_container_width=True, key="view_schema_from_drop"):
            st.session_state.pending_column_drop = None
            st.session_state.show_schema_manager = True
            st.rerun()
    
    with col3:
        if st.button("❌ Cancel", use_container_width=True, key="cancel_column_drop"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Column drop cancelled."
            })
            st.session_state.pending_column_drop = None
            st.rerun()
    
# ========== Get Table Schema ==========
def get_table_columns(table_name):
    """Get table columns (excluding auto-increment ID)"""
    conn = sqlite3.connect(st.session_state.selected_db)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()
    
    return [col[1] for col in columns if col[5] == 0 or not col[1].endswith('_id')]

def get_foreign_key_data(table_name, column_name):
    """Get data for foreign key dropdowns — generic by current DB"""
    conn = sqlite3.connect(st.session_state.selected_db)
    cursor = conn.cursor()
    
    # Support for different database schemas
    try:
        # Task Manager DB
        if table_name == "task":
            if 'employee' in column_name.lower():
                cursor.execute("SELECT employee_id, employee_name FROM employee")
                data = cursor.fetchall()
                conn.close()
                return {f"{row[1]} (ID: {row[0]})": row[0] for row in data}
            elif 'client' in column_name.lower():
                cursor.execute("SELECT client_id, name FROM client")
                data = cursor.fetchall()
                conn.close()
                return {f"{row[1]} (ID: {row[0]})": row[0] for row in data}
        
        # Hospital DB
        elif table_name == "appointment":
            if 'doctor' in column_name.lower():
                cursor.execute("SELECT doctor_id, doctor_name FROM doctor")
                data = cursor.fetchall()
                conn.close()
                return {f"Dr. {row[1]} (ID: {row[0]})": row[0] for row in data}
            elif 'patient' in column_name.lower():
                cursor.execute("SELECT patient_id, patient_name FROM patient")
                data = cursor.fetchall()
                conn.close()
                return {f"{row[1]} (ID: {row[0]})": row[0] for row in data}
        
        # Sports DB
        elif table_name == "player":
            if 'team' in column_name.lower():
                cursor.execute("SELECT team_id, team_name FROM team")
                data = cursor.fetchall()
                conn.close()
                return {f"{row[1]} (ID: {row[0]})": row[0] for row in data}
        elif table_name == "match":
            if 'team' in column_name.lower():
                cursor.execute("SELECT team_id, team_name FROM team")
                data = cursor.fetchall()
                conn.close()
                return {f"{row[1]} (ID: {row[0]})": row[0] for row in data}
    except:
        pass
    
    conn.close()
    return {}

# ========== Location Search Helper ==========
def search_location(search_term):
    """Search for locations using OpenStreetMap Nominatim API"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": search_term, "format": "json", "limit": 5}
        res = requests.get(url, params=params, headers={"User-Agent": "AI-Assistant"}, timeout=5)
        if res.status_code == 200 and res.json():
            return [r["display_name"] for r in res.json()]
    except Exception as e:
        return []
    return []

# ========== Universal Record Creation Form ==========
def show_record_creation_form(table_name):
    """Universal form for creating records in any table"""
    st.info(f"📝 Let's create a new {table_name}! Fill in all the details below:")
    columns = get_table_columns(table_name)
    
    if f"{table_name}_location_selected" not in st.session_state:
        st.session_state[f"{table_name}_location_selected"] = None
    
    form_data = {}
    
    st.subheader(f"{table_name.title()} Information")
    col1, col2 = st.columns(2)
    
    for idx, column in enumerate(columns):
        current_col = col1 if idx % 2 == 0 else col2
        
        with current_col:
            display_name = column.replace('_', ' ').title()
            
            fk_data = get_foreign_key_data(table_name, column)
            
            if fk_data:
                options = [""] + list(fk_data.keys())
                form_data[column] = st.selectbox(
                    f"{display_name}*",
                    options=options,
                    key=f"{table_name}_{column}_input"
                )
            elif 'date' in column.lower() and 'time' not in column.lower():
                form_data[column] = st.date_input(
                    f"{display_name}*",
                    value=datetime.date.today(),
                    key=f"{table_name}_{column}_input"
                )
            elif 'time' in column.lower() and 'date' not in column.lower():
                form_data[column] = st.time_input(
                    f"{display_name}*",
                    value=datetime.time(9, 0),
                    key=f"{table_name}_{column}_input"
                )
            elif column.lower() in ['location', 'city', 'address']:
                st.write(f"**{display_name}***")
                search_col, btn_col = st.columns([4, 1])
                with search_col:
                    search_term = st.text_input(
                        "Search location:",
                        placeholder="Type to search...",
                        key=f"{table_name}_{column}_search",
                        label_visibility="collapsed"
                    )
                with btn_col:
                    search_button = st.button("🔍", key=f"{table_name}_{column}_btn", help="Search location")
                
                if search_button and search_term:
                    with st.spinner("Searching..."):
                        locations = search_location(search_term)
                        if locations:
                            st.session_state[f"{table_name}_{column}_results"] = locations
                        else:
                            st.warning("No locations found")
                            st.session_state[f"{table_name}_{column}_results"] = []
                
                if f"{table_name}_{column}_results" in st.session_state and st.session_state[f"{table_name}_{column}_results"]:
                    selected_location = st.selectbox(
                        "Select from results:",
                        options=[""] + st.session_state[f"{table_name}_{column}_results"],
                        key=f"{table_name}_{column}_select"
                    )
                    if selected_location:
                        st.session_state[f"{table_name}_location_selected"] = selected_location
                        form_data[column] = selected_location
                    else:
                        form_data[column] = ""
                elif st.session_state[f"{table_name}_location_selected"]:
                    form_data[column] = st.text_input(
                        "Selected location:",
                        value=st.session_state[f"{table_name}_location_selected"],
                        key=f"{table_name}_{column}_display"
                    )
                else:
                    form_data[column] = st.text_input(
                        "Or enter manually:",
                        placeholder="Enter location",
                        key=f"{table_name}_{column}_manual"
                    )
            elif column.lower() in ['description', 'notes', 'project', 'task_description', 'diagnosis', 'symptoms']:
                form_data[column] = st.text_area(
                    f"{display_name}",
                    placeholder=f"Enter {display_name.lower()}",
                    key=f"{table_name}_{column}_input",
                    height=100
                )
            elif column.lower() in ['email', 'email_id']:
                form_data[column] = st.text_input(
                    f"{display_name}*",
                    placeholder="example@company.com",
                    key=f"{table_name}_{column}_input"
                )
            elif column.lower() in ['status']:
                form_data[column] = st.selectbox(
                    f"{display_name}*",
                    options=["Pending", "In Progress", "Completed", "On Hold"],
                    key=f"{table_name}_{column}_input"
                )
            elif column.lower() in ['priority']:
                form_data[column] = st.selectbox(
                    f"{display_name}*",
                    options=["Low", "Medium", "High", "Critical"],
                    key=f"{table_name}_{column}_input"
                )
            elif column.lower() in ['gender']:
                form_data[column] = st.selectbox(
                    f"{display_name}*",
                    options=["Male", "Female", "Other"],
                    key=f"{table_name}_{column}_input"
                )
            elif column.lower() in ['age', 'jersey_number']:
                form_data[column] = st.number_input(
                    f"{display_name}*",
                    min_value=0,
                    max_value=150 if 'age' in column.lower() else 99,
                    value=25 if 'age' in column.lower() else 10,
                    key=f"{table_name}_{column}_input"
                )
            elif column.lower() in ['specialization', 'specialty', 'position']:
                form_data[column] = st.text_input(
                    f"{display_name}*",
                    placeholder=f"Enter {display_name.lower()}",
                    key=f"{table_name}_{column}_input"
                )
            else:
                form_data[column] = st.text_input(
                    f"{display_name}*",
                    placeholder=f"Enter {display_name.lower()}",
                    key=f"{table_name}_{column}_input"
                )
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button(f"✅ Create {table_name.title()}", type="primary", use_container_width=True, key=f"{table_name}_submit"):
            empty_fields = []
            for col, val in form_data.items():
                if val is None or (isinstance(val, str) and val.strip() == ''):
                    empty_fields.append(col.replace('_', ' ').title())
            
            if empty_fields:
                st.error(f"❌ Please fill in all required fields: {', '.join(empty_fields)}")
            else:
                column_names = ', '.join(columns)
                values = []
                
                for col in columns:
                    val = form_data[col]
                    
                    fk_data = get_foreign_key_data(table_name, col)
                    if fk_data and val in fk_data:
                        values.append(str(fk_data[val]))
                        continue
                    
                    if isinstance(val, (datetime.date, datetime.time)):
                        values.append(f"'{val}'")
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    elif isinstance(val, str):
                        escaped_val = val.replace("'", "''")
                        values.append(f"'{escaped_val}'")
                    else:
                        values.append(f"'{val}'")
                
                values_str = ', '.join(values)
                query = f"INSERT INTO {table_name} ({column_names}) VALUES ({values_str})"
                try:
                    conn = sqlite3.connect(st.session_state.selected_db)
                    cursor = conn.cursor()
                    cursor.execute(query)
                    conn.commit()
                    conn.close()
                    
                    keys_to_clear = [k for k in st.session_state.keys() if k.startswith(f"{table_name}_")]
                    for key in keys_to_clear:
                        del st.session_state[key]
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ {table_name.title()} created successfully!"
                    })
                    st.session_state.show_form = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error creating {table_name}: {str(e)}")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True, key=f"{table_name}_cancel"):
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith(f"{table_name}_")]
            for key in keys_to_clear:
                del st.session_state[key]
            st.session_state.show_form = None
            st.rerun()

# ========== Schema Management UI ==========
def show_schema_manager():
    """Display schema backup and restoration interface"""
    st.title("🔧 Schema Management Dashboard")
    
    if not st.session_state.is_super_admin:
        st.error("🚫 Access Denied: Schema management requires Super Admin privileges")
        if st.button("← Back to Chat"):
            st.session_state.show_schema_manager = False
            st.rerun()
        return
    
    st.info("💡 Manage table schemas, view backups, and restore previous states")
    
    # Get all tables
    conn = sqlite3.connect(st.session_state.selected_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    tab1, tab2, tab3 = st.tabs(["📊 Current Schemas", "💾 Backups", "🗑️ Drop Columns"])
    
    # TAB 1: Current Schemas
    with tab1:
        st.subheader("Current Table Schemas")
        
        selected_table = st.selectbox("Select Table:", tables, key="schema_view_table")
        
        if selected_table:
            conn = sqlite3.connect(st.session_state.selected_db)
            cursor = conn.cursor()
            
            # Get schema
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{selected_table}'")
            schema = cursor.fetchone()[0]
            
            # Get columns
            cursor.execute(f"PRAGMA table_info({selected_table})")
            columns = cursor.fetchall()
            conn.close()
            
            st.code(schema, language="sql")
            
            st.write("**Columns:**")
            df = pd.DataFrame(columns, columns=["cid", "name", "type", "notnull", "default", "pk"])
            st.dataframe(df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"💾 Backup {selected_table} Schema", use_container_width=True, type="primary"):
                    backup = backup_table_schema(st.session_state.selected_db, selected_table)
                    if backup:
                        if selected_table not in st.session_state.schema_backups:
                            st.session_state.schema_backups[selected_table] = []
                        st.session_state.schema_backups[selected_table].append(backup)
                        st.success(f"✅ Schema backed up for {selected_table}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to create backup")
    
    # TAB 2: Backups and Restore
    with tab2:
        st.subheader("Schema Backups")
        
        if not st.session_state.schema_backups:
            st.warning("📭 No backups available. Create backups before making schema changes.")
        else:
            backup_table = st.selectbox("Select Table:", list(st.session_state.schema_backups.keys()), key="backup_table")
            
            if backup_table and st.session_state.schema_backups[backup_table]:
                st.write(f"**Available Backups for {backup_table}:**")
                
                backups = st.session_state.schema_backups[backup_table]
                
                for idx, backup in enumerate(reversed(backups)):
                    with st.expander(f"🕐 Backup {len(backups) - idx} - {backup['timestamp']}", expanded=(idx == 0)):
                        st.code(backup['schema_sql'], language="sql")
                        
                        st.write("**Columns:**")
                        df = pd.DataFrame(backup['columns'], columns=["cid", "name", "type", "notnull", "default", "pk"])
                        st.dataframe(df, use_container_width=True)
                        
                        st.write(f"**Data Rows:** {len(backup['data'])}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"🔄 Restore This Backup", key=f"restore_{backup_table}_{idx}", type="primary", use_container_width=True):
                                st.session_state.pending_restore = backup
                                st.session_state.pending_restore_idx = len(backups) - idx - 1
                                st.rerun()
                        
                        with col2:
                            if st.button(f"🗑️ Delete Backup", key=f"delete_backup_{backup_table}_{idx}", use_container_width=True):
                                st.session_state.schema_backups[backup_table].pop(len(backups) - idx - 1)
                                st.success("✅ Backup deleted")
                                st.rerun()
    
    # TAB 3: Drop Added Columns
    with tab3:
        st.subheader("Drop Added Columns")
        st.warning("⚠️ Only columns added AFTER the original schema can be dropped here")
        
        drop_table = st.selectbox("Select Table:", tables, key="drop_column_table")
        
        if drop_table:
            # Check if we have a backup to compare against
            if drop_table in st.session_state.schema_backups and st.session_state.schema_backups[drop_table]:
                original_backup = st.session_state.schema_backups[drop_table][0]
                added_columns = get_added_columns(st.session_state.selected_db, drop_table, original_backup['columns'])
                
                if added_columns:
                    st.write("**Added Columns (can be dropped):**")
                    for col in added_columns:
                        st.info(f"📌 {col}")
                    
                    selected_column = st.selectbox("Select Column to Drop:", added_columns, key="column_to_drop")
                    
                    st.error(f"⚠️ You are about to drop column: **{selected_column}**")
                    st.write("This action cannot be undone unless you restore from a backup!")
                    
                    if st.button(f"🗑️ Drop Column: {selected_column}", type="primary"):
                        try:
                            # SQLite doesn't support DROP COLUMN directly, need to recreate table
                            conn = sqlite3.connect(st.session_state.selected_db)
                            cursor = conn.cursor()
                            
                            # Get current columns
                            cursor.execute(f"PRAGMA table_info({drop_table})")
                            all_columns = cursor.fetchall()
                            remaining_cols = [col[1] for col in all_columns if col[1] != selected_column]
                            
                            # Create new table without the column
                            cursor.execute(f"CREATE TABLE {drop_table}_temp AS SELECT {', '.join(remaining_cols)} FROM {drop_table}")
                            cursor.execute(f"DROP TABLE {drop_table}")
                            cursor.execute(f"ALTER TABLE {drop_table}_temp RENAME TO {drop_table}")
                            
                            conn.commit()
                            conn.close()
                            
                            st.success(f"✅ Column '{selected_column}' dropped from {drop_table}")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error dropping column: {str(e)}")
                else:
                    st.info("✅ No additional columns have been added to this table")
            else:
                st.warning("⚠️ No backup found for this table. Create a backup first to track added columns.")
                
                if st.button(f"💾 Create Backup for {drop_table}"):
                    backup = backup_table_schema(st.session_state.selected_db, drop_table)
                    if backup:
                        if drop_table not in st.session_state.schema_backups:
                            st.session_state.schema_backups[drop_table] = []
                        st.session_state.schema_backups[drop_table].append(backup)
                        st.success(f"✅ Backup created for {drop_table}")
                        st.rerun()
    
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("← Back to Chat", use_container_width=True):
            st.session_state.show_schema_manager = False
            st.rerun()

# ========== Handle Restore Confirmation ==========
def show_restore_confirmation():
    """Show confirmation dialog for schema restoration"""
    backup = st.session_state.pending_restore
    
    st.warning(f"⚠️ **Confirm Schema Restoration**")
    st.write(f"You are about to restore **{backup['table_name']}** to the state from:")
    st.info(f"🕐 {backup['timestamp']}")
    
    st.write("**Schema to be restored:**")
    st.code(backup['schema_sql'], language="sql")
    
    st.error("⚠️ Current table will be replaced with this backup!")
    st.write("- All current data may be lost if incompatible with old schema")
    st.write("- Any columns added after this backup will be removed")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Restore", type="primary", use_container_width=True):
            if restore_table_schema(backup):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ Successfully restored **{backup['table_name']}** to backup from {backup['timestamp']}"
                })
                st.session_state.pending_restore = None
                st.session_state.show_schema_manager = False
                st.success("✅ Schema restored successfully!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Failed to restore schema")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.pending_restore = None
            st.rerun()

# ========== Initialize pending restore ==========
if "pending_restore" not in st.session_state:
    st.session_state.pending_restore = None

# ========== Initialize pending states ==========
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None

if "pending_column_drop" not in st.session_state:
    st.session_state.pending_column_drop = None

# ========== Show Schema Manager if active ==========
if st.session_state.show_schema_manager:
    show_schema_manager()
    st.stop()

# ========== Handle restore confirmation ==========
if st.session_state.pending_restore:
    show_restore_confirmation()
    st.stop()

# ========== Handle column drop confirmation ==========
if st.session_state.pending_column_drop:
    show_column_drop_confirmation()
    st.stop()

# ========== Handle pending delete confirmation ==========
if st.session_state.pending_delete:
    show_delete_confirmation()
    st.stop()

# ========== Handle admin authentication ==========
if st.session_state.admin_auth_pending:
    show_admin_auth()
    st.stop()

# ========== Handle pending query from sidebar ==========
if st.session_state.pending_query:
    query = st.session_state.pending_query
    is_sql = st.session_state.is_direct_sql
    st.session_state.pending_query = None
    st.session_state.is_direct_sql = False
    
    if is_sql:
        user_message = "Executing quick query..."
    else:
        user_message = query
    
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.chat_message("user").write(user_message)
    
    with st.chat_message("assistant"):
        response = process_query(query, is_direct_sql=is_sql)
        df = try_render_table(response)
        if df is not None:
            st.write("📊 **Results:**")
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"query_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_pending_{len(st.session_state.messages)}"
            )
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"📊 Displaying {len(df)} results as a table.",
                "dataframe": df.to_dict()
            })
        else:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    st.rerun()

# ========== Chat Input ==========
user_query = st.chat_input("Ask something...")

if "show_form" not in st.session_state:
    st.session_state.show_form = None

if st.session_state.show_form:
    show_record_creation_form(st.session_state.show_form)

# ========== Process user input ==========
if user_query:
    # Check for super admin trigger
    if check_super_admin_trigger(user_query):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.admin_auth_pending = True
        st.rerun()
    
    intent = detect_intent(user_query)
    
    # Handle column drop intent
    if intent.startswith("drop_column_"):
        parts = intent.replace("drop_column_", "").rsplit("_", 1)
        if len(parts) == 2:
            table_name, column_name = parts
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            if handle_column_drop_intent(table_name, column_name, user_query):
                st.rerun()
            else:
                # Error message already added in handle_column_drop_intent
                st.rerun()
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ Could not parse the column drop request. Please use format:\n"
                           "`remove column [column_name] from [table_name] table`"
            })
            st.rerun()
    
    # Handle delete record intent
    elif intent.startswith("delete_"):
        table_name = intent.replace("delete_", "")
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        if handle_delete_intent(table_name, user_query):
            st.rerun()
        else:
            # Message already added in handle_delete_intent
            st.rerun()
    
    # Handle create intent
    elif intent.startswith("create_"):
        table_name = intent.replace("create_", "")
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Great! I'll help you create a new {table_name}. Please fill in the form below with all the required information."
        })
        st.session_state.show_form = table_name
        st.rerun()
    
    # Handle normal query
    else:
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            response = process_query(user_query, is_direct_sql=False)
            df = try_render_table(response)
            if df is not None:
                st.write("📊 **Results:**")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_new_{len(st.session_state.messages)}"
                )
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"📊 Displaying {len(df)} results as a table.",
                    "dataframe": df.to_dict()
                })
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)

# Trim old messages and keep last 10 conversations
if len(st.session_state.messages) > 10:
    st.session_state.messages = st.session_state.messages[-10:]
