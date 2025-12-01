# AI-Database-Assistant
A hybrid AI assistant built with Streamlit, LangChain, and Groq. Features text-to-SQL conversion for database interaction, RAG for PDF querying using ChromaDB, and a Super Admin dashboard for schema management.

🚀 OverviewThis is a Streamlit-based AI assistant that allows users to interact with SQL databases and PDF documents using natural language. It leverages Groq (Qwen-32B) for high-speed inference and LangChain for SQL Agent orchestration.✨ FeaturesText-to-SQL: Converts natural language questions into complex SQL queries automatically.Super Admin Dashboard: Secure management for database schemas (Backup/Restore/Drop Columns).RAG (Chat with PDF): Upload PDFs and ask questions using Vector Search (ChromaDB).Dynamic Forms: Automatically generates UI forms for inserting data into any table.🛠️ Tech StackFrontend: StreamlitAI/LLM: Groq API (Qwen-32B), LangChainDatabase: SQLite, ChromaDB (Vector Store)Language: Python 3.10+⚙️ Installation & SetupFollow these steps to set up the project locally.1. Clone the Repositorygit clone [https://github.com/yourusername/ai-database-assistant.git](https://github.com/yourusername/ai-database-assistant.git)
cd ai-database-assistant
2. Create a Virtual Environment (Recommended)It's best practice to use a virtual environment to manage dependencies.On Windows:python -m venv venv
venv\Scripts\activate
On macOS/Linux:python3 -m venv venv
source venv/bin/activate
3. Install DependenciesInstall the required Python packages using pip.pip install -r requirements.txt
4. Set Up Environment VariablesCreate a .env file in the root directory.Add your Groq API key to the file. You can get an API key from Groq Console.GROQ_API_KEY=your_actual_api_key_here
5. Add Database FilesEnsure your SQLite database files (e.g., task-manager.db, hospital.db) are placed in the root directory of the project.6. Run the ApplicationStart the Streamlit app.streamlit run app.py
The application should now be running at http://localhost:8501.📂 Project Structureai-database-assistant/
├── app.py                # Main application file
├── requirements.txt      # Project dependencies
├── .env.example          # Example environment variables
├── README.md             # Project documentation
├── pdfs/                 # Directory for PDF files (auto-created)
└── *.db                  # SQLite database files
🤝 ContributingContributions are welcome! Please feel free to submit a Pull Request.
