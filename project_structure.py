# create_project_structure.py
import os

def create_project_structure():
    """Create the complete project structure for Advance Plant AI Assistant"""
    
    # Define the project structure
    structure = {
        "app.py": "# Main application entry point\n",
        "auth.py": "# Authentication functionality\n",
        "styles.py": "# UI styles and configuration\n",
        "requirements.txt": "# Dependencies\n",
        "Database": {},
        "models": {},
        "rag": {
            "__init__.py": "# RAG package\n",
            "core.py": "# RAG core functionality\n",
            "document_processing.py": "# Document processing functions\n",
            "retrieval.py": "# Retrieval functions\n"
        },
        "models": {
            "__init__.py": "# Models package\n",
            "adapters.py": "# Model adapters for different model types\n",
            "model_manager.py": "# Model management functions\n"
        },
        "utils": {
            "__init__.py": "# Utils package\n",
            "helpers.py": "# General helper functions\n",
            "nltk_setup.py": "# NLTK setup and download\n"
        },
        "pages": {
            "__init__.py": "# Pages package\n",
            "sidebar.py": "# Sidebar rendering functions\n",
            "home.py": "# Main chat interface\n",
            "models.py": "# Model management page\n",
            "index.py": "# Index management page\n",
            "database_analysis.py": "# Database analysis page\n",
            "presets.py": "# Presets page\n",
            "logs.py": "# Logs page\n",
            "analytics.py": "# Analytics page\n",
            "exports.py": "# Exports page\n",
            "developer_tools.py": "# Developer tools page\n"
        }
    }
    
    # Create the directory structure
    def create_dirs(base, structure):
        for name, content in structure.items():
            path = os.path.join(base, name)
            if isinstance(content, dict):
                # It's a directory
                os.makedirs(path, exist_ok=True)
                create_dirs(path, content)
            else:
                # It's a file
                with open(path, 'w') as f:
                    f.write(content)
    
    # Create the structure
    create_dirs(".", structure)
    
    # Create requirements.txt content
    requirements = """streamlit
numpy
pandas
PyPDF2
sentence-transformers
faiss-cpu
scikit-learn
nltk
transformers
torch
ctransformers
huggingface-hub
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("Project structure created successfully!")
    print("Directories created:")
    print("- rag/")
    print("- models/")
    print("- utils/")
    print("- pages/")
    print("- Database/")
    print("- models/")
    print("\nMain files created:")
    print("- app.py (main application)")
    print("- auth.py (authentication)")
    print("- styles.py (UI styles)")
    print("- requirements.txt (dependencies)")
    print("\nNext steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Add your PDF files to the Database/ folder")
    print("3. Run: streamlit run app.py")

if __name__ == "__main__":
    create_project_structure()