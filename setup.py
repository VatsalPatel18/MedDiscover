from setuptools import setup, find_packages

setup(
    name="vatsalpatel18_rag_llm_metabolomics",
    version="0.1.0",
    description="A tool for RAG-LLM performance analysis on research papers (metabolomics) with support for GPU and CPU modes.",
    author="Vatsal Patel",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
         "torch",
         "torchvision>=0.15.0,<0.30.0",
         "faiss-cpu",
         "PyPDF2",
         "transformers",
         "numpy",
         "openai",
         "python-dotenv",
         "gradio",
         "nltk",
         "rouge-score",
         "pandas",
         "scipy",
         "statsmodels",
         "datasets",
         "ragas"
    ],
    package_data={
        "med_discover_ai": [
            "eval_samples/sample_qa.csv",
            "eval_samples/sample_pdfs/*.pdf",
            "evaluation_runner_README.md",
        ]
    },
    entry_points={
        "console_scripts": [
            "meddiscover-eval=med_discover_ai.cli_eval:main",
        ]
    },
)
