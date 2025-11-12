# Multimodal Fashion Search Engine with LanceDB

A powerful multimodal search engine for fashion products using LanceDB and CLIP embeddings. This application allows users to search for fashion items using both text queries and reference images.

## 🚀 Features

- **Multimodal Search**: Search using text descriptions or reference images
- **Vector Database**: Utilizes LanceDB for efficient similarity search
- **CLIP Embeddings**: Leverages OpenAI's CLIP model for understanding both images and text
- **Streamlit UI**: User-friendly web interface for easy interaction
- **Scalable**: Handles large datasets with efficient indexing

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/multimodal-search-engine.git
   cd multimodal-search-engine
   ```

2. **Create and activate conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate lance-env
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dataset Setup

1. Download the Myntra Fashion Product Dataset from [Kaggle](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset)
2. Create the following directory structure:
   ```
   input/
   ├── Images/
   │   ├── 0.jpg
   │   ├── 1.jpg
   │   └── ...
   └── Fashion Dataset.csv
   ```

## 🚀 Usage

### 1. Create the Vector Database
```bash
python src/make_table.py --database "~/.lancedb" --table_name "myntra" --data_path "input/Images" --num_samples 1000
```

### 2. Run the Streamlit App
```bash
streamlit run src/app.py -- --table_name myntra
```

### 3. Using the Web Interface
- Enter text queries in the search box
- Or upload an image to find similar items
- Adjust the number of results using the slider

### Command Line Search

#### Text Search
```bash
python src/vector_search.py --database ~/.lancedb --table_name myntra --schema "Myntra" --search_query "Blue Jeans" --output_folder "output"
```

#### Image Search
```bash
python src/vector_search.py --database ~/.lancedb --table_name myntra --schema "Myntra" --search_query "path/to/your/image.jpg" --output_folder "output"
```

## 📁 Project Structure

```
multimodal-search-engine/
├── src/
│   ├── app.py              # Streamlit web application
│   ├── make_table.py       # Database and table creation
│   ├── vector_search.py    # Core search functionality
│   ├── schema.py           # Database schema definition
│   └── embedding_model.py  # CLIP model integration
├── input/                  # Dataset directory (not included in git)
├── output/                 # Search results (not included in git)
├── .gitignore
├── environment.yml
└── README.md
```