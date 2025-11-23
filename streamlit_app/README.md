# 🎬 Movie ROI Predictor

A Streamlit application that uses Random Forest machine learning to predict the Return on Investment (ROI) of movies based on various features like budget, runtime, genres, country, and ratings.

## 🚀 Features

- **ROI Prediction**: Predict movie ROI using machine learning
- **Interactive Data Analysis**: Explore the movie dataset with interactive visualizations
- **Model Training**: Train and optimize Random Forest models
- **Feature Importance**: Understand which factors most influence ROI
- **Real-time Predictions**: Get instant ROI predictions for new movies

## 📊 Dataset

The application uses movie data from TMDB (The Movie Database) including:
- 9,999+ movies with detailed information
- Financial data (budget, revenue, ROI)
- Movie metadata (genres, countries, languages, ratings)
- Temporal data (release dates, decades)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd streamlit_app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the database**:
   - Ensure PostgreSQL is running
   - Import the movie data using the provided backup.sql
   - Update database connection settings in `utils/database.py` if needed

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🗄️ Database Setup

The application requires a PostgreSQL database with movie data. Follow these steps:

1. **Start PostgreSQL** (using Docker):
   ```bash
   # From the main project directory
   ./scripts/docker-setup.sh start-db
   ```

2. **Import the data**:
   ```bash
   docker exec -i tmdb_movie_db pg_restore -U postgres -d movie_database < data/backup.sql
   ```

3. **Verify connection**:
   The app will automatically test the database connection on startup.

## 🎯 Usage

### 1. Home Page
- Overview of the dataset
- Key metrics and insights
- Quick visualizations

### 2. Predict ROI
- Input movie characteristics
- Get ROI predictions with confidence intervals
- View predicted revenue and profit

### 3. Data Analysis
- Interactive visualizations
- Financial analysis
- Genre and country analysis
- Temporal trends

### 4. Model Training
- Train Random Forest models
- Optimize hyperparameters
- View feature importance
- Save trained models

### 5. Model Performance
- View model metrics
- Analyze prediction accuracy
- Load/save models

## 🔧 Configuration

### Environment Variables

You can configure the database connection using environment variables:

```bash
export DB_HOST=localhost
export DB_PORT=25432
export DB_NAME=movie_database
export DB_USER=postgres
export DB_PASSWORD=postgres
```

### Model Parameters

The Random Forest model can be configured with:
- Number of estimators (trees)
- Maximum depth
- Minimum samples per split
- Minimum samples per leaf
- Random state for reproducibility

## 📈 Model Performance

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```


## 📝 License

This project is licensed under the MIT License.
