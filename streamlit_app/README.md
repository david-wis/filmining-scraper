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

The model typically achieves:
- R² Score: 0.6-0.8 (depending on data quality)
- RMSE: 1.5-2.5 ROI units
- MAE: 1.0-1.8 ROI units

## 🎭 Features Used for Prediction

### Numerical Features
- Budget (log-transformed)
- Runtime
- Vote average and count
- Release year, month, quarter
- Budget per minute

### Categorical Features
- Genres (one-hot encoded)
- Main country (one-hot encoded)
- Original language (one-hot encoded)
- Movie status
- Adult content flag

### Engineered Features
- Vote confidence (log-transformed vote count)
- Rating-popularity score
- Budget and runtime categories
- Temporal features (decade, quarter)

## 🔍 Data Quality

The application applies data quality filters:
- Minimum budget: $100,000 (removes data errors)
- Valid ROI calculation
- Complete release date information
- Non-null essential fields

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

### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables
4. Deploy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- TMDB (The Movie Database) for providing the movie data
- Streamlit for the web framework
- Scikit-learn for machine learning capabilities
- Plotly for interactive visualizations

## 📞 Support

For questions or issues:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**Note**: This application is for educational and research purposes. Movie ROI predictions should not be used as the sole basis for investment decisions.


