# Stock Price Forecasting using Sentiment and Deep Learning Models

This project explores the impact of news sentiment on stock price prediction using advanced deep learning architectures including GRU, GAN, and WGAN-GP. The entire workflow is implemented across multiple Jupyter Notebooks with clearly defined stages for data preparation, sentiment analysis, merging, and model training.

---

## 📁 Project Workflow

### 1. **Data Importing**

**Notebook**: `Importing_dataset.ipynb`

- Stock price and news datasets are imported and stored in the `Data/` directory.

---

### 2. **Data Preprocessing**

**Notebook**: `data_preprocessing.ipynb`

- Preprocessing is performed on both sentiment and stock price datasets.
- Cleaned and structured data is stored back into the `Data/` directory.

---

### 3. **Sentiment Analysis**

**Notebook**: `Sentiment_Analysis_using_Finbert.ipynb`

- Sentiment scores are computed from the news data using FinBERT.
- The results are stored in the `Data/` directory for further use.

---

### 4. **Combining Sentiment and Price Data**

**Notebook**: `News_and_stock_price_data_combined.ipynb`

- Sentiment scores are merged with stock prices.
- The combined dataset is saved in `Data/`.
- Training and testing datasets are prepared and saved as `.npy` files in the `Model_train_data/` directory.

---

## 🧠 Model Training

### 5. **GRU Model**

**Notebook**: `GRU.ipynb`

- GRU model is trained using the combined dataset.
- Training loss is stored in the `Training_loss/` directory.
- The trained model is saved in the `Model/` directory.

---

### 6. **GAN Model**

**Notebook**: `GAN.ipynb`

- GAN model is trained on the prepared dataset.
- Losses and outputs are saved in `Training_loss/` and the model is stored in `Model/`.

---

### 7. **WGAN-GP Model**

**Notebook**: `WGAN_GP.ipynb`

- WGAN-GP model is trained similarly.
- Training results are saved in `Training_loss/`, and the model is saved in `Model/`.

---

## 🛠️ Libraries & Dependencies

To be filled in later...

```bash
# === Data Handling and Manipulation ===
- pandas
- numpy
- os
- csv
- datetime

# === Plotting and Visualization ===
- matplotlib.pyplot
- matplotlib.dates.DateFormatter
- pyplot

# === Machine Learning & Deep Learning Libraries ===
- torch
- tensorflow
- tensorflow.keras.models.Sequential
- tensorflow.keras.models.load_model
- tensorflow.keras.layers.Dense
- tensorflow.keras.layers.Dropout
- tensorflow.keras.layers.Bidirectional
- tensorflow.keras.layers.BatchNormalization
- tensorflow.keras.layers.Embedding
- tensorflow.keras.layers.TimeDistributed
- tensorflow.keras.layers.LeakyReLU
- tensorflow.keras.layers.GRU
- tensorflow.keras.layers.LSTM
- tensorflow.keras.layers.Flatten
- tensorflow.keras.layers.Conv1D
- tensorflow.keras.layers.ELU
- tensorflow.keras.layers.ReLU
- tensorflow.keras.optimizers.Adam
- tensorflow.keras.regularizers
- tensorflow.keras.losses.mean_squared_error
- tensorflow.python.client.device_lib

# === Scikit-Learn Utilities ===
- sklearn.preprocessing.LabelEncoder
- sklearn.preprocessing.MinMaxScaler
- sklearn.preprocessing.OneHotEncoder
- sklearn.metrics.mean_squared_error

# === Transformers and NLP ===
- transformers.AutoTokenizer
- transformers.AutoModelForSequenceClassification

# === Progress Display ===
- tqdm
- trange

# === Serialization ===
- pickle (load, dump)

# === External Data Sources ===
- alpha_vantage.timeseries.TimeSeries
- feedparser
```

## References:

Lin, H., Chen, C., Huang, G. & Jafari, A. (2021). Stock price prediction using Generative Adversarial Networks. Journal of Computer Science, 17(3), 188-196. https://doi.org/10.3844/jcssp.2021.188.196
