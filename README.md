# WhatsApp Analysis with ML & Generative AI

## Overview

A comprehensive WhatsApp chat analysis application that combines traditional data science techniques with advanced machine learning and generative AI capabilities. This project goes beyond basic statistics to provide deep insights into conversation patterns, sentiment analysis, and behavioral trends using state-of-the-art LLMs.

## Key Features

### Advanced Analytics
- **Sentiment Analysis**: Real-time emotion detection using machine learning models
- **Mood Trend Analysis**: Temporal analysis of conversation sentiment patterns
- **Apology & Gratitude Frequency**: Automated detection and quantification of polite expressions
- **Persona Analysis**: AI-driven personality profiling based on messaging patterns
- **Conversation Insights**: Deep learning-powered conversation theme extraction

### Traditional Statistics
- Message frequency analysis and temporal patterns
- User activity heatmaps and timeline visualizations
- Word cloud generation with customizable filtering
- Emoji usage analysis and sentiment correlation
- Media sharing patterns and file type distribution

### Generative AI Integration
- LLM-powered conversation summaries
- Automated insight generation
- Behavioral pattern recognition
- Contextual conversation analysis

## Technical Architecture

```
Whatsapp_analysis/
├── app.py             # Streamlit entrypoint and UI orchestration
├── preprocessor.py    # Chat file parsing and data preprocessing
├── helper.py          # Statistics, visualizations, ML models
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Installation & Setup

### Prerequisites
- **Python 3.11** (strongly recommended for optimal performance)
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/RajeebLochan/Whatsapp_analysis.git
   cd Whatsapp_analysis
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

### Data Preparation
1. Export your WhatsApp chat from the mobile app
2. Select "Without Media" for faster processing
3. Upload the exported `.txt` file to the application

### Analysis Features
- **Individual Analysis**: Focus on specific participants
- **Group Analysis**: Comprehensive group conversation insights
- **Temporal Analysis**: Time-based pattern recognition
- **Comparative Analysis**: Multi-user behavior comparison

## Technical Implementation

### Data Processing Pipeline
- **Preprocessing**: Robust text cleaning and normalization
- **Feature Engineering**: Advanced NLP feature extraction
- **Model Integration**: Seamless ML model deployment
- **Visualization**: Interactive data visualization with Plotly/Matplotlib

### Machine Learning Components
- Sentiment classification using transformer models
- Time series analysis for trend detection
- Clustering algorithms for conversation segmentation
- Named entity recognition for context extraction

### Generative AI Features
- Integration with modern LLMs for conversation analysis
- Automated report generation
- Contextual insight extraction
- Natural language summaries

## Core Modules

### `preprocessor.py`
- WhatsApp chat file parsing
- Data structure normalization
- Timestamp handling and timezone conversion
- Message type classification

### `helper.py`
- Statistical analysis functions
- Visualization generators
- ML model implementations
- AI-powered insight generation

### `app.py`
- Streamlit interface design
- User interaction management
- Real-time analysis orchestration
- Report generation and export

## Performance Optimizations

- Efficient data processing for large chat files
- Caching mechanisms for repeated analyses
- Optimized visualization rendering
- Memory-efficient model loading

## Privacy & Security

- Local processing ensures data privacy
- No external data transmission
- Secure file handling protocols
- Optional data anonymization features

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
```

## Technical Requirements

- **Python**: 3.11+ (recommended)
- **Memory**: 4GB+ RAM for large datasets
- **Storage**: 1GB+ free space for dependencies
- **Browser**: Modern web browser with JavaScript enabled

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Rajeeb Lochan**
- Twitter: [@rajeeb_thedev](https://x.com/rajeeb_thedev)
- LinkedIn: [rajeeb-lochan](https://www.linkedin.com/in/rajeeb-lochan/)
- GitHub: [RajeebLochan](https://github.com/RajeebLochan)

## Acknowledgments

This project represents a novel approach to WhatsApp chat analysis, combining traditional data science methodologies with cutting-edge AI technologies to provide unprecedented insights into digital communication patterns.

## Future Enhancements

- Real-time chat monitoring capabilities
- Advanced NLP model fine-tuning
- Multi-language support expansion
- Integration with additional messaging platforms
- Cloud deployment options

---

*This project demonstrates the intersection of data science, machine learning, and generative AI in practical applications, providing users with comprehensive insights into their digital communication patterns.*
