# WhatsApp Conversation Analyzer

## Overview

A comprehensive WhatsApp chat analysis tool that combines machine learning, data science, and generative AI to provide deep insights into your conversations. This project goes beyond traditional message statistics to analyze mood trends, communication patterns, user personas, and conversation dynamics using advanced NLP techniques.

**Live Demo**: [WhatsApp Conversation Analyzer](https://whatsapp-conversation-analyzer.streamlit.app/)

## Features

### Core Analytics
- **Message Statistics**: Total messages, words, media shared, links shared
- **User Activity Analysis**: Most active users, message frequency patterns
- **Temporal Analysis**: Activity timelines, peak usage hours, monthly trends
- **Content Analysis**: Word clouds, most common words, emoji usage patterns

### Advanced AI-Powered Insights
- **Mood Trend Analysis**: Track emotional patterns over time using sentiment analysis
- **Apology & Gratitude Detection**: Identify frequency of apologies and expressions of gratitude
- **User Persona Analysis**: Generate personality profiles based on communication patterns
- **Conversation Flow Analysis**: Understand dialogue dynamics and response patterns
- **Topic Modeling**: Discover hidden themes and subjects in conversations

### Multi-Language Support
- **Language Detection**: Automatically detect and analyze conversations in multiple languages
- **Cross-Language Analytics**: Support for major languages including English, Hindi, Spanish, French, and more
- **Unicode Emoji Analysis**: Comprehensive emoji sentiment and usage analysis

### Data Visualization
- **Interactive Dashboards**: Built with Streamlit for seamless user experience
- **Dynamic Charts**: Activity heatmaps, timeline visualizations, and statistical plots
- **Word Clouds**: Customizable word frequency visualizations
- **Emoji Analytics**: Visual emoji usage patterns and sentiment mapping

## Technical Architecture

### Project Structure
```
Whatsapp_analysis/
├── app.py              # Streamlit application entrypoint
├── preprocessor.py     # Chat file parsing and data preprocessing
├── helper.py          # Statistical analysis, visualizations, and utility functions
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

### Technology Stack
- **Python 3.11**: Core programming language (strongly recommended)
- **Streamlit**: Web application framework for interactive dashboards
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Matplotlib/Seaborn**: Data visualization libraries
- **Plotly**: Interactive plotting and visualization
- **NLTK/spaCy**: Natural language processing
- **Scikit-learn**: Machine learning algorithms for pattern recognition
- **Emoji**: Emoji analysis and sentiment mapping
- **LangDetect**: Multi-language detection capabilities

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RajeebLochan/Whatsapp_analysis.git
   cd Whatsapp_analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage Guide

### Exporting WhatsApp Chat Data

1. **For Individual Chats**:
   - Open the chat in WhatsApp
   - Tap on the contact name → More → Export Chat
   - Choose "Without Media" for faster processing
   - Save the .txt file

2. **For Group Chats**:
   - Open the group chat
   - Tap on group name → More → Export Chat
   - Choose "Without Media"
   - Save the .txt file

### Using the Analyzer

1. **Upload Chat File**: Use the file uploader to select your exported .txt file
2. **Select Analysis Scope**: Choose between "Overall" analysis or specific user analysis
3. **Explore Insights**: Navigate through different analysis sections:
   - Statistical Overview
   - Activity Patterns
   - Content Analysis
   - AI-Powered Insights
   - Multi-Language Analytics

### Supported Chat Formats

The analyzer supports WhatsApp chat exports in multiple date formats:
- DD/MM/YYYY format
- MM/DD/YYYY format
- YYYY-MM-DD format
- 12-hour and 24-hour time formats

## Advanced Features

### Machine Learning Capabilities
- **Sentiment Analysis**: Tracks emotional tone using pre-trained models
- **Topic Modeling**: Uses LDA (Latent Dirichlet Allocation) for theme discovery
- **Clustering Analysis**: Groups similar conversation patterns
- **Predictive Analytics**: Forecast communication trends

### Generative AI Integration
- **Conversation Summarization**: AI-generated summaries of chat themes
- **Persona Generation**: Detailed personality profiles based on communication style
- **Insight Generation**: Automated discovery of interesting conversation patterns
- **Relationship Dynamics**: Analysis of interpersonal communication patterns

### Privacy and Security
- **Local Processing**: All data processing happens locally on your machine
- **No Data Storage**: Chat data is not stored or transmitted to external servers
- **Privacy First**: Your conversations remain completely private

## Performance Optimizations

- **Efficient Data Processing**: Optimized pandas operations for large chat files
- **Memory Management**: Chunked processing for handling extensive chat histories
- **Caching**: Streamlit caching for improved performance
- **Parallel Processing**: Multi-threaded analysis for faster results

## Contributing

We welcome contributions to improve the WhatsApp Conversation Analyzer! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Implement new features or fix bugs
4. **Test thoroughly**: Ensure all functionality works as expected
5. **Submit a pull request**: Describe your changes and their benefits

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## Roadmap

### Upcoming Features
- **Export Options**: PDF reports and data export functionality
- **Advanced ML Models**: Custom-trained models for better accuracy
- **Group Dynamics**: Enhanced group conversation analysis
- **Integration APIs**: REST API for external applications

### Version History
- **v1.0**: Initial release with basic analytics
- **v1.1**: Added multi-language support
- **v1.2**: Integrated AI-powered insights
- **v1.3**: Enhanced visualization capabilities

## Troubleshooting

### Common Issues
- **File Upload Errors**: Ensure the file is a valid WhatsApp .txt export
- **Memory Issues**: For large files, consider analyzing smaller date ranges
- **Encoding Problems**: Ensure the chat file is UTF-8 encoded

### Performance Tips
- Use Python 3.11 for optimal performance
- Close other applications to free up system memory
- For very large chats, consider splitting the analysis by date ranges

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Rajeeb Lochan**
- Twitter: [@rajeeb_thedev](https://x.com/rajeeb_thedev)
- LinkedIn: [rajeeb-lochan](https://www.linkedin.com/in/rajeeb-lochan/)
- GitHub: [@RajeebLochan](https://github.com/RajeebLochan)

## Acknowledgments

- Thanks to the open-source community for providing excellent libraries
- Special recognition to the Streamlit team for their amazing framework
- Appreciation to all contributors and users who have provided feedback

## Citation

If you use this tool in your research or projects, please cite:

```bibtex
@software{lochan2024whatsapp,
  title={WhatsApp Conversation Analyzer},
  author={Lochan, Rajeeb},
  year={2024},
  url={https://github.com/RajeebLochan/Whatsapp_analysis}
}
```

---

**Note**: This tool is for educational and personal use only. Please respect privacy and obtain consent before analyzing shared conversations.

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
