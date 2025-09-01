# ⚡ Energy Advisor MVP

**Personal energy analyzer for Irish MPRN smart meter data**

A Streamlit-based web application that analyzes Irish smart meter data, detects appliance usage patterns, and provides AI-powered recommendations for energy savings.

## 🚀 Features

### ✅ **Implemented & Working**
- **Data Parser**: MPRN CSV format parsing with validation
- **Usage Analysis**: Daily/weekly patterns, peak detection, user classification
- **Enhanced Analytics**: Time-of-use breakdown, efficiency metrics
- **Streamlit UI**: Interactive web interface with 4 main pages
- **Sample Data**: 24-hour and 20-day sample datasets included
- **Comprehensive Testing**: 32 unit tests with 49% coverage

### 🚧 **In Development**
- **AI Integration**: OpenAI/Claude API integration for advanced analysis
- **Appliance Detection**: AI-powered identification of EV, washing, heating loads
- **Tariff Engine**: Cost calculations and tariff optimization
- **Recommendations**: AI-generated savings advice with €€ impact
- **Benchmarking**: Comparison with similar households

## 📊 Project Status

**Current Progress**: Day 1 Complete ✅ | Day 2 In Progress 🚧 | Day 3 Planned 📋

- **Module 1: File Parser & Validation** ✅ COMPLETED
- **Module 2: Tariff Engine** 🚧 NOT STARTED
- **Module 3: Usage Analysis Engine** ✅ COMPLETED
- **Module 4: AI-Powered Appliance Detection** 🚧 BASIC STRUCTURE
- **Module 5: AI-Powered Recommendation Engine** 🚧 PLACEHOLDER
- **Module 6: Benchmarking Engine** 🚧 NOT STARTED
- **Module 7: Streamlit UI** ✅ COMPLETED
- **Module 8: AI Integration & Testing** ✅ BASIC STRUCTURE

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone <your-repo-url>
cd energy_advisor_mvp

# Run the setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Run the application
python run_app.py
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python run_app.py
```

## 📁 Project Structure

```
energy_advisor_mvp/
├── run_app.py                 # Streamlit app launcher
├── src/
│   ├── streamlit_app.py       # Main Streamlit application
│   ├── data_parser.py         # MPRN file parsing & validation
│   ├── usage_analyzer.py      # Usage pattern analysis
│   └── ai_engine.py           # AI integration and analysis engine
├── data/
│   ├── sample_mprn.csv        # 24-hour sample MPRN data
│   └── twenty_day_sample_mprn_fixed.csv  # 20-day sample data
├── tests/
│   ├── test_data_parser.py    # Data parser testing
│   ├── test_streamlit_app.py  # Streamlit app testing
│   └── test_usage_analyzer.py # Usage analyzer testing
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🔧 Usage

### 1. **Data Upload & Analysis**
- Upload your MPRN CSV file or use sample data
- View data preview and validation results
- See basic statistics and data structure

### 2. **Usage Patterns**
- **Daily Pattern**: 24-hour usage profile
- **Weekly Pattern**: Day-of-week heatmap
- **Half-Hourly Pattern**: Detailed time-series view
- **Enhanced Analysis**: User classification, peak detection, efficiency metrics

### 3. **Appliance Detection** (Coming Soon)
- AI-powered identification of major appliances
- EV charging patterns, washing cycles, heating loads
- Solar generation analysis

### 4. **Recommendations** (Coming Soon)
- Tariff optimization suggestions
- Time-shifting recommendations
- Peak avoidance strategies
- Estimated savings calculations

## 📊 Sample Data

The project includes two sample datasets:

1. **24-Hour Sample** (`sample_mprn.csv`): 55 records, 0.95 kWh total
2. **20-Day Sample** (`twenty_day_sample_mprn_fixed.csv`): 920 records, 179 kWh total

Both datasets use the standard Irish MPRN format with 30-minute intervals.

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Current Test Status
- **Total Tests**: 32
- **Passing**: 32 ✅
- **Failing**: 0 ✅
- **Coverage**: 49%

## 🔮 AI Integration

The application is designed to integrate with:
- **OpenAI GPT-4**: For advanced pattern analysis and insights
- **Anthropic Claude**: Alternative AI provider for analysis

### Setup AI Integration
1. Create a `.env` file in the project root
2. Add your API key: `OPENAI_API_KEY=your_key_here`
3. Or: `ANTHROPIC_API_KEY=your_key_here`

## 📈 Performance

### Current Performance
- **24-hour data**: <1 second processing
- **20-day data**: <3 seconds processing
- **Memory usage**: <100MB for typical datasets
- **UI responsiveness**: <2 seconds for all interactions

### Target Performance
- **Annual data**: <30 seconds processing
- **Memory usage**: <500MB peak
- **Real-time analysis**: <5 seconds for AI insights

## 🚨 Known Issues & Limitations

### Current Limitations
- **Test Coverage**: Only 49% - needs improvement
- **Code Quality**: Some flake8 warnings need fixing
- **Missing Modules**: Tariff engine and benchmarking not implemented
- **AI Integration**: Basic structure only, needs API keys for full functionality

### Planned Improvements
- Complete test coverage to >90%
- Fix all code quality issues
- Implement missing core modules
- Add comprehensive error handling
- Performance optimization for large datasets

## 🤝 Contributing

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Test** thoroughly with `pytest`
5. **Format** code with `black`
6. **Lint** with `flake8`
7. **Submit** a pull request

### Code Standards
- **Formatting**: Black (88 character line length)
- **Linting**: flake8 with max-line-length=88
- **Testing**: pytest with >90% coverage target
- **Documentation**: Clear docstrings for all public functions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

### Phase 1 (Current) ✅
- [x] Core data pipeline
- [x] Basic usage analysis
- [x] Streamlit UI framework
- [x] Sample data and testing

### Phase 2 (Next) 🚧
- [ ] Tariff engine implementation
- [ ] AI-powered appliance detection
- [ ] Recommendation engine
- [ ] Enhanced testing and coverage

### Phase 3 (Final) 📋
- [ ] Benchmarking engine
- [ ] Performance optimization
- [ ] Production deployment
- [ ] User documentation

## 📞 Support

For questions or issues:
1. Check the [Issues](../../issues) page
2. Review the [blueprint.md](blueprint.md) for project details
3. Check [githubsubmit.md](githubsubmit.md) for quality gates

---

**Built for National AI Hackathon 2024** 🚀
**Status**: MVP Phase 1 Complete ✅ | Phase 2 In Progress 🚧 
