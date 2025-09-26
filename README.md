# 🧠 Hallucinations.cloud H-LLM Multi-Model Platform

A sophisticated AI comparison platform that queries multiple Large Language Models simultaneously and provides reliability scoring through the proprietary H-Score algorithm.

## 🚀 Features

- **8 AI Model Integrations**: GPT-4o, Claude 3, Gemini Pro, Cohere Command-R, Deepseek, WizardLM, Grok, Perplexity
- **H-Score Analysis**: Proprietary algorithm for measuring response reliability and consistency
- **Authentication System**: Twilio SMS verification and email authentication
- **Payment Integration**: Stripe subscription management with tiered pricing
- **Real-time Comparison**: Parallel querying of multiple AI models
- **Export Capabilities**: Download results in JSON and text formats
- **Modular Architecture**: Clean, maintainable codebase structure

## 📁 Project Structure

### Production Version
- `Hallucinations_9_23.py` - Main production application with full features
- `Hallucinations_9_23_FIXED.py` - Debug-friendly version for development

### Modular Architecture
```
├── main_modular.py          # Clean entry point
├── run_modular.py           # Development runner
├── config/                  # Configuration management
├── auth/                    # Authentication modules
├── models/                  # AI model integrations
├── analysis/                # H-Score calculation engine
├── ui/                      # User interface components
├── utils/                   # Utilities and performance
└── database/                # Database modules
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository
```bash
git clone <your-repo-url>
cd HallucinationsCloudProject
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables
```bash
cp .env.template .env
# Edit .env with your API keys
```

### Required API Keys
- **OpenAI**: GPT-4o integration
- **Anthropic**: Claude 3 integration
- **Google**: Gemini Pro integration
- **Cohere**: Command-R integration
- **Deepseek**: Deepseek Chat integration
- **OpenRouter**: WizardLM integration
- **Grok**: X.AI integration
- **Perplexity**: Sonar integration

### Optional Services
- **Stripe**: Payment processing (required for production)
- **Twilio**: SMS authentication (required for production)
- **Supabase**: Database backend (required for production)

## 🚀 Running the Application

### Development Mode (Modular)
```bash
python run_modular.py
```
Features:
- ✅ Demo authentication (5 queries)
- ✅ Clean modular architecture
- ✅ Fast development iteration

### Production Mode
```bash
streamlit run Hallucinations_9_23.py
```
Features:
- ✅ Full authentication system
- ✅ Stripe payment integration
- ✅ Supabase database
- ✅ Apple App Store review compatibility

## 🎯 H-Score Algorithm

The proprietary H-Score system evaluates AI responses based on:
- **Consistency**: Agreement between multiple models
- **Completeness**: Thoroughness of responses
- **Accuracy Indicators**: Detection of potential hallucinations
- **Contradiction Analysis**: Identification of conflicting information

Scores range from 0-10:
- **7-10**: High reliability - Strong consensus
- **5-7**: Moderate reliability - Some variation
- **0-5**: Low reliability - Significant issues detected

## 🍎 Apple App Store Integration

The application includes special authentication bypass for Apple App Store review:
- **Test Phone**: +13014426175
- **Verification Code**: 612485

This allows Apple's review team to test the application without requiring live SMS verification.

## 📊 Architecture Benefits

### Modular Version Advantages
- Individual component testing
- Faster development iteration
- Better code maintainability
- Team collaboration support
- Clear separation of concerns

### Production Version Features
- Comprehensive authentication
- Payment processing
- Database persistence
- Full feature set
- Production-ready stability

## 🔧 Development Workflow

1. **Feature Development**: Work in modular version
2. **Testing**: Use demo authentication for quick iteration
3. **Integration**: Port tested features to production
4. **Deployment**: Deploy via existing production pipeline

## 📝 API Documentation

### Supported Models
| Provider | Model | Status |
|----------|-------|--------|
| OpenAI | GPT-4o Mini | ✅ Active |
| Anthropic | Claude 3 Haiku | ✅ Active |
| Google | Gemini Pro | ✅ Active |
| Cohere | Command-R | ✅ Active |
| Deepseek | Deepseek Chat | 🔄 Pending |
| OpenRouter | WizardLM | 🔄 Pending |
| Grok | Grok-1 | 🔄 Pending |
| Perplexity | Sonar | 🔄 Pending |

## 💰 Pricing Tiers

- **Free**: 5 queries per session
- **Basic**: $9.99/month - 100 queries
- **Professional**: $29.99/month - 500 queries
- **Enterprise**: $99.99/month - Unlimited queries

## 🔐 Security Features

- Environment variable protection
- API key encryption
- Secure payment processing
- Phone verification system
- Content moderation integration

## 📈 Performance

- **Parallel Processing**: Simultaneous model querying
- **Caching System**: Response optimization
- **Session Management**: Efficient state handling
- **Memory Optimization**: Resource management

## 🤝 Contributing

This is a production application. For development:
1. Use the modular version for feature development
2. Test thoroughly before integration
3. Follow existing code patterns
4. Maintain security best practices

## 📄 License

Proprietary software - All rights reserved

## 🆘 Support

For technical support or feature requests, please contact the development team.

---

**Built with ❤️ using Streamlit, Python, and 8 powerful AI models**