# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hallucinations.cloud** - A multi-model AI comparison platform that queries 8 different LLMs simultaneously to detect hallucinations, contradictions, and provide reliability scoring through a proprietary H-Score algorithm.

- **Current Stack**: Python/Streamlit web application
- **Deployment**: AWS via GitHub → Render pipeline
- **Live URL**: http://hallucinations-alb-1094880033.us-east-1.elb.amazonaws.com
- **Business**: Hallucinations.cloud LLC with Stripe payments and Twilio authentication

## Architecture Overview

### Current State
The project has been **successfully reorganized** into a clean modular architecture:
- **Production Application**: `Hallucinations_9_23.py` - Main production file (unchanged for stability)
- **Modular Development**: `main_modular.py` - Clean modular version for development
- **Organized Structure**: All components properly organized into directories
- **Development Tools**: `run_modular.py` script and `DEVELOPMENT.md` guide

### Supported AI Models (8 total)
1. OpenAI GPT-4o
2. Anthropic Claude 3
3. Google Gemini Pro
4. Cohere Command-R
5. Deepseek Chat
6. OpenRouter (Microsoft WizardLM)
7. xAI Grok
8. Perplexity Sonar

### Core Components Architecture
```
Authentication System (Twilio SMS + Email)
    ↓
AI Models Manager (8 LLM integrations)
    ↓
H-Score Analysis Engine (Proprietary scoring)
    ↓
Results Display (Red/Blue/Purple team analysis)
    ↓
Supabase Database (User data + query history)
```

## Development Commands

### Running the Application
```bash
# Production version (full features)
streamlit run Hallucinations_9_23.py

# Modular development version (clean architecture)
python run_modular.py
# OR directly: streamlit run main_modular.py

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
Create `.env` file with:
```bash
# Core AI APIs
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
COHERE_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
OPENROUTER_API_KEY=your_key
GROK_API_KEY=your_key
PERPLEXITY_API_KEY=your_key

# Authentication & Payments
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_VERIFY_SERVICE_SID=your_service_sid
STRIPE_LIVE_SECRET_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

### Testing Commands
```bash
# Test Supabase connection
python test_supabase.py

# Test database connection
python test_database_connection.py

# Debug authentication
python debug_supabase_connection.py
```

## Key Technical Details

### Database Schema (Supabase)
- **users**: Authentication, profiles, subscription status
- **queries**: Query history, responses, H-scores
- **conversations**: Chat history and context

### Authentication Flow
1. Phone verification (Twilio SMS)
2. Email confirmation
3. Stripe subscription check
4. Session management via Streamlit

### H-Score Calculation
Proprietary algorithm analyzing:
- Response consistency across models
- Factual accuracy detection
- Contradiction identification
- Confidence scoring (0-10 scale)

### Subscription Tiers
- **Free Trial**: 3 days, 5 queries/day
- **Consumer**: $9.99/month, 25 queries/day
- **Professional**: $29.99/month, unlimited
- **Enterprise**: $99.99/month, API access

## File Structure Guidance

### Current Organization
```
/ (root directory)
├── Hallucinations_9_23.py          # Main production app
├── main_modular.py                 # Modular development version
├── run_modular.py                  # Development runner script
├── requirements.txt                # Dependencies
├── DEVELOPMENT.md                  # Development guide
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration management
├── auth/
│   ├── __init__.py
│   └── authentication.py          # Authentication system
├── models/
│   ├── __init__.py
│   └── ai_models.py               # AI model integrations
├── analysis/
│   ├── __init__.py
│   └── hscore.py                  # H-Score calculation
├── ui/
│   ├── __init__.py
│   ├── sidebar.py                 # User dashboard
│   ├── main_interface.py          # Query interface
│   └── results_display.py         # Results visualization
├── utils/
│   ├── __init__.py
│   └── performance.py             # Performance utilities
├── database/
│   └── __init__.py
└── [legacy files in root]         # Original files (preserved)
```

### Working with the Codebase
1. **Production Changes**: Use `Hallucinations_9_23.py` for live system updates
2. **Feature Development**: Use `main_modular.py` for new feature development
3. **Component Testing**: Individual modules in organized directories
4. **Development Workflow**: See `DEVELOPMENT.md` for detailed guidance

### Development vs Production

| Feature | Modular Version | Production Version |
|---------|----------------|-------------------|
| **File** | `main_modular.py` | `Hallucinations_9_23.py` |
| **Architecture** | Clean modular | Monolithic |
| **Authentication** | Demo (5 queries) | Full Twilio SMS |
| **Payments** | Not implemented | Stripe integration |
| **Database** | Session state | Supabase |
| **Purpose** | Development/Testing | Live production |

### Common Development Patterns
- **Streamlit Session State**: Used extensively for user state management
- **Error Handling**: Try-catch blocks around all external API calls
- **Caching**: Streamlit caching for performance optimization
- **Responsive Design**: Mobile-friendly Streamlit components

## Important Notes

### Security Considerations
- API keys managed via environment variables
- Phone verification for authentication
- Stripe for secure payment processing
- Session token validation

### Performance Optimizations
- Parallel API calls to all 8 models
- Streamlit caching for repeated queries
- Database connection pooling
- Response streaming for large outputs

### Deployment Pipeline
GitHub → Render → AWS (load balancer configured)

## iOS App Development
- **Approach**: WebView wrapper using Capacitor
- **Status**: Ready for App Store deployment
- **Architecture**: iPhone App → WebView → hallucinations.cloud → Python Backend → 8 LLMs

## Development Workflow

**Recommended approach for new features:**

1. **Develop** in modular version (`main_modular.py`) for clean architecture
2. **Test** thoroughly with demo authentication and sample data
3. **Integrate** stable features into production version (`Hallucinations_9_23.py`)
4. **Deploy** following existing production pipeline

Use `python run_modular.py` for rapid development iteration. The modular structure provides better code organization, easier testing, and clearer separation of concerns while maintaining production system stability.