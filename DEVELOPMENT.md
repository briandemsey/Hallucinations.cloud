# Development Guide - Hallucinations.cloud

## Architecture Consolidation Completed ✅

The codebase has been successfully reorganized from a monolithic structure to a clean, modular architecture:

### New Directory Structure

```
hallucinations_cloud/
├── main_modular.py              # Clean entry point for development
├── Hallucinations_9_23.py       # Production version (unchanged)
├── run_modular.py               # Development runner script
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration management
├── auth/
│   ├── __init__.py
│   └── authentication.py       # Simplified authentication
├── models/
│   ├── __init__.py
│   └── ai_models.py            # All 8 AI model integrations
├── analysis/
│   ├── __init__.py
│   └── hscore.py               # H-Score calculation engine
├── ui/
│   ├── __init__.py
│   ├── sidebar.py              # User dashboard
│   ├── main_interface.py       # Query interface
│   └── results_display.py      # Results visualization
├── utils/
│   ├── __init__.py
│   └── performance.py          # Session management & caching
└── database/
    └── __init__.py
```

## Running the Applications

### Modular Version (Development)
```bash
# Using the runner script
python run_modular.py

# Or directly with Streamlit
streamlit run main_modular.py
```

### Production Version (Live)
```bash
streamlit run Hallucinations_9_23.py
```

## Key Differences

| Feature | Modular Version | Production Version |
|---------|----------------|-------------------|
| **Authentication** | Demo mode (5 queries) | Full Twilio SMS + Email |
| **Payments** | Not implemented | Stripe integration |
| **Database** | Session state only | Supabase integration |
| **Architecture** | Clean modular | Monolithic (3,000+ lines) |
| **Purpose** | Development/Testing | Live production |

## Development Workflow

### 1. Feature Development
- Work on features in the modular version
- Test individual components in isolation
- Use demo authentication for quick iteration

### 2. Integration Testing
- Test feature in modular environment
- Verify all imports and dependencies work
- Check UI/UX with demo data

### 3. Production Integration
- Port tested features to `Hallucinations_9_23.py`
- Test with full authentication system
- Deploy to production environment

## Component Overview

### Configuration (`config/settings.py`)
- API key management for all 8 AI models
- Client initialization (OpenAI, Anthropic, Google, Cohere)
- Subscription tier definitions
- Application-wide settings

### Authentication (`auth/authentication.py`)
- Simplified demo authentication
- Session state management
- Query limit tracking
- User information handling

### AI Models (`models/ai_models.py`)
- Unified interface for 8 AI models:
  - GPT-4o (OpenAI)
  - Claude 3 Haiku (Anthropic)
  - Gemini Pro (Google)
  - Command-R (Cohere)
  - Deepseek Chat
  - WizardLM (OpenRouter)
  - Grok (X.AI) - *pending implementation*
  - Perplexity Sonar
- Parallel querying capabilities
- Error handling and fallback logic

### H-Score Analysis (`analysis/hscore.py`)
- Proprietary scoring algorithm
- Consistency analysis between models
- Accuracy indicators detection
- Contradiction identification
- Response completeness evaluation

### User Interface Components
- **Sidebar** (`ui/sidebar.py`): User dashboard, settings, help
- **Main Interface** (`ui/main_interface.py`): Query input, model selection
- **Results Display** (`ui/results_display.py`): Response visualization, H-Score display

### Utilities (`utils/performance.py`)
- Session state initialization
- Performance monitoring
- Caching mechanisms
- Memory optimization

## Environment Setup

Create `.env` file with API keys:

```bash
# Required for core functionality
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Optional models (add as available)
GOOGLE_API_KEY=your_google_key
COHERE_API_KEY=your_cohere_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENROUTER_API_KEY=your_openrouter_key
GROK_API_KEY=your_grok_key
PERPLEXITY_API_KEY=your_perplexity_key

# Production only (not needed for modular version)
STRIPE_LIVE_SECRET_KEY=your_stripe_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Testing Individual Components

### Test Configuration
```python
from config.settings import get_api_keys, get_api_clients
keys = get_api_keys()
clients = get_api_clients()
print("Available API keys:", [k for k, v in keys.items() if v])
```

### Test AI Models
```python
from models.ai_models import AIModelsManager
manager = AIModelsManager()
available = manager.get_available_models()
result = manager.query_model("GPT-4o", "Hello, world!")
```

### Test H-Score
```python
from analysis.hscore import calculate_h_score
sample_responses = {
    "GPT-4o": {"response": "The sky is blue due to light scattering"},
    "Claude": {"response": "Blue sky results from Rayleigh scattering"}
}
score = calculate_h_score(sample_responses)
```

## Benefits of Modular Architecture

1. **Easier Development**: Work on individual components without affecting others
2. **Better Testing**: Unit test each module independently
3. **Faster Iteration**: Quick startup and testing with demo authentication
4. **Code Reusability**: Components can be reused in other projects
5. **Team Collaboration**: Multiple developers can work on different modules
6. **Maintainability**: Clear separation of concerns and responsibilities

## Migration Strategy

The production version (`Hallucinations_9_23.py`) remains unchanged to ensure stability. New features should be:

1. **Developed** in the modular version
2. **Tested** thoroughly with demo authentication
3. **Integrated** into production version once stable
4. **Deployed** following existing production pipeline

## Next Steps

- [ ] Add comprehensive unit tests for each module
- [ ] Implement proper logging throughout the application
- [ ] Add API rate limiting and retry logic
- [ ] Create database abstraction layer for easier testing
- [ ] Set up automated testing pipeline
- [ ] Add performance benchmarking
- [ ] Implement proper error monitoring

This modular architecture provides a solid foundation for continued development while maintaining the stability of the production system.