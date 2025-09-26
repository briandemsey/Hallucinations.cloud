# main.py imports everything:
from config.settings import initialize_configuration
from auth.authentication import require_authentication
from ui.sidebar import render_sidebar
from ui.main_interface import render_main_interface

# Each module imports what it needs:
# auth/authentication.py imports database, utils
# ui/sidebar.py imports auth, config
# models/ai_models.py imports config only
