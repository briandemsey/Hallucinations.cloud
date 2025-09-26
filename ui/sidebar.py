# ui/sidebar.py
"""
Sidebar component for Hallucinations.cloud
User dashboard and navigation
"""

import streamlit as st
from auth.authentication import StreamlitHLLMAuth
from config.settings import get_subscription_tiers

def render_sidebar():
    """Render the application sidebar"""

    auth_system = StreamlitHLLMAuth()

    if not auth_system.is_authenticated():
        return

    user_info = auth_system.get_user_info()
    if not user_info:
        return

    # User info section
    st.header("👤 Dashboard")

    # User details
    with st.container():
        st.write(f"**Email:** {user_info.get('email', 'Unknown')}")
        st.write(f"**Subscription:** {st.session_state.get('subscription_tier', 'Demo')}")

        # Query usage
        queries_remaining = user_info.get('queries_remaining', 0)
        st.metric("Queries Left", queries_remaining)

        if queries_remaining <= 1:
            st.warning("⚠️ Low query count!")

    st.markdown("---")

    # Settings section
    st.header("⚙️ Settings")

    # Model preferences
    st.subheader("Model Preferences")
    show_errors = st.checkbox("Show API Errors", value=True)
    show_tokens = st.checkbox("Show Token Usage", value=False)

    # Analysis options
    st.subheader("Analysis Options")
    auto_hscore = st.checkbox("Auto-calculate H-Score", value=True)
    show_contradictions = st.checkbox("Show Contradictions", value=True)

    # Store preferences in session state
    st.session_state.show_errors = show_errors
    st.session_state.show_tokens = show_tokens
    st.session_state.auto_hscore = auto_hscore
    st.session_state.show_contradictions = show_contradictions

    st.markdown("---")

    # Subscription info
    st.header("💳 Subscription")

    tiers = get_subscription_tiers()
    current_tier = st.session_state.get('subscription_tier', 'demo')

    if current_tier == 'demo':
        st.info("🚀 Currently in Demo Mode")
        st.caption("Demo provides 5 queries for testing")
        st.caption("Production version has full subscription management")
    else:
        tier_info = tiers.get(current_tier, {})
        st.write(f"**Plan:** {current_tier.title()}")
        st.write(f"**Price:** ${tier_info.get('price', 0)}/month")

        daily_limit = tier_info.get('queries_per_day', 0)
        if daily_limit == -1:
            st.write("**Queries:** Unlimited")
        else:
            st.write(f"**Queries:** {daily_limit}/day")

    st.markdown("---")

    # Actions
    st.header("🔧 Actions")

    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        auth_system.logout()

    # Help section
    with st.expander("❓ Help & Support"):
        st.markdown("""
        **How to use Hallucinations.cloud:**

        1. **Enter your question** in the main interface
        2. **Choose query method**:
           - Query All Models: Gets responses from all available AI models
           - Query Single Model: Test with just one model
        3. **Review results** with H-Score analysis
        4. **Check contradictions** between different models

        **H-Score Explained:**
        - 0-3: Low reliability
        - 4-6: Moderate reliability
        - 7-8: High reliability
        - 9-10: Excellent reliability

        **For support:** Contact support@hallucinations.cloud
        """)

    # Debug info (if enabled)
    if st.checkbox("🐛 Debug Mode"):
        st.subheader("Debug Information")
        st.json({
            "session_state_keys": list(st.session_state.keys()),
            "user_info": user_info,
            "subscription_tier": current_tier
        })