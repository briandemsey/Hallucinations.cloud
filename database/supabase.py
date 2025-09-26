# database.py
"""
H-LLM Database Integration with Supabase
Handles user data, H-Score history, and subscription management
Updated for Supabase Auth integration
"""

import os
from supabase import create_client, Client
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HLLMDatabase:
    def __init__(self):
        """Initialize Supabase client"""
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")  # Support both names
        
        if not self.url or not self.key:
            raise ValueError("Missing Supabase credentials in .env file. Need SUPABASE_URL and SUPABASE_KEY")
        
        print(f"Connecting to Supabase: {self.url[:30]}...")  # Debug info
        self.supabase: Client = create_client(self.url, self.key)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection and return status"""
        try:
            # Test connection by counting users
            users_response = self.supabase.table("users").select("count", count="exact").execute()
            queries_response = self.supabase.table("hscore_queries").select("count", count="exact").execute()
            
            # Test auth
            try:
                auth_test = self.supabase.auth.get_session()
                auth_status = "ready"
            except:
                auth_status = "error"
            
            return {
                "status": "connected",
                "message": "Database connection successful",
                "users_count": users_response.count,
                "queries_count": queries_response.count,
                "auth_status": auth_status
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database connection failed: {str(e)}",
                "users_count": 0,
                "queries_count": 0,
                "auth_status": "error"
            }
    
    # === USER MANAGEMENT ===
    def create_user(self, username: str, email: str, password_hash: str) -> Optional[Dict]:
        """Create a new user (legacy method - use create_user_profile for Supabase Auth)"""
        try:
            user_data = {
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "subscription_tier": "free",
                "queries_used_today": 0,
                "total_queries": 0,
                "created_at": datetime.utcnow().isoformat(),
                "email_verified": False,
                "phone_verified": False
            }
            
            response = self.supabase.table("users").insert(user_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error creating user: {e}")
            return None
    
    def create_user_profile(self, user_data: Dict) -> Optional[Dict]:
        """Create user profile in the users table (for Supabase Auth users)"""
        try:
            # Ensure required fields
            profile_data = {
                "id": user_data["id"],  # Supabase Auth ID
                "username": user_data["username"],
                "email": user_data["email"],
                "subscription_tier": user_data.get("subscription_tier", "free"),
                "queries_used_today": 0,
                "total_queries": 0,
                "created_at": user_data.get("created_at", datetime.utcnow().isoformat()),
                "email_verified": user_data.get("email_verified", False),
                "phone_verified": user_data.get("phone_verified", False),
                "phone": user_data.get("phone")
            }
            
            response = self.supabase.table("users").insert(profile_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error creating user profile: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            response = self.supabase.table("users").select("*").eq("username", username).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        try:
            response = self.supabase.table("users").select("*").eq("email", email).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        try:
            response = self.supabase.table("users").select("*").eq("id", user_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None
    
    def update_user_verification(self, user_id: str, email_verified: bool = None, phone_verified: bool = None) -> bool:
        """Update user verification status"""
        try:
            update_data = {}
            if email_verified is not None:
                update_data["email_verified"] = email_verified
            if phone_verified is not None:
                update_data["phone_verified"] = phone_verified
                
            if update_data:
                self.supabase.table("users").update(update_data).eq("id", user_id).execute()
            return True
        except Exception as e:
            print(f"Error updating verification status: {e}")
            return False
    
    def update_user_queries(self, user_id: str, increment: int = 1) -> bool:
        """Update user query counts"""
        try:
            # Get current user data
            user = self.supabase.table("users").select("*").eq("id", user_id).execute()
            if not user.data:
                return False
            
            current_user = user.data[0]
            today = datetime.utcnow().date()
            last_query_date_str = current_user.get("last_query_date")
            
            # Handle last_query_date
            if last_query_date_str:
                last_query_date = datetime.fromisoformat(last_query_date_str.replace('Z', '+00:00')).date()
            else:
                last_query_date = today
            
            # Reset daily count if new day
            queries_today = current_user.get("queries_used_today", 0)
            if today > last_query_date:
                queries_today = 0
            
            update_data = {
                "queries_used_today": queries_today + increment,
                "total_queries": current_user.get("total_queries", 0) + increment,
                "last_query_date": datetime.utcnow().isoformat()
            }
            
            self.supabase.table("users").update(update_data).eq("id", user_id).execute()
            return True
        except Exception as e:
            print(f"Error updating user queries: {e}")
            return False
    
    def check_user_limits(self, user_id: str) -> Dict[str, Any]:
        """Check if user can make more queries today"""
        try:
            user = self.supabase.table("users").select("*").eq("id", user_id).execute()
            if not user.data:
                return {"can_query": False, "reason": "User not found"}
            
            user_data = user.data[0]
            subscription_tier = user_data.get("subscription_tier", "free")
            queries_today = user_data.get("queries_used_today", 0)
            
            # Check if it's a new day
            today = datetime.utcnow().date()
            last_query_date_str = user_data.get("last_query_date")
            if last_query_date_str:
                last_query_date = datetime.fromisoformat(last_query_date_str.replace('Z', '+00:00')).date()
                if today > last_query_date:
                    queries_today = 0  # Reset count for new day
            
            # Get subscription limits
            limits = self.get_subscription_limits(subscription_tier)
            daily_limit = limits.get("daily_queries", 10)
            
            if queries_today >= daily_limit:
                return {
                    "can_query": False,
                    "reason": f"Daily limit of {daily_limit} queries reached",
                    "queries_used": queries_today,
                    "daily_limit": daily_limit
                }
            
            return {
                "can_query": True,
                "queries_used": queries_today,
                "daily_limit": daily_limit,
                "remaining": daily_limit - queries_today
            }
        except Exception as e:
            print(f"Error checking user limits: {e}")
            return {"can_query": False, "reason": "Database error"}
    
    # === H-SCORE QUERY MANAGEMENT ===
    def save_hscore_query(self, user_id: str, query_data: Dict) -> Optional[str]:
        """Save an H-Score query with all components"""
        try:
            hscore_data = {
                "user_id": user_id,
                "user_input": query_data.get("user_input", ""),
                "ai_response": query_data.get("ai_response", ""),
                "confidence_score": query_data.get("confidence_score", 0),
                "factual_accuracy": query_data.get("factual_accuracy", 0),
                "source_reliability": query_data.get("source_reliability", 0),
                "reasoning_quality": query_data.get("reasoning_quality", 0),
                "final_hscore": query_data.get("final_hscore", 0),
                "analysis_breakdown": json.dumps(query_data.get("analysis_breakdown", {})),
                "model_used": query_data.get("model_used", "unknown"),
                "processing_time": query_data.get("processing_time", 0),
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = self.supabase.table("hscore_queries").insert(hscore_data).execute()
            return response.data[0]["id"] if response.data else None
        except Exception as e:
            print(f"Error saving H-Score query: {e}")
            return None
    
    def get_user_hscore_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's H-Score query history"""
        try:
            response = self.supabase.table("hscore_queries")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data if response.data else []
        except Exception as e:
            print(f"Error getting H-Score history: {e}")
            return []
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        try:
            queries = self.get_user_hscore_history(user_id, limit=1000)
            
            if not queries:
                return {
                    "total_queries": 0,
                    "average_hscore": 0,
                    "best_hscore": 0,
                    "worst_hscore": 0,
                    "trend": "No data"
                }
            
            hscores = [q["final_hscore"] for q in queries if q["final_hscore"]]
            
            # Calculate recent trend (last 10 vs previous 10)
            trend = "Stable"
            if len(hscores) >= 20:
                recent_avg = sum(hscores[:10]) / 10
                previous_avg = sum(hscores[10:20]) / 10
                if recent_avg > previous_avg + 5:
                    trend = "Improving"
                elif recent_avg < previous_avg - 5:
                    trend = "Declining"
            
            return {
                "total_queries": len(queries),
                "average_hscore": round(sum(hscores) / len(hscores), 1) if hscores else 0,
                "best_hscore": max(hscores) if hscores else 0,
                "worst_hscore": min(hscores) if hscores else 0,
                "trend": trend,
                "last_query": queries[0]["created_at"] if queries else None
            }
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {"error": str(e)}
    
    # === SUBSCRIPTION MANAGEMENT ===
    def get_subscription_limits(self, tier: str) -> Dict[str, Any]:
        """Get subscription tier limits"""
        # Hardcoded limits for now - you can move to database table later
        limits = {
            "free": {
                "tier_name": "free",
                "daily_queries": 10,
                "monthly_queries": 300,
                "features": ["basic_hscore"],
                "price_monthly": 0
            },
            "premium": {
                "tier_name": "premium",
                "daily_queries": 100,
                "monthly_queries": 3000,
                "features": ["basic_hscore", "advanced_analysis", "export"],
                "price_monthly": 19.99
            },
            "pro": {
                "tier_name": "pro",
                "daily_queries": 1000,
                "monthly_queries": 30000,
                "features": ["basic_hscore", "advanced_analysis", "export", "api_access"],
                "price_monthly": 99.99
            },
            "demo": {
                "tier_name": "demo",
                "daily_queries": 3,
                "monthly_queries": 3,
                "features": ["basic_hscore"],
                "price_monthly": 0
            }
        }
        
        return limits.get(tier, limits["free"])
    
    def upgrade_user_subscription(self, user_id: str, new_tier: str) -> bool:
        """Upgrade user subscription tier"""
        try:
            update_data = {
                "subscription_tier": new_tier,
                "subscription_updated_at": datetime.utcnow().isoformat()
            }
            
            self.supabase.table("users").update(update_data).eq("id", user_id).execute()
            return True
        except Exception as e:
            print(f"Error upgrading subscription: {e}")
            return False
    
    # === SESSION MANAGEMENT ===
    def create_session(self, user_id: str, session_token: str) -> bool:
        """Create a new user session"""
        try:
            session_data = {
                "user_id": user_id,
                "session_token": session_token,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "is_active": True
            }
            
            self.supabase.table("user_sessions").insert(session_data).execute()
            return True
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate a session token and return user data"""
        try:
            response = self.supabase.table("user_sessions")\
                .select("*, users(*)")\
                .eq("session_token", session_token)\
                .eq("is_active", True)\
                .gt("expires_at", datetime.utcnow().isoformat())\
                .execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error validating session: {e}")
            return None
    
    def end_session(self, session_token: str) -> bool:
        """End a user session"""
        try:
            self.supabase.table("user_sessions")\
                .update({"is_active": False})\
                .eq("session_token", session_token)\
                .execute()
            return True
        except Exception as e:
            print(f"Error ending session: {e}")
            return False
    
    # === ANALYTICS ===
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global platform statistics"""
        try:
            users_count = self.supabase.table("users").select("count", count="exact").execute().count
            queries_count = self.supabase.table("hscore_queries").select("count", count="exact").execute().count
            
            # Get average H-Score
            queries = self.supabase.table("hscore_queries").select("final_hscore").execute()
            hscores = [q["final_hscore"] for q in queries.data if q["final_hscore"]]
            avg_hscore = sum(hscores) / len(hscores) if hscores else 0
            
            return {
                "total_users": users_count,
                "total_queries": queries_count,
                "average_hscore": round(avg_hscore, 1),
                "active_today": self.get_active_users_today()
            }
        except Exception as e:
            print(f"Error getting global stats: {e}")
            return {"error": str(e)}
    
    def get_active_users_today(self) -> int:
        """Get count of users active today"""
        try:
            today = datetime.utcnow().date().isoformat()
            response = self.supabase.table("users")\
                .select("count", count="exact")\
                .gte("last_query_date", today)\
                .execute()
            
            return response.count
        except Exception as e:
            print(f"Error getting active users: {e}")
            return 0
    
    def get_recent_queries(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent queries for a user"""
        return self.get_user_hscore_history(user_id, limit)


# Database singleton
_database_instance = None

def get_database() -> HLLMDatabase:
    """Get or create database instance"""
    global _database_instance
    if _database_instance is None:
        _database_instance = HLLMDatabase()
    return _database_instance


# Utility functions for easy integration
def hash_password(password: str) -> str:
    """Hash a password for storage (legacy - use Supabase Auth instead)"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash (legacy - use Supabase Auth instead)"""
    return hash_password(password) == password_hash