"""
Authentication module - Twilio SMS OTP verification
"""
import os
from twilio.rest import Client
from datetime import datetime, timedelta
import jwt
from typing import Dict, Any

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID")

# JWT configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 30  # 30 days

# Initialize Twilio client
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as e:
        print(f"Warning: Twilio initialization failed: {e}")


async def send_otp(phone_number: str) -> bool:
    """
    Send OTP verification code to phone number via Twilio SMS

    Args:
        phone_number: Phone number in E.164 format (+1234567890)

    Returns:
        bool: True if OTP sent successfully, False otherwise
    """
    if not twilio_client:
        # Development mode - simulate sending OTP
        print(f"[DEV MODE] Would send OTP to {phone_number}. Use code: 123456")
        return True

    try:
        verification = twilio_client.verify \
            .v2 \
            .services(TWILIO_VERIFY_SERVICE_SID) \
            .verifications \
            .create(to=phone_number, channel='sms')

        return verification.status == 'pending'

    except Exception as e:
        print(f"Failed to send OTP: {str(e)}")
        return False


async def verify_otp_code(phone_number: str, code: str) -> Dict[str, Any]:
    """
    Verify OTP code and create user session with JWT tokens

    Args:
        phone_number: Phone number in E.164 format
        code: 6-digit OTP code

    Returns:
        dict: {
            "success": bool,
            "tokens": {"access": str, "refresh": str},
            "user": {"phone": str, "id": str}
        }
    """
    if not twilio_client:
        # Development mode - accept code 123456
        if code == "123456":
            tokens = generate_jwt_tokens(phone_number)
            return {
                "success": True,
                "tokens": tokens,
                "user": {
                    "phone": phone_number,
                    "id": f"user_{phone_number}",
                    "subscription_tier": "free"
                }
            }
        else:
            return {"success": False}

    try:
        verification_check = twilio_client.verify \
            .v2 \
            .services(TWILIO_VERIFY_SERVICE_SID) \
            .verification_checks \
            .create(to=phone_number, code=code)

        if verification_check.status == 'approved':
            # TODO: Check/create user in Supabase database
            tokens = generate_jwt_tokens(phone_number)

            return {
                "success": True,
                "tokens": tokens,
                "user": {
                    "phone": phone_number,
                    "id": f"user_{phone_number}",
                    "subscription_tier": "free"  # TODO: fetch from database
                }
            }
        else:
            return {"success": False}

    except Exception as e:
        print(f"OTP verification failed: {str(e)}")
        return {"success": False}


def generate_jwt_tokens(phone_number: str) -> Dict[str, str]:
    """
    Generate JWT access and refresh tokens

    Args:
        phone_number: User's phone number

    Returns:
        dict: {"access": str, "refresh": str}
    """
    now = datetime.utcnow()

    # Access token (24 hours)
    access_payload = {
        "phone": phone_number,
        "type": "access",
        "exp": now + timedelta(hours=24),
        "iat": now
    }
    access_token = jwt.encode(access_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    # Refresh token (30 days)
    refresh_payload = {
        "phone": phone_number,
        "type": "refresh",
        "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": now
    }
    refresh_token = jwt.encode(refresh_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {
        "access": access_token,
        "refresh": refresh_token
    }


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and extract payload

    Args:
        token: JWT token string

    Returns:
        dict: Token payload if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
