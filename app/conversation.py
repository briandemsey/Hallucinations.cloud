"""
Conversation Management module
In-memory conversation store for maintaining context across queries.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

# In-memory conversation store
_conversations: Dict[str, 'Conversation'] = {}


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Represents a conversation with history."""
    id: str
    created_at: str
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'created_at': self.created_at,
            'messages': [asdict(m) for m in self.messages],
            'metadata': self.metadata
        }


def create_conversation(metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a new conversation and return its ID.

    Args:
        metadata: Optional metadata to attach to the conversation

    Returns:
        The new conversation ID
    """
    conversation_id = str(uuid.uuid4())
    _conversations[conversation_id] = Conversation(
        id=conversation_id,
        created_at=datetime.utcnow().isoformat(),
        messages=[],
        metadata=metadata or {}
    )
    return conversation_id


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Add a message to an existing conversation.

    Args:
        conversation_id: The conversation ID
        role: 'user' or 'assistant'
        content: Message content
        metadata: Optional message metadata

    Returns:
        True if successful, False if conversation not found
    """
    if conversation_id not in _conversations:
        return False

    message = Message(
        role=role,
        content=content,
        metadata=metadata or {}
    )
    _conversations[conversation_id].messages.append(message)
    return True


def get_history(conversation_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get the message history for a conversation.

    Args:
        conversation_id: The conversation ID

    Returns:
        List of message dicts, or None if conversation not found
    """
    if conversation_id not in _conversations:
        return None

    return [asdict(m) for m in _conversations[conversation_id].messages]


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Get full conversation data.

    Args:
        conversation_id: The conversation ID

    Returns:
        Conversation dict, or None if not found
    """
    if conversation_id not in _conversations:
        return None

    return _conversations[conversation_id].to_dict()


def format_context_for_query(conversation_id: str, max_messages: int = 10) -> str:
    """
    Format conversation history as context for a new query.

    Args:
        conversation_id: The conversation ID
        max_messages: Maximum number of previous messages to include

    Returns:
        Formatted context string for prepending to queries
    """
    if conversation_id not in _conversations:
        return ""

    messages = _conversations[conversation_id].messages[-max_messages:]

    if not messages:
        return ""

    context_parts = ["CONVERSATION HISTORY:"]
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        context_parts.append(f"\n{role_label}: {msg.content}")

    context_parts.append("\n\nCurrent question (answer based on the conversation context above):")
    return "\n".join(context_parts)


def export_conversation(conversation_id: str) -> Optional[str]:
    """
    Export conversation as a formatted text transcript.

    Args:
        conversation_id: The conversation ID

    Returns:
        Formatted transcript string, or None if not found
    """
    if conversation_id not in _conversations:
        return None

    conv = _conversations[conversation_id]
    lines = [
        "=" * 50,
        "H-LLM Multi-Model Conversation Transcript",
        f"Conversation ID: {conv.id}",
        f"Created: {conv.created_at}",
        "=" * 50,
        ""
    ]

    for msg in conv.messages:
        role_label = "USER" if msg.role == "user" else "ASSISTANT"
        lines.append(f"[{msg.timestamp}] {role_label}:")
        lines.append(msg.content)
        lines.append("")

    lines.append("=" * 50)
    lines.append("End of Transcript")
    lines.append("=" * 50)

    return "\n".join(lines)


def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation.

    Args:
        conversation_id: The conversation ID

    Returns:
        True if deleted, False if not found
    """
    if conversation_id in _conversations:
        del _conversations[conversation_id]
        return True
    return False


def conversation_exists(conversation_id: str) -> bool:
    """Check if a conversation exists."""
    return conversation_id in _conversations


def cleanup_old_conversations(max_age_hours: int = 24) -> int:
    """
    Remove conversations older than specified age.

    Args:
        max_age_hours: Maximum age in hours

    Returns:
        Number of conversations removed
    """
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    to_remove = []

    for conv_id, conv in _conversations.items():
        try:
            created = datetime.fromisoformat(conv.created_at.replace('Z', '+00:00').replace('+00:00', ''))
            if created < cutoff:
                to_remove.append(conv_id)
        except (ValueError, AttributeError):
            continue

    for conv_id in to_remove:
        del _conversations[conv_id]

    return len(to_remove)
