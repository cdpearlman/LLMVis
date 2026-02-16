"""
Chatbot UI Components

Dash components for the AI chatbot interface including the floating icon,
chat window, message display, and input area.
"""

from dash import html, dcc
from typing import List, Dict, Optional


# Greeting message shown to new users when chat opens
GREETING_MESSAGE = """Hi there! I'm your AI assistant for exploring transformer models.

I can help you understand:
- How attention heads and layers process your input
- What various experiments can reveal about model behavior
- General transformer and ML concepts

Try asking: "What does attention head 0 in layer 1 do?" or "Why did ablating this head change the output?"
"""


def create_chat_icon():
    """
    Create the floating robot icon button.
    
    Returns:
        Dash HTML component for the chat toggle icon
    """
    return html.Button(
        html.I(className="fas fa-robot"),
        id="chat-toggle-btn",
        className="chat-toggle-btn",
        title="Open AI Assistant"
    )


def create_chat_header():
    """
    Create the chat window header with title and controls.
    
    Returns:
        Dash HTML component for the chat header
    """
    return html.Div([
        html.Div([
            html.I(className="fas fa-robot", style={'marginRight': '10px'}),
            html.Span("AI Assistant", style={'fontWeight': '500'})
        ], style={'display': 'flex', 'alignItems': 'center'}),
        
        html.Div([
            # Clear chat button
            html.Button(
                html.I(className="fas fa-trash-alt"),
                id="chat-clear-btn",
                className="chat-header-btn",
                title="Clear chat history"
            ),
            # Close button
            html.Button(
                html.I(className="fas fa-times"),
                id="chat-close-btn",
                className="chat-header-btn",
                title="Close chat"
            )
        ], style={'display': 'flex', 'gap': '8px'})
    ], className="chat-header")


def create_message_bubble(message: Dict, index: int) -> html.Div:
    """
    Create a single message bubble.
    
    Args:
        message: Dict with 'role' (user/assistant) and 'content'
        index: Message index for unique IDs
        
    Returns:
        Dash HTML component for the message bubble
    """
    is_user = message.get('role') == 'user'
    content = message.get('content', '')
    
    bubble_class = "chat-message user-message" if is_user else "chat-message assistant-message"
    
    # For assistant messages, add copy button
    copy_btn = None
    if not is_user:
        copy_btn = html.Button(
            html.I(className="fas fa-copy"),
            id={'type': 'copy-message-btn', 'index': index},
            className="copy-message-btn",
            title="Copy message",
            **{'data-content': content}
        )
    
    return html.Div([
        html.Div([
            # Message content - use dcc.Markdown for formatting
            dcc.Markdown(
                content,
                className="message-content",
                dangerously_allow_html=False
            ),
            copy_btn
        ], className=bubble_class)
    ], className="message-wrapper user-wrapper" if is_user else "message-wrapper assistant-wrapper")


def create_typing_indicator():
    """
    Create the typing indicator shown while AI is responding.
    
    Returns:
        Dash HTML component for typing indicator
    """
    return html.Div([
        html.Div([
            html.Span(className="typing-dot"),
            html.Span(className="typing-dot"),
            html.Span(className="typing-dot")
        ], className="typing-indicator")
    ], id="chat-typing-indicator", style={'display': 'none'})


def create_messages_container(messages: Optional[List[Dict]] = None):
    """
    Create the scrollable messages container.
    
    Args:
        messages: List of message dicts to display
        
    Returns:
        Dash HTML component for messages area
    """
    message_elements = []
    
    if messages:
        for i, msg in enumerate(messages):
            message_elements.append(create_message_bubble(msg, i))
    
    return html.Div([
        html.Div(
            message_elements,
            id="chat-messages-list",
            className="chat-messages-list"
        ),
        create_typing_indicator()
    ], id="chat-messages-container", className="chat-messages-container")


def create_input_area():
    """
    Create the chat input area with textarea and send button.
    
    Returns:
        Dash HTML component for input area
    """
    return html.Div([
        html.Div([
            dcc.Textarea(
                id="chat-input",
                placeholder="Ask about transformers, experiments, or ML concepts...",
                className="chat-input-textarea",
                persistence=False
            ),
            html.Button(
                html.I(className="fas fa-paper-plane"),
                id="chat-send-btn",
                className="chat-send-btn",
                title="Send message (Enter)"
            )
        ], className="chat-input-wrapper")
    ], className="chat-input-area")


def create_chat_window():
    """
    Create the full chat window component.
    
    Returns:
        Dash HTML component for the chat window
    """
    return html.Div([
        html.Div(className="chat-resize-handle", id="chat-resize-handle"),
        create_chat_header(),
        create_messages_container(),
        create_input_area()
    ], id="chat-window", className="chat-window", style={'display': 'none'})


def create_chatbot_container():
    """
    Create the complete chatbot container with icon and window.
    
    This component should be added to the main app layout.
    
    Returns:
        Dash HTML component containing all chatbot elements
    """
    return html.Div([
        # Stores for chat state
        dcc.Store(id='chat-history-store', storage_type='local', data=[
            {'role': 'assistant', 'content': GREETING_MESSAGE}
        ]),
        dcc.Store(id='chat-open-store', storage_type='memory', data=False),
        dcc.Store(id='chat-pending-message', storage_type='memory', data=None),
        
        # Interval for handling async responses
        dcc.Interval(id='chat-response-interval', interval=500, disabled=True),
        
        # Chat window
        create_chat_window(),
        
        # Floating toggle button
        create_chat_icon()
    ], id="chatbot-container", className="chatbot-container")


def render_messages(messages: List[Dict]) -> List:
    """
    Render a list of messages as Dash components.
    
    Args:
        messages: List of message dicts
        
    Returns:
        List of Dash HTML components
    """
    return [create_message_bubble(msg, i) for i, msg in enumerate(messages)]


def format_error_message(error: str) -> Dict:
    """
    Format an error as an assistant message.
    
    Args:
        error: Error message string
        
    Returns:
        Message dict with error styling
    """
    return {
        'role': 'assistant',
        'content': f"⚠️ {error}"
    }
