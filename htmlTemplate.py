# Updated CSS
# CSS Styles
css = '''
<style>
    /* Styling for the chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        border: 1px solid #d3d3d3; /* Add a subtle border */
    }
    /* Styling for user messages */
    .chat-message.user {
        background-color: #2b313e;
    }
    /* Styling for bot messages */
    .chat-message.bot {
        background-color: #475063;
    }
    /* Styling for the avatar */
    .chat-message .avatar {
        width: 15%; /* Adjust avatar size */
    }
    /* Styling for the avatar image */
    .chat-message .avatar img {
        max-width: 60px;
        max-height: 60px;
        border-radius: 50%;
        object-fit: cover;
    }
    /* Styling for the message content */
    .chat-message .message {
        width: auto; /* Adjust message width */
        padding: 0.75rem;
        color: #fff;
        margin-right: 1rem; /* Add margin to the left of the message */
        margin-left: 1rem;
    }
    /* Styling for strong (name) in the message */
    .chat-message .message strong {
        margin-right: 0.25rem; /* Adjust the margin as needed */
    }
</style>
'''

# HTML Templates for Bot and User Messages
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/3pvQJ2B/bot-icon.jpg">
    </div>
    <div class="message">
        <strong>Model:</strong> {{MSG}}
    </div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/HY8rRpL/human.jpg">
    </div>    
    <div class="message">
        <strong>User:</strong> {{MSG}}
    </div>
</div>
'''
