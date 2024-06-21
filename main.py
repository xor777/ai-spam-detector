import asyncio
import logging
import os
import json
from collections import defaultdict, deque
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)

spam_logger = logging.getLogger('spam_logger')
spam_handler = logging.FileHandler('spam_log.txt', encoding='utf-8')
spam_handler.setLevel(logging.INFO)
spam_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
spam_logger.addHandler(spam_handler)

def load_safe_users(file_path='safe_users.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return defaultdict(lambda: {'count': 0, 'username': None}, data)
    except FileNotFoundError:
        return defaultdict(lambda: {'count': 0, 'username': None})
    except json.JSONDecodeError:
        logger.error("Error decoding JSON from file.")
        return defaultdict(lambda: {'count': 0, 'username': None})

def save_safe_users(data, file_path='safe_users.json'):
    data_to_save = {user_id: info for user_id, info in data.items()}
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_to_save, file)

safe_messages_count = load_safe_users()
recent_messages = defaultdict(lambda: deque(maxlen=5))

client = OpenAI(api_key=OPENAI_API_KEY)

def is_spam(message, context):
    prompt = f"""
    You are an AI assistant trained to identify potential spam messages in a Telegram chat dedicated to automobiles and lifestyle. 
    While the chat mainly focuses on car-related topics, members can discuss various subjects.
    Your task is to carefully analyze a specific message within the context of the ongoing conversation 
    and determine if it is likely spam or a genuine message.
    Consider the following criteria when assessing the message:

    1. Promises of high earnings in a short period or with little effort (e.g., "300-400 dollars per week", "500$+ per week", "pure profit of 400-500$ per day").
    2. Mentions of remote work or collaboration without specific details about the job itself (e.g., "remote employment", "remote work", "remote collaboration").
    3. Calls to write to private messages (PM) for details, instead of openly describing the offer (e.g., "for details in PM", "if interested, write to PM", "for details, write to PM").
    4. Claims that no experience is required and everything will be taught (e.g., "no experience required, but welcomed", "we'll teach you everything ourselves", "training from scratch").
    5. Use of emojis to attract attention and give a positive image to the message.
    6. Vague and unclear wording, lack of specifics in describing the work or project (e.g., "interesting project", "new field", "good passive income", "mutually beneficial cooperation").
    7. Generic greetings followed by vague offers of work or collaboration (e.g., "Hello, I am looking for partners for remote collaboration.").
    8. Unusual text formatting, such as mixing character sets or excessive punctuation, likely to grab attention or evade spam filters.
    9. Recruiting personnel or partners with minimal requirements (e.g., age being the only criteria).
    10. Vague invitations to join a project or group without providing any specific details about the nature of the work or the organization (e.g., "Join us", "We are looking for reliable people", "Become part of our team").
    11. Out of context of the ongoing conversation in the chat.

    If 2 or more of these signs are present in the message, even if they are not strongly expressed, then it should be classified as spam.
    Be cautious not to flag the message as spam simply because it mentions job opportunities or collaborations. 
    Genuine messages may share some characteristics with spam, such as inviting people to discuss details in private. 
    Only classify the message as spam if it exhibits a clear pattern of unsolicited, suspicious, or off-topic content.

    Context of the last 5 messages in the chat:
    {context}

    New message to be evaluated: 
    {message}

    Response format:
    SPAM - if the message is spam. Add explanations of what signs of spam are present.
    SAFE - if the message is not spam. Briefly explain why you didn't consider it spam, and note if any additional context might still be needed to fully determine the message's legitimacy.
    Put the mark SPAM or SAFE at the beginning of your response.
    """
    try:
        logger.info(f'Waiting for the model to evaluate message: {message}')

        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are an AI assistant trained to identify potential spam messages in a Telegram chat."},
                {"role": "user", "content": prompt}
            ]
        )

        result = completion.choices[0].message.content.lower()
        logger.info(f'Model answer: {result}')
        return 'spam' in result[:8]
    except Exception as e:
        logger.error(f"Error in is_spam function: {type(e).__name__}: {e}")
        return False

allowed_chats = [-1001474293774, -1002051811264]

async def handle_message(update: Update, context):
    message = update.message
    if message is None or message.from_user is None:
        logger.info("Skipping update without message or from_user attribute")
        return

    user_id = str(message.from_user.id)  # ensure user_id is string for JSON keys
    chat_id = message.chat_id
    message_text = message.text
    message_id = message.message_id
    username = message.from_user.username

    if chat_id not in allowed_chats:
        logger.info(f'Skipping message from chat ID {chat_id} not in allowed list')
        return

    if message.chat.type in ['group', 'supergroup'] and not message.from_user.is_bot:
        if safe_messages_count[user_id]['count'] < 2:
            context_text = '\n'.join(recent_messages[chat_id])
            logger.info(f'Recent messages for chat {chat_id}: {context_text}')

            if is_spam(message_text, context_text):
                await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
                #await context.bot.send_message(chat_id=chat_id, text=f"🚨{username} запостил спам, а я его удалил")
                spam_logger.info(f"User ID: {user_id}, Username: {username}, Deleted Message: {message_text}")
                logger.info(f'Deleted spam message: {message_text}')
                message_text = 'here was a spam message'
            else:
                update_safe_messages_count(user_id, username)
        else:
            logger.info(f'Skipping message from user {user_id} ({username}) due to safe messages limit')

        recent_messages[chat_id].append(message_text)

    else:
        logger.info(f'Skipping message from user {user_id} ({username}) due to the message from the bot')

def update_safe_messages_count(user_id, username):
    safe_messages_count[user_id]['count'] += 1
    safe_messages_count[user_id]['username'] = username
    save_safe_users(safe_messages_count)

def main():
    application = ApplicationBuilder().token(TELEGRAM_API_TOKEN).build()

    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    application.add_handler(message_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
