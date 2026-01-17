import base64
import logging
import os
import json
from collections import defaultdict, deque
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters
from telegram import ChatPermissions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_API_TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Model configuration (with defaults)
TEXT_MODEL = os.getenv('TEXT_MODEL', 'deepseek/deepseek-chat-v3.1')
IMAGE_MODEL = os.getenv('IMAGE_MODEL', 'google/gemini-3-flash-preview')

# Image spam configuration
BAN_FOR_IMAGE_SPAM = os.getenv('BAN_FOR_IMAGE_SPAM', 'false').lower() == 'true'
MAX_IMAGE_SIZE_MB = float(os.getenv('MAX_IMAGE_SIZE_MB', '5'))
IMAGE_ANALYSIS_TIMEOUT = float(os.getenv('IMAGE_ANALYSIS_TIMEOUT', '5'))

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)

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

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

def is_spam(message: str, context: str) -> bool:
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
    12. Sales advertisements or promotional messages for products/services (e.g., "selling", "buy now", "special offer", "discount", "promotion").

    If 2 or more of these signs are present in the message, even if they are not strongly expressed, then it should be classified as spam.
    Be cautious not to flag the message as spam simply because it mentions job opportunities or collaborations. 

    Return a JSON response in the following format:
    {{
        "is_spam": true/false,
        "confidence": float between 0 and 1,
        "spam_signs": [
            {{
                "type": "string (one of: high_earnings, remote_work, pm_redirect, no_experience, emoji_abuse, vague_wording, generic_greeting, unusual_formatting, minimal_requirements, vague_invitation, out_of_context, sales_ad)",
                "description": "string explaining why this sign was detected"
            }}
        ],
        "explanation": "string with detailed explanation of the decision"
    }}

    Context of the last 5 messages in the chat:
    {context}

    New message to evaluate: 
    {message}

    Return ONLY the JSON response without any additional text.
    """
    try:
        logger.info(f'Waiting for the model to evaluate message: {message}')

        completion = client.chat.completions.create(
            model=TEXT_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an AI assistant that analyzes messages for spam. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        result = json.loads(completion.choices[0].message.content)
        
        analysis_log = (
            f"\n[TEXT] Message: {message}\n"
            f"Is spam: {result['is_spam']}\n"
            f"Confidence: {result['confidence']}\n"
            f"Spam signs detected: {len(result['spam_signs'])}\n"
        )
        for sign in result['spam_signs']:
            analysis_log += f"- {sign['type']}: {sign['description']}\n"
        analysis_log += f"Explanation: {result['explanation']}\n"
        
        spam_logger.info(analysis_log)
        logger.info(f'Model analysis: {json.dumps(result, indent=2, ensure_ascii=False)}')
        
        is_spam_message = (
            (result['is_spam'] and result['confidence'] >= 0.7) or
            (len(result['spam_signs']) >= 2 and result['confidence'] >= 0.6)
        )
        
        return is_spam_message

    except Exception as e:
        logger.error(f"Error in is_spam function: {type(e).__name__}: {e}")
        return False


def is_image_spam(image_bytes: bytes, user_id: str, username: str) -> dict:
    """Analyze image for spam content using vision model via OpenRouter."""
    prompt = """You are an AI assistant analyzing images for spam in a Telegram chat about automobiles and lifestyle.

Analyze this image and determine if it is spam. Look for these spam indicators:
1. promotional_banner - Promotional banners or advertising images
2. qr_code - QR codes (often used for scam redirects)
3. adult_content - Adult or inappropriate content
4. scam_offer - Scam offers or get-rich-quick schemes
5. mlm_scheme - MLM/pyramid scheme promotions
6. crypto_promotion - Cryptocurrency promotion or investment schemes
7. contact_info_overlay - Phone numbers, social media handles, or contact info overlaid on image

If 2 or more spam signs are present, classify as spam.
Legitimate car photos, lifestyle images, and normal conversation images should NOT be flagged.

Return a JSON response:
{
    "is_spam": true/false,
    "confidence": float between 0 and 1,
    "spam_signs": [
        {"type": "spam_sign_type", "description": "why this was detected"}
    ],
    "explanation": "brief explanation of the decision"
}

Return ONLY the JSON response without any additional text."""

    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{base64_image}"

        logger.info(f'Analyzing image from user {user_id} ({username}) for spam')

        completion = client.chat.completions.create(
            model=IMAGE_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=IMAGE_ANALYSIS_TIMEOUT,
            messages=[
                {"role": "system", "content": "You are an AI assistant that analyzes images for spam. Always respond with valid JSON."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]}
            ]
        )

        result = json.loads(completion.choices[0].message.content)

        analysis_log = (
            f"\n[IMAGE] User ID: {user_id}, Username: {username}\n"
            f"Is spam: {result['is_spam']}\n"
            f"Confidence: {result['confidence']}\n"
            f"Spam signs detected: {len(result['spam_signs'])}\n"
        )
        for sign in result['spam_signs']:
            analysis_log += f"- {sign['type']}: {sign['description']}\n"
        analysis_log += f"Explanation: {result['explanation']}\n"

        spam_logger.info(analysis_log)
        logger.info(f'Image analysis: {json.dumps(result, indent=2, ensure_ascii=False)}')

        is_spam_image = (
            (result['is_spam'] and result['confidence'] >= 0.7) or
            (len(result['spam_signs']) >= 2 and result['confidence'] >= 0.6)
        )

        return {'is_spam': is_spam_image, 'result': result}

    except Exception as e:
        logger.error(f"Error in is_image_spam function: {type(e).__name__}: {e}")
        return {'is_spam': False, 'result': None}


allowed_chats = [-1001474293774, -1002051811264, -4684100667, -5111113304]

async def handle_message(update: Update, context):
    message = update.message
    if message is None or message.from_user is None:
        logger.info("Skipping update without message or from_user attribute")
        return

    user_id = str(message.from_user.id)
    chat_id = message.chat_id
    message_text = message.text or message.caption or ''
    message_id = message.message_id
    username = message.from_user.username
    has_photo = message.photo is not None and len(message.photo) > 0
    has_text = bool(message_text.strip())

    if chat_id not in allowed_chats:
        logger.info(f'Skipping message from chat ID {chat_id} not in allowed list')
        return

    if message.chat.type in ['group', 'supergroup'] and not message.from_user.is_bot:
        if safe_messages_count[user_id]['count'] < 2:
            context_text = '\n'.join(recent_messages[chat_id])
            logger.info(f'Recent messages for chat {chat_id}: {context_text}')

            text_is_spam = False
            image_is_spam = False
            should_ban = False

            # Analyze image if present
            if has_photo:
                photo = message.photo[-1]  # Get largest photo size
                file_size_mb = (photo.file_size or 0) / (1024 * 1024)

                if file_size_mb > MAX_IMAGE_SIZE_MB:
                    logger.info(f'Skipping image analysis: size {file_size_mb:.2f}MB > {MAX_IMAGE_SIZE_MB}MB limit')
                else:
                    try:
                        file = await context.bot.get_file(photo.file_id)
                        image_bytes = bytes(await file.download_as_bytearray())
                        image_result = is_image_spam(image_bytes, user_id, username)
                        image_is_spam = image_result['is_spam']
                        if image_is_spam and BAN_FOR_IMAGE_SPAM:
                            should_ban = True
                    except Exception as e:
                        logger.error(f"Failed to download image: {e}")

            # Analyze text if present
            if has_text:
                text_is_spam = is_spam(message_text, context_text)
                if text_is_spam:
                    should_ban = True

            # Determine content type for logging
            if has_photo and has_text:
                content_type = '[TEXT+IMAGE]'
            elif has_photo:
                content_type = '[IMAGE]'
            else:
                content_type = '[TEXT]'

            # Handle spam detection
            if text_is_spam or image_is_spam:
                bot_member = await context.bot.get_chat_member(chat_id, context.bot.id)

                can_delete = getattr(bot_member, 'can_delete_messages', False)
                can_restrict = getattr(bot_member, 'can_restrict_members', False)

                logger.info(f"Bot permissions in chat {chat_id}:")
                logger.info(f"  - Status: {bot_member.status}")
                logger.info(f"  - Can delete messages: {can_delete}")
                logger.info(f"  - Can restrict members: {can_restrict}")
                logger.info(f"  - Chat type: {message.chat.type}")

                spam_reason = []
                if text_is_spam:
                    spam_reason.append('text')
                if image_is_spam:
                    spam_reason.append('image')

                action_log = f"{content_type} User ID: {user_id}, Username: {username}, Spam detected in: {', '.join(spam_reason)}"

                if can_delete:
                    try:
                        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
                        action_log += ", Action: Message deleted"
                        logger.info(f'Deleted spam message from {username}')
                        message_text = 'here was a spam message'
                    except Exception as e:
                        logger.error(f"Failed to delete message: {e}")
                        action_log += f", Action: Failed to delete message ({e})"
                else:
                    await context.bot.send_message(chat_id=chat_id, text=f"🚨у меня нет прав на удаление сообщения от {username}")
                    action_log += ", Action: No permission to delete message"

                if should_ban and can_restrict:
                    try:
                        logger.info(f"Attempting to ban user {user_id} ({username}) in chat {chat_id}")
                        await context.bot.ban_chat_member(chat_id=chat_id, user_id=int(user_id))
                        action_log += ", User banned"
                        logger.info(f"User {user_id} banned successfully")
                    except Exception as e:
                        logger.error(f"Failed to ban user: {e}")
                        action_log += f", Failed to ban user ({e})"
                elif should_ban and not can_restrict:
                    logger.warning(f"No permission to ban user {user_id} in chat {chat_id}")
                    action_log += ", No permission to ban user"
                elif not should_ban:
                    action_log += ", User not banned (image spam only)"

                spam_logger.info(action_log)
            else:
                update_safe_messages_count(user_id, username)
        else:
            logger.info(f'Skipping message from user {user_id} ({username}) due to safe messages limit')

        if message_text:
            recent_messages[chat_id].append(message_text)

    else:
        logger.info(f'Skipping message from user {user_id} ({username}) due to the message from the bot')

def update_safe_messages_count(user_id: str, username: str) -> None:
    safe_messages_count[user_id]['count'] += 1
    safe_messages_count[user_id]['username'] = username
    save_safe_users(safe_messages_count)

def main() -> None:
    application = ApplicationBuilder().token(TELEGRAM_API_TOKEN).build()

    message_handler = MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, handle_message)
    application.add_handler(message_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
