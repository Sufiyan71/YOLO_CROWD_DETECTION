# send_email.py
import boto3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from botocore.exceptions import ClientError
import os
import threading 
import logging

# Import the default config to get AWS credentials
from config import config_web as config

email_logger = logging.getLogger('email_sender')

# Global SES client to reuse connections
SES_CLIENT = None

def get_ses_client():
    """Initializes and returns a resilient AWS SES client using central configuration."""
    global SES_CLIENT
    if SES_CLIENT is None:
        try:
            # Use credentials from the central config file
            if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
                email_logger.error("AWS credentials (KEY_ID or SECRET_KEY) are not set in config.py.")
                return None
            
            SES_CLIENT = boto3.client(
                'ses',
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                region_name=config.AWS_REGION
            )

            # Test connection to ensure credentials are valid
            SES_CLIENT.get_send_quota() 
            email_logger.info(f"Initialized AWS SES client in region: {config.AWS_REGION}.")
        except ClientError as e:
            email_logger.error(f"Failed to initialize SES client: {e.response['Error']['Message']}", exc_info=True)
            SES_CLIENT = None
        except Exception as e:
            email_logger.error(f"An unexpected error occurred initializing SES client: {e}", exc_info=True)
            SES_CLIENT = None
            
    return SES_CLIENT

# This function will be the actual target for the thread
def _send_email_sync_target(location, threshold, timestamp, image_path, count, sender_email, recipient_emails):
    """
    Synchronously sends an email with an attached image using AWS SES.
    This function is designed to be called by a background thread.
    """
    client = get_ses_client()
    if not client:
        email_logger.error("SES client not available. Cannot send email.")
        return

    if not recipient_emails:
        email_logger.warning("No recipient emails specified. Skipping email send.")
        return
        
    if not os.path.exists(image_path):
        email_logger.error(f"Screenshot file not found for email attachment: {image_path}. Cannot send email.")
        return

    subject = f"⚠️ Crowd Density Alert: {location}" # Also good to have location in subject

    # --- START OF FIX ---
    # The `body_html` string is now an f-string (notice the 'f' at the beginning).
    # This will correctly substitute the variables like {location}, {count}, etc.
    body_html=f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            padding: 20px;
            color: #333;
            }}
            .container {{
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-left: 5px solid #e53935;
            padding: 20px;
            max-width: 600px;
            margin: auto;
            }}
            .header {{
            font-size: 22px;
            font-weight: bold;
            color: #d32f2f;
            }}
            .content {{
            margin-top: 15px;
            font-size: 16px;
            line-height: 1.6;
            }}
            .details {{
            margin-top: 15px;
            margin-bottom: 25px;
            padding-left: 10px;
            }}
            .detail-item {{
            margin-bottom: 10px;
            font-size: 16px;
            }}
            .location {{
            font-weight: bold;
            color: #1565c0;
            }}
            .footer {{
            margin-top: 30px;
            font-size: 12px;
            color: #999999;
            text-align: center;
            }}
            strong.highlight {{
                color: #e53935;
                font-weight: bold;
            }}
        </style>
        </head>
        <body>
        <div class="container">
            <div class="header">⚠️ Crowd Density Alert</div>
            <div class="content">
            Our monitoring system has detected an unusually high crowd density.
            <div class="details">
                <div class="detail-item">📍 Location: <span class="location">{location}</span></div>
                <div class="detail-item">🚨 Threshold: <strong>{threshold}</strong></div>
                <div class="detail-item">🚨 Count: <strong class="highlight">{count}</strong></div>
                <div class="detail-item">⏰ Time: <strong>{timestamp}</strong></div>
            </div>
            The current crowd level has <strong class="highlight">exceeded</strong> the predefined safety threshold.
            <br><br>
            <strong>Action Recommended:</strong> Please notify security personnel or take appropriate safety measures to ensure crowd control and avoid potential hazards.
            </div>
            <div class="footer">
            This is an automated notification from the MoodScope AI Smart Crowd Monitoring System.<br>
            Please do not reply to this email.
            </div>
        </div>
        </body>
        </html>
    """
    # --- END OF FIX ---

    # Create a multipart/mixed parent container
    msg = MIMEMultipart('related')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipient_emails)

    # Create the body with HTML and plain-text
    msg_alt = MIMEMultipart('alternative')
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(body_html, 'html'))

    # Read and attach image
    try:
        with open(image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read())
            img.add_header('Content-ID', '<image1>') # This ID is used to reference the image in the HTML
            img.add_header('Content-Disposition', 'inline', filename=os.path.basename(image_path))
            msg.attach(img)
    except Exception as e:
        email_logger.error(f"Failed to attach image {image_path} to email: {e}", exc_info=True)
        return False

    # This part remains the same
    try:
        response = client.send_raw_email(
            Source=sender_email,
            Destinations=recipient_emails,
            RawMessage={'Data': msg.as_string()}
        )
        email_logger.info(f"Email sent! Message ID: {response['MessageId']}")
        return True
    except ClientError as e:
        email_logger.error(f"SES Error sending email: {e.response['Error']['Message']}", exc_info=True)
        return False
    except Exception as e:
        email_logger.error(f"Unexpected error sending email: {e}", exc_info=True)
        return False

# This function remains unchanged as it just starts the thread
def send_email_with_image(location, threshold, timestamp, image_path, count, sender_email, recipient_emails):
    """
    Sends an email with an image in a background thread.
    This function matches the requested signature and parameters.
    """
    if not recipient_emails:
        email_logger.info("No email recipients configured. Skipping email send.")
        return

    thread = threading.Thread(
        target=_send_email_sync_target,
        args=(location, threshold, timestamp, image_path, count, sender_email, recipient_emails),
        daemon=True
    )
    thread.start()
    email_logger.info(f"Email sending thread started for location: {location}")