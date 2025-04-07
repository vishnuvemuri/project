from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from apscheduler.schedulers.background import BackgroundScheduler
import mysql.connector
import random
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database import get_db_connection,create_connection,fetch_museum_data_by_category, fetch_museum_data_by_name_with_prices, fetch_ticket_prices_by_type, fetch_museum_data, is_museum_open,update_booking_with_date_time,update_booking_with_payment_id,fetch_booking_details,insert_booking,fetch_museum_by_category,fetch_data_of_museum,is_museum_open,fetch_price_of_ticket_by_type,update_booking_status,update_booking_with_qr_code,update_booking_with_email,fetch_last_booking_id,update_booking_with_no_total,update_booking_with_feedback,is_future_date
from mysql.connector import Error
from datetime import datetime,timedelta
import sqlite3
import joblib
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import logging
from math import radians, sin, cos, sqrt, asin
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from rapidfuzz import process
import re
from io import BytesIO
from email.mime.image import MIMEImage
import smtplib,ssl
from flask_cors import CORS
from translation import detect_language, translate_text
import numpy as np
import pandas as pd
import os
import razorpay
import qrcode

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Store user-selected languages
user_languages = {}

# Initialize Razorpay client
client = razorpay.Client(auth=("rzp_test_Vc1dMULkCvrbi2", "AbFrPLAmRPAQoo4039F79LVq"))

# Set up logging
logging.basicConfig(level=logging.INFO)

QR_DIR = "qrcodes"
os.makedirs(QR_DIR, exist_ok=True)

def convert_to_int(price):
    try:
        return 0 if price.lower() == "free" or not price.strip() else int(price)
    except ValueError:
        return 0

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "korukoppulamohanapriya@gmail.com"
SMTP_PASS = "oace ajek woxx szwu"

# MySQL database configuration
db_config = {
    'user': 'root',
    'password': 'Heysiri1207',
    'host': 'museum-db.czc02syc8orz.ap-south-1.rds.amazonaws.com',
    'database': 'museum'
}

failed_notifications_list = []
retry_job = None
sent_notifications = set()  # Store sent notifications (email, museum, datetime)

def send_notification_email(recipient_email, museum_name, visit_datetime):
    """
    Send an email notification to the user.
    Returns True if successful, False otherwise.
    """
    try:
        sender_email = SMTP_USER
        sender_password = SMTP_PASS
        subject = "Upcoming Museum Visit Reminder"
        body = f"""
        <html>
        <body>
        <h2>Dear Visitor,</h2>
        <p>This is a reminder for your upcoming visit to {museum_name}.</p>
        <p><strong>Visit Date and Time:</strong> {visit_datetime}</p>
        <p>We look forward to welcoming you!</p>
        </body>
        </html>
        """
        
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        print(f"✅ Notification email sent to {recipient_email} for {museum_name} at {visit_datetime}")
        return True
    except Exception as e:
        print(f"❌ Failed to send notification email to {recipient_email}: {e}")
        return False

def check_upcoming_visits():
    """
    Check for upcoming visits within the next 3 hours and send reminders.
    """
    global retry_job
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        # Query bookings table
        query_bookings = """
        SELECT user_email, museum_name, visit_datetime
        FROM bookings
        WHERE visit_datetime BETWEEN NOW() AND NOW() + INTERVAL 3 HOUR
        """
        cursor.execute(query_bookings)
        upcoming_visits_bookings = cursor.fetchall()

        # Query ticket_booking table
        query_ticket_booking = """
        SELECT user_email, museum_name, CONCAT(visit_date, ' ', visit_time) AS visit_datetime
        FROM ticket_booking
        WHERE CONCAT(visit_date, ' ', visit_time) BETWEEN NOW() AND NOW() + INTERVAL 3 HOUR
        """
        cursor.execute(query_ticket_booking)
        upcoming_visits_ticket_booking = cursor.fetchall()

        # Combine results
        upcoming_visits = upcoming_visits_bookings + upcoming_visits_ticket_booking

        failed_notifications = []
        for visit in upcoming_visits:
            visit_key = (visit['user_email'], visit['museum_name'], visit['visit_datetime'])

            # Check if notification was already sent
            if visit_key in sent_notifications:
                continue  # Skip sending duplicate notifications

            success = send_notification_email(visit['user_email'], visit['museum_name'], visit['visit_datetime'])
            if success:
                sent_notifications.add(visit_key)  # Mark as sent
            else:
                failed_notifications.append(visit)

        cursor.close()
        connection.close()

        # Store failed notifications for retry
        global failed_notifications_list
        failed_notifications_list = failed_notifications

        # Schedule retry job if there are failed notifications
        if failed_notifications_list and not retry_job:
            retry_job = scheduler.add_job(retry_failed_notifications, 'interval', minutes=5, misfire_grace_time=60)

    except mysql.connector.Error as e:
        print(f"❌ Database error: {e}")

def retry_failed_notifications():
    """
    Retry sending failed notifications.
    """
    global retry_job
    global failed_notifications_list

    if failed_notifications_list:
        for visit in failed_notifications_list[:]:  # Iterate over a copy of the list
            success = send_notification_email(visit['user_email'], visit['museum_name'], visit['visit_datetime'])
            if success:
                failed_notifications_list.remove(visit)  # Remove from retry list
                sent_notifications.add((visit['user_email'], visit['museum_name'], visit['visit_datetime']))  # Mark as sent

    # If all notifications are sent, remove the retry job
    if not failed_notifications_list and retry_job:
        retry_job.remove()
        retry_job = None

# Scheduler to run check every hour
scheduler = BackgroundScheduler()
scheduler.add_job(check_upcoming_visits, 'interval', hours=1, misfire_grace_time=60)
scheduler.start()

# Razorpay Test Credentials
razorpay_client = razorpay.Client(auth=("rzp_test_Vc1dMULkCvrbi2", "AbFrPLAmRPAQoo4039F79LVq"))

# Function to load the pre-trained model
def load_model():
    try:
        vectorizer = joblib.load('vectorizer.pkl')
        train_names = joblib.load('train_names.pkl')
        train_vectors = joblib.load('train_vectors.pkl')
        return vectorizer, train_names, train_vectors
    except Exception as e:
        print("Error loading model:", str(e))
        return None, None, None

# Function to find the best match using cosine similarity and difflib
def find_best_match(query, vectorizer, train_vectors, train_names, threshold=0.5):
    query = query.strip().lower()  # Normalize input
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, train_vectors).flatten()

    # Use difflib to refine the best match
    difflib_scores = [difflib.SequenceMatcher(None, query, name).ratio() for name in train_names]

    # Combine scores
    combined_scores = [(sim + diff) / 2 for sim, diff in zip(similarities, difflib_scores)]
    best_combined_match_index = np.argmax(combined_scores)
    best_combined_match_name = train_names[best_combined_match_index]

    print(f"Query: {query} | Best match: {best_combined_match_name} | Score: {combined_scores[best_combined_match_index]}")

    # Check if the best match score is above the threshold
    if combined_scores[best_combined_match_index] < threshold:
        return None
    
    return best_combined_match_name

# In-memory storage for bookings (for demonstration purposes)
user_bookings = {}

def send_ticket_email(recipient_email, booking_id):
    """
    Sends a museum ticket confirmation email with a QR code containing ticket details
    and stores the QR code image path in the database.
    """
    try:
        # Database connection
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)

        # Retrieve booking details
        query = """
            SELECT id, museum_name, category, adult_tickets, children_tickets, photography_tickets, visit_date, visit_time
            FROM ticket_booking
            WHERE id = %s
        """
        cursor.execute(query, (booking_id,))
        booking_details = cursor.fetchone()

        if not booking_details:
            print(f"❌ No booking details found for ID: {booking_id}")
            cursor.close()
            connection.close()
            return False

        # Generate QR Code
        qr_data = f"Ticket ID: {booking_id}\nMuseum: {booking_details['museum_name']}\nVisit Date: {booking_details['visit_date']}"
        qr_path = f"qrcodes/{booking_id}.png"  # Store relative path
        qr = qrcode.make(qr_data)
        qr.save(qr_path)

        # Update database with QR code path
        update_query = "UPDATE ticket_booking SET user_email=%s, qr_code_path=%s WHERE id=%s"
        cursor.execute(update_query, (recipient_email, qr_path, booking_id))
        connection.commit()  # Commit the update

        cursor.close()
        connection.close()

        sender_email = "korukoppulamohanapriya@gmail.com"
        sender_password = "oace ajek woxx szwu"
        subject = "🎟️ Your Museum Ticket Confirmation"

        # Email body
        body = f"""
        <html>
        <body>
            <h2>Dear Visitor,</h2>
            <p>Thank you for booking tickets with us! Here are your details:</p>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><th>Ticket Id</th><td>{booking_details['id']}</td></tr>
                <tr><th>Museum</th><td>{booking_details['museum_name']}</td></tr>
                <tr><th>Category</th><td>{booking_details['category']}</td></tr>
                <tr><th>Adult Tickets</th><td>{booking_details['adult_tickets']}</td></tr>
                <tr><th>Children Tickets</th><td>{booking_details['children_tickets']}</td></tr>
                <tr><th>Photography Passes</th><td>{booking_details['photography_tickets']}</td></tr>
                <tr><th>Visit Date</th><td>{booking_details['visit_date']}</td></tr>
                <tr><th>Visit Time</th><td>{booking_details['visit_time']}</td></tr>
            </table>
            <h3>Your QR Code:</h3>
            <p>Scan the QR code below at the museum entrance:</p>
            <img src="cid:qrcode" alt="QR Code" width="200">
        </body>
        </html>
        """

        # Create email
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        # Attach QR Code Image
        with open(qr_path, "rb") as qr_file:
            qr_img = MIMEImage(qr_file.read(), name="qrcode.png")
            qr_img.add_header("Content-ID", "<qrcode>")
            msg.attach(qr_img)

        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        print(f"✅ Email with QR Code sent to {recipient_email}")
        return True

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def get_refund_amount(visit_datetime, amount_paid):
    """Determine refund amount based on cancellation time."""
    current_datetime = datetime.now()
    time_difference = visit_datetime - current_datetime
    hours_difference = time_difference.total_seconds() / 3600  # Convert seconds to hours

    if hours_difference <= 24:
        return 0, "❌ No refund available as cancellation is within 24 hours."
    else:
        refund_amount = round(amount_paid * 0.3, 2)  # 30% refund
        return refund_amount, f"✅ ₹{refund_amount} (30%) refund will be processed."

def process_razorpay_refund(payment_id, refund_amount, booking_id, user_id):
    """Processes refund using Razorpay and stores details in the database."""
    try:
        refund_response = razorpay_client.payment.refund(payment_id, int(refund_amount * 100))  # Convert to paise
        refund_id = refund_response["id"]  # Razorpay refund ID

        # Store refund details in database
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO refunds (booking_id, user_id, refund_amount, refund_status, razorpay_refund_id)
            VALUES (%s, %s, %s, 'Processed', %s)
        """, (booking_id, user_id, refund_amount, refund_id))
        connection.commit()
        cursor.close()
        connection.close()

        print(f"✅ Razorpay Refund ID: {refund_id} for ₹{refund_amount}")
        flash(f"✅ Refund of ₹{refund_amount} has been processed.", "success")

    except Exception as e:
        print(f"❌ Razorpay Refund Initiated: {e}")
        flash("Refund processing intiated. The amount will be refund in 3 working days", "error")

def get_museum_pricing(museum_name):
    try:
        conn = mysql.connector.connect(
            host="museum-db.czc02syc8orz.ap-south-1.rds.amazonaws.com",
            user="root",
            password="Heysiri1207",
            database="museum"
        )
        if conn.is_connected():
            cursor = conn.cursor(dictionary=True) 

            query = "SELECT pricing_factor, factor_status FROM museum_pricing WHERE museum_name = %s"
            cursor.execute(query, (museum_name,))
            result = cursor.fetchone()
            
            return result if result else None

    except Error as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# User session storage (to track multiple user interactions)
user_sessions = {}

def convert_to_24_hour(time_str):
    """Convert 12-hour time (e.g., 10:00 AM) to 24-hour format (e.g., 10:00)."""
    try:
        return datetime.strptime(time_str.strip(), "%I:%M %p").strftime("%H:%M")
    except ValueError:
        return None  # Invalid format handling

# Load the trained model
crowd_model = joblib.load('crowd_prediction.pkl')

# Load the dataset for museum details
df = pd.read_csv(r"C:\Users\vishn\Downloads\Musuembot1(a)\dataset\dataset for crowd prediction.csv", encoding="ISO-8859-1")

# Preprocess dataset
df['Category'] = df['Category'].astype('category').cat.codes
df['Location'] = df['Location'].astype('category').cat.codes
df['Required Time'] = df['Required Time'].str.extract('(\d+)').astype(float)

def is_holiday(date_input, df):
    """Check if the given date corresponds to a holiday in the dataset."""
    try:
        selected_date = datetime.strptime(date_input, "%Y-%m-%d").date()
        selected_day = selected_date.strftime("%A")  # Convert date to weekday name
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD."

    # Standardize column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Safely get the holidays column
    holiday_entries = df.get("holidays", pd.Series()).dropna()

    if holiday_entries.empty:
        return False  # No holidays found

    holiday_dates = set()
    holiday_days = set()

    # Parse holiday data
    for entry in holiday_entries:
        items = [x.strip() for x in entry.split(",")]
        for item in items:
            try:
                holiday_dates.add(datetime.strptime(item, "%Y-%m-%d").date())
            except ValueError:
                holiday_days.add(item)  # Assume it's a weekday name

    # Check if selected date is a holiday
    return selected_date in holiday_dates or selected_day in holiday_days

def predict_crowd(museum_name, date_input, time_str, df):
    crowd_model = joblib.load('crowd_prediction.pkl')

    try:
        selected_datetime = datetime.strptime(f"{date_input} {time_str}", "%Y-%m-%d %H:%M")
        selected_date = selected_datetime.date()
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD for date and HH:MM for time."

    today = datetime.today().date()
    if selected_date < today:
        return f"Cannot predict for past dates ({date_input}). Please enter a future date."

    # Standardize column names before accessing
    df.columns = df.columns.str.strip().str.lower()

    # Check if the given date is a holiday
    if is_holiday(date_input, df):
        return f"The museum is closed on {date_input} due to a holiday."

    # Ensure museum exists
    if 'name' not in df.columns:
        return "Dataset is missing the 'name' column."

    museum = df[df['name'].str.lower().str.strip() == museum_name.lower().strip()]
    if museum.empty:
        return "Museum not found."

    try:
        # Extract and clean opening hours
        opening_hours_str = museum.iloc[0].get('opening_hours', "").strip().lower()
        time_range = opening_hours_str.split(" to ")

        if len(time_range) != 2:
            return "Error processing museum opening hours: Invalid format."

        opening_time_24 = convert_to_24_hour(time_range[0])
        closing_time_24 = convert_to_24_hour(time_range[1])

        if not opening_time_24 or not closing_time_24:
            return "Error processing museum opening hours: Invalid time format."

        opening_time = datetime.strptime(opening_time_24, "%H:%M").time()
        closing_time = datetime.strptime(closing_time_24, "%H:%M").time()
        selected_time = selected_datetime.time()

        if not (opening_time <= selected_time <= closing_time):
            return f"The museum is closed at {time_str}. It is open from {opening_time_24} to {closing_time_24}."

    except Exception as e:
        return f"Error processing museum opening hours: {str(e)}"

    try:
        category = int(museum.iloc[0].get('category', -1))
        location = int(museum.iloc[0].get('location', -1))
        required_time = float(museum.iloc[0].get('required time', 0))

        if category == -1 or location == -1:
            return "Error: Missing category or location data."

    except ValueError:
        return "Error processing museum data."

    # Prepare input for prediction
    input_features = np.array([[selected_datetime.weekday(), selected_datetime.hour, category, location, required_time]])
    input_features = input_features.reshape(1, -1)

    predicted_crowd_numeric = int(round(crowd_model.predict(input_features)[0]))
    predicted_crowd_numeric = np.clip(predicted_crowd_numeric, 0, 2)

    crowd_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
    return f"Predicted crowd level at {museum_name} on {date_input} at {time_str} is {crowd_mapping[predicted_crowd_numeric]}."

# Load the pre-trained Nearest Neighbors model
with open('recommend.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the location recommendation model
vectorizer, train_names, train_vectors = joblib.load('location_vectorizer.pkl'), joblib.load('location_train_names.pkl'), joblib.load('location_train_vectors.pkl')

# Initialize geolocator
geolocator = Nominatim(user_agent="museum_recommendation")

# Function to calculate the Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))

    return R * c

# Function to generate a random OTP
def generate_otp():
    return random.randint(100000, 999999)

def create_payment_link(user_session, amount, currency="INR", description="Museum Ticket Booking"):
    try:
        print(f"Creating payment link for amount: {amount}, currency: {currency}, description: {description}")

        payment_link = razorpay_client.payment_link.create({
            "amount": amount * 100,  # Convert INR to paise
            "currency": currency,
            "description": description,
            "notes": {
                "booking_type": "museum_tickets"
            }
        })

        print(f"Payment link created successfully: {payment_link}")

        user_session["payment_link"] = payment_link["short_url"]
        user_session["payment_link_id"] = payment_link["id"]  # Store payment link ID in session

        # Store payment link ID in the database
        update_booking_with_payment_id(user_session["booking_id"], payment_link["id"])

        user_session["state"] = "WAITING_PAYMENT"
        return f"💳 Please complete your payment using this link: {payment_link['short_url']}.\nAfter payment, type 'I have paid' to proceed."
    except Exception as e:
        print(f"❌ Error creating payment link: {e}")
        return "❌ Failed to generate payment link. Please try again."

def check_payment_status(payment_link_id, booking_id):
    try:
        # Fetch payment link details using the payment link ID
        payment_link_details = razorpay_client.payment_link.fetch(payment_link_id)
        print(f"DEBUG: Payment Link Details = {payment_link_details}")

        # Check if the payment link is marked as "paid"
        if payment_link_details["status"] == "paid":
            return "Paid"
        else:
            return "Pending"
    except Exception as e:
        print(f"❌ Error fetching payment status: {e}")
        return "Failed"

def handle_payment_callback(user_session):
    booking_id = user_session.get("booking_id")
    payment_link_id = user_session.get("payment_link_id")  # Use Payment Link ID

    if not booking_id or not payment_link_id:
        return "❌ Missing booking details. Please try again."

    print(f"DEBUG: Checking payment for booking_id={booking_id}, payment_link_id={payment_link_id}")

    # Check payment status using the Payment Link ID and Booking ID
    payment_status = check_payment_status(payment_link_id, booking_id)

    if payment_status == "Paid":
        update_booking_status(booking_id, "Paid")

        user_session["state"] = "ASK_EMAIL"
        return f"✅ Payment successful for Booking ID: {booking_id}! Please enter your email to receive your tickets."
    
    return f"❌ Payment not completed for Booking ID: {booking_id}. Please try again."

import mysql.connector
from mysql.connector import Error, cursor

def load_pricing_model():
    return joblib.load("dynamic_pricing_model.pkl")

def get_museum_pricing(museum_name):
    try:
        conn = mysql.connector.connect(
            host="museum-db.czc02syc8orz.ap-south-1.rds.amazonaws.com",
            user="root",
            password="Heysiri1207",
            database="museum"
        )
        if conn.is_connected():
            cursor = conn.cursor(dictionary=True)
            query = "SELECT pricing_factor, factor_status FROM museum_pricing WHERE museum_name = %s"
            cursor.execute(query, (museum_name,))
            result = cursor.fetchone()
            print(f"🔍 Debug: Retrieved Pricing Factor for {museum_name} -> {result}")  # Debugging
            return result if result else None
    except Error as e:
        print(f"❌ Error fetching pricing factor: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def calculate_total_amount(museum_name, visitor_type, adult_tickets, children_tickets, photography_tickets, student_passes):
    print(f"🔍 Debug: Calculating total amount for {museum_name}")  # Debugging
    ticket_prices = fetch_price_of_ticket_by_type(museum_name, visitor_type)
    if not ticket_prices:
        print("❌ Error: Ticket prices not found!")
        return None
    museum_pricing = get_museum_pricing(museum_name)
    if not museum_pricing:
        pricing_factor = 1.0  # Default to 1.0 if pricing factor is missing
        print("⚠️ Warning: No pricing factor found, using 1.0")
    else:
        try:
            factor_status = int(museum_pricing["factor_status"])  # Convert to int for proper evaluation
            pricing_factor = float(museum_pricing["pricing_factor"]) if factor_status == 1 else 1.0
            print(f"✅ Pricing Factor Applied: {pricing_factor}x")  # Debugging
        except Exception as e:
            print(f"❌ Error processing pricing factor: {e}")
            pricing_factor = 1.0  # Fail-safe fallback
    def parse_price(price):
        if isinstance(price, str) and price.lower() == "free":
            return 0
        try:
            return float(price)  # Ensure numeric conversion
        except (ValueError, TypeError):
            return 0
    try:
        adult_price = parse_price(ticket_prices["adult_price"]) * pricing_factor
        children_price = parse_price(ticket_prices["children_price"]) * pricing_factor
        photography_price = parse_price(ticket_prices["photography_fee"]) * pricing_factor
        student_price = parse_price(ticket_prices.get("student_fee", 0)) * pricing_factor
        total_amount = (
            (adult_tickets * adult_price) +
            (children_tickets * children_price) +
            (photography_tickets * photography_price) +
            (student_passes * student_price)
        )
        print(f"✅ Final Total Amount: ₹{round(total_amount, 2)}")
        return round(total_amount, 2)
    except Exception as e:
        print(f"❌ Error calculating total amount: {e}")
        return None  # Prevent chatbot crash

def send_email_ticket(recipient_email, booking_id):
    try:
        print(f"DEBUG: Attempting to send email to {recipient_email} for booking ID: {booking_id}")

        # Fetch booking details
        booking_details = fetch_booking_details(booking_id)
        if not booking_details:
            print(f"❌ No booking details found for ID: {booking_id}")
            return False

        # Generate QR Code
        qr_data = f"Ticket ID: {booking_id}\nMuseum: {booking_details['museum_name']}\nVisit Date: {booking_details['visit_datetime']}\nAdult Tickets: {booking_details['adult_tickets']}\nChildren Tickets: {booking_details['children_tickets']}\nPhotography Tickets: {booking_details['photography_tickets']}\nStudent Passes: {booking_details['student_passes']}"
        qr_path = f"qrcodes/{booking_id}.png"
        qr = qrcode.make(qr_data)
        qr.save(qr_path)
        print(f"✅ QR Code generated at: {qr_path}")

        # Update the database with QR code path
        update_booking_with_qr_code(booking_id, qr_path)

        # Email Setup
        sender_email = "korukoppulamohanapriya@gmail.com"
        sender_password = "oace ajek woxx szwu"  # Replace with Google App Password
        subject = "🎟️ Your Museum Ticket Confirmation"

        body = f"""
        <html>
        <body>
            <h2>Dear Visitor,</h2>
            <p>Thank you for booking tickets with us! Here are your details:</p>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><th>Ticket ID</th><td>{booking_details['id']}</td></tr>
                <tr><th>Museum</th><td>{booking_details['museum_name']}</td></tr>
                <tr><th>Visit Date</th><td>{booking_details['visit_datetime']}</td></tr>
                <tr><th>Adult Tickets</th><td>{booking_details['adult_tickets']}</td></tr>
                <tr><th>Children Tickets</th><td>{booking_details['children_tickets']}</td></tr>
                <tr><th>Photography Tickets</th><td>{booking_details['photography_tickets']}</td></tr>
                <tr><th>Student Passes</th><td>{booking_details['student_passes']}</td></tr>
            </table>
            <h3>Your QR Code:</h3>
            <p>Scan the QR code below at the museum entrance:</p>
            <img src="cid:qrcode" alt="QR Code" width="200">
        </body>
        </html>
        """

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        # Attach QR Code Image
        with open(qr_path, "rb") as qr_file:
            qr_img = MIMEImage(qr_file.read(), name="qrcode.png")
            qr_img.add_header("Content-ID", "<qrcode>")
            msg.attach(qr_img)

        # Send Email
        print("DEBUG: Connecting to SMTP server...")
        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            print("DEBUG: SMTP login successful!")
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"✅ Email successfully sent to {recipient_email}")

        return True

    except smtplib.SMTPAuthenticationError as auth_err:
        print(f"❌ SMTP Authentication Error: {auth_err}")
    except smtplib.SMTPException as smtp_err:
        print(f"❌ SMTP Exception: {smtp_err}")
    except FileNotFoundError as file_err:
        print(f"❌ QR Code File Error: {file_err}")
    except Exception as e:
        print(f"❌ General Error in sending email: {e}")

    return False


# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Load the trained KNN model for location-based recommendations
with open("recommend.pkl", "rb") as f:
    knn_model = pickle.load(f)

# Load the trained components for spelling correction
vectorize = joblib.load("vectorizer.pkl")
trained_names = joblib.load("train_names.pkl")
trained_vectors = joblib.load("train_vectors.pkl")

# Initialize geolocator
geolocator = Nominatim(user_agent="museum_locator")

# User sessions
user_sessions = {}

greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]

MUSEUM_CATEGORIES = [
    "Arts", 
    "Historical Museum", 
    "Science and Technology",
    "Museum-house",
    "Archeology Museum", 
    "General"
]

def get_lat_lon_from_location(location_name):
    location = geolocator.geocode(location_name)
    if location:
        return float(location.latitude), float(location.longitude)
    return None, None

# Recommend museums near a given location
def recommend_nearby_museums(location_name, num_results=5):
    user_lat, user_lon = get_lat_lon_from_location(location_name)
    if user_lat is None or user_lon is None:
        return "Sorry, I couldn't find that location. Please enter a valid place."

    df = fetch_data_of_museum()
    if df.empty:
        return "No museum data available."

    distances, indices = knn_model.kneighbors([[user_lat, user_lon]], n_neighbors=min(num_results, len(df)))

    recommendations = []
    for idx in indices[0]:
        museum = df.iloc[idx]
        distance_km = geodesic((user_lat, user_lon), (museum['latitude'], museum['longitude'])).km
        recommendations.append({
            "id": museum["id"],
            "name": museum["name"],
            "location": museum["location"],
            "distance_km": round(distance_km, 2)
        })

    return recommendations

def recommend_nearby_museums_by_coords(latitude, longitude, num_results=5):
    user_lat, user_lon = float(latitude), float(longitude)
    df = fetch_data_of_museum()
    if df.empty:
        return "No museum data available."
    distances, indices = knn_model.kneighbors([[user_lat, user_lon]], n_neighbors=min(num_results, len(df)))
    recommendations = []
    for idx in indices[0]:
        museum = df.iloc[idx]
        distance_km = geodesic((user_lat, user_lon), (museum['latitude'], museum['longitude'])).km
        recommendations.append({
            "id": museum["id"],
            "name": museum["name"],
            "location": museum["location"],
            "distance_km": round(distance_km, 2)
        })
    return recommendations

def find_correct_match(query, vectorize, trained_vectors, trained_names, threshold=0.5):
    query_vector = vectorize.transform([query])
    similarities = cosine_similarity(query_vector, trained_vectors)
    best_match_index = similarities.argmax()
    best_match_name = trained_names[best_match_index]
    
    # Use difflib to refine the best match
    difflib_scores = [difflib.SequenceMatcher(None, query, name).ratio() for name in trained_names]
    best_difflib_match_index = difflib_scores.index(max(difflib_scores))
    best_difflib_match_name = trained_names[best_difflib_match_index]
    
    # Combine scores
    combined_scores = [(similarity + difflib_score) / 2 for similarity, difflib_score in zip(similarities[0], difflib_scores)]
    best_combined_match_index = combined_scores.index(max(combined_scores))
    best_combined_match_name = trained_names[best_combined_match_index]
    
    # Check if the best match score is above the threshold
    if combined_scores[best_combined_match_index] < threshold:
        return None
    
    return best_combined_match_name

def get_museum_details(museum_name):
    corrected_name = find_correct_match(museum_name, vectorize, trained_vectors, trained_names)

    print(f"🔍 User entered: {museum_name}")
    print(f"✅ Suggested correction: {corrected_name}")

    if corrected_name is None:
        return "❌ Sorry, I couldn't find details for that museum. Please check the name."

    connection = get_db_connection()
    with connection.cursor() as cursor:
        query = """
            SELECT id, name, address, location, opening_hours, holidays
            FROM museumdetails
            WHERE name = %s
        """
        cursor.execute(query, (corrected_name,))
        result = cursor.fetchone()

    connection.close()

    if result:
        return (
            f"🏛️ **Museum Details:**\n"
            f"🔹 **ID:** {result['id']}\n"
            f"🔹 **Name:** {result['name']} (Did you mean **{corrected_name}**?)\n"
            f"📍 **Address:** {result['address']}\n"
            f"🌍 **Location:** {result['location']}\n"
            f"⏰ **Opening Hours:** {result['opening_hours']}\n"
            f"📅 **Holidays:** {result['holidays']}\n\n"
            f"Would you like to continue with booking?"
            f"If yes, enter Indian or Foreigner else enter no or exit."
        )
    else:
        print("❌ Museum not found in database.")
        return "❌ Sorry, I couldn't find details for that museum."
    
def normalize_input(user_input):
    # Remove the word "Museums" (case-insensitive)
    user_input = user_input.lower().replace("museums", "").strip()
    
    # Map common variations to the correct category names
    variations = {
        "history": "historical",
        "archeological": "archeology",
        "science and tech": "science and technology",
    }
    
    # Replace variations with the correct category names
    for variation, correct_term in variations.items():
        if variation in user_input:
            user_input = correct_term
            break
    
    return user_input

def classify_intent(user_input):
    user_input = user_input.lower()

    intent_phrases = {}
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            intent_phrases[pattern.lower()] = intent["tag"]

    # Find the best match for user input
    best_match, score, _ = process.extractOne(user_input, list(intent_phrases.keys()))

    print(f"🔍 User entered: {user_input}")  # Debugging
    print(f"✅ Best match found: {best_match} (Score: {score})")  # Debugging

    # If the similarity score is above 80, consider it a match
    if score > 80:
        return intent_phrases[best_match]
    
    return "unknown"

# Function to send OTP email
def send_otp_email(sender_email, sender_password, recipient_email, otp):
    try:
        subject = "Your OTP Verification Code"
        body = f"Hello,\n\nYour OTP code is: {otp}\n\nPlease do not share it with anyone.\n\nRegards,\nYour App Team"

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        print("OTP email sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def chatbot_response(user_id, text):
    global user_sessions

    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "state": None, "museum": None, "museum_id": None, "category": None, "tickets": {},
            "visit_datetime": None, "opening_hours": None
        }

    user_session = user_sessions[user_id]
    intent = classify_intent(text)  # Uses fuzzy matching

    if "details" in text.lower() or "information" in text.lower() or "about" in text.lower():
        museum_name = text.lower().replace("details", "").replace("information", "").replace("about", "").strip()
        corrected_name = find_correct_match(museum_name, vectorize, trained_vectors, trained_names)
        if corrected_name:
            museum_details = get_museum_details(corrected_name)
            
            # Store the museum name in the session for booking continuation
            user_session["museum"] = corrected_name
            user_session["state"] = "ASK_BOOKING_CONFIRMATION"  # Set state to ask if the user wants to book
            return museum_details
        else:
            return "❌ I couldn't find that museum. Please enter a valid name."

    if intent == "greeting":
        return "Hello! 😊 How can I assist you?"
    
    if intent=="well_being":
        return "I am Fine. How are you?"
    
    if "good" in text.lower() or "fine" in text.lower():
        return "😊 How can I assist you?"

    if intent == "book_ticket":
        text = text.lower().strip()
        parts = text.split("for")  # Check if user provided a museum name
    
        # ✅ If user provided a museum name (e.g., "Book Ticket for National Museum")
        if len(parts) > 1:
            museum_query = parts[1].strip()
            corrected_museum = find_correct_match(museum_query, vectorize, trained_vectors, trained_names)
    
            if corrected_museum:
                user_session["museum"] = corrected_museum
                museum_details = get_museum_details(corrected_museum)
                user_session["state"] = "ASK_TYPE"  # Proceed to ticket type selection
                return museum_details
            else:
                return "❌ Sorry, I couldn't find that museum. Please try again or type 'No' to exit."
    
        # ✅ If user did **not** specify a museum name
        else:
            user_session["state"] = "ASK_RECOMMENDATION"
            return "Do you want me to recommend a museum or do you already have one in mind?"

    if intent == "ticket_price":
        # Check if the user has already provided a museum name in their query
        museum_name = None
        
        # Try to extract the museum name from the sentence (handles "for" and "of")
        match = re.search(r"ticket prices? (?:for|of) (.+)", text.lower())
        if match:
            museum_name = match.group(1).strip()  # Extract museum name from user input
        
        # If no museum name was found, ask the user to provide one
        if not museum_name:
            user_session["state"] = "ASK_TICKET_MUSEUM"
            return "Ticket prices vary by museum and visitor type. Which museum are you asking about?"
        
        # If a museum name was found, correct it and proceed
        corrected_name = find_correct_match(museum_name, vectorize, trained_vectors, trained_names)
        
        if corrected_name:
            user_session["museum"] = corrected_name  # Store corrected museum name
            print(f"✅ Corrected museum name for ticket inquiry: {corrected_name}")  # Debugging
            
            # Proceed to ask user type (Indian/Foreigner)
            user_session["state"] = "ASK_TYPE"
            return f"Are you an **Indian** or a **Foreigner** for ticket pricing of {corrected_name}?"
        
        return "❌ I couldn't find that museum. Please enter a valid museum name."

    if intent == "museum_category":
        user_session["state"] = "ASK_CATEGORY"  # Set session state to track category selection
        return (
            "We have museums under these categories:\n" +
            "\n".join([f"🔹 {category}" for category in MUSEUM_CATEGORIES]) +
            "\n\nPlease enter the category you are interested in."
        )
    
    if intent == "opening_hours":
        user_session["state"] = "ASK_MUSEUM"
        return "Please provide the museum name to check its opening hours."

    if intent == "holidays":
        user_session["state"] = "ASK_MUSEUM"
        return "Please enter the museum name to check its holiday schedule."

    if intent == "help":
        return "I can help you with museum details, ticket booking, recommendations, and ticket pricing. How can I assist you today?"

    if intent == "goodbye":
        return random.choice(["Goodbye! Have a great day!", "See you again!"])

    if intent=="recommend":
        user_session["state"] = "ASK_RECOMMEND_TYPE"
        return "Do you want museums **near your location** or from a **specific category**?"
    
    '''# Step 1: If user provides a future date & time first, ask for museum name
    if re.match(r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}', text):
        try:
            selected_datetime = datetime.strptime(text, "%d-%m-%Y %H:%M")
            if selected_datetime < datetime.now():
                return "Past times cannot be predicted. Please enter a future date and time."
            user_session["date_time"] = text.split()
            return "Please enter the museum name for prediction."
        except ValueError:
            return "Invalid date format. Please enter in DD-MM-YYYY HH:MM format."
    
    # Step 2: If user requests crowd prediction but hasn't provided museum name
    if intent == "predict crowd" or "predict crowd" in text.lower():
        return "Please enter the museum name for prediction."
    
    # Step 3: If user provides a museum name, reset and ask for date/time
    if text.strip().lower() != user_session.get("museum_name", "").lower():
        user_session["museum_name"] = text.strip()
        user_session.pop("date_time", None)  # Reset date and time for new museum
        return "Please enter the date (DD-MM-YYYY) and time (HH:MM) for prediction."
    
    # Step 4: If both museum name & future date exist, trigger prediction
    if "museum_name" in user_session and "date_time" in user_session:
        return predict_crowd(user_session["museum_name"], user_session["date_time"][0], user_session["date_time"][1], user_id)'''
    
    if "near_me" in text.lower():
        try:
            # Extract latitude and longitude from the text
            latitude, longitude = map(float, text.split()[1].split(","))
            recommendations = recommend_nearby_museums_by_coords(latitude, longitude)
            
            if isinstance(recommendations, str):  # If an error message is returned
                return recommendations  # Show error (e.g., "Sorry, I couldn't find that location.")
            
            if not recommendations:  # If no museums are found
                return "❌ No museums found near this location. Please enter a different location."
            
            # Store the list of recommended museums in the session
            user_session["recommended_museums"] = [m["name"] for m in recommendations]
            
            # If valid recommendations exist, ask for selection
            response = "Here are some museums near you:\n"
            response += "\n".join([f"🏛️ {m['name']} - {m['distance_km']} km away" for m in recommendations])
            response += "\nWhich one would you like to visit?"
            
            user_session["state"] = "ASK_MUSEUM"  # Transition to ASK_MUSEUM state
            return response
        except ValueError:
            return "Invalid location format. Please provide your location as 'latitude,longitude'."
    
    if user_session.get("state") == "WAITING_PAYMENT" or text.lower() == "i have paid":
        if text.lower() == "i have paid":
            return handle_payment_callback(user_session)
        else:
            return "💳 Please complete your payment first. Once done, type 'I have paid' to proceed."
        
    if user_session["state"] == "ASK_FEEDBACK":
        feedback = text.strip()
        booking_id = user_session.get("booking_id")
        if booking_id and feedback:
            update_booking_with_feedback(booking_id, feedback)
            user_session["state"] = None  # Reset the state
            return "✅ Thank you for your feedback! We appreciate your input."
        else:
            return "❌ Failed to save feedback. Please try again."

    # Ask for recommendations
    elif user_session["state"] == "ASK_RECOMMENDATION":
        if "recommend" in text.lower():
            user_session["state"] = "ASK_RECOMMEND_TYPE"
            return "Do you want museums **near your location** or from a **specific category**?"
        
        # Instead of going back to ASK_RECOMMENDATION, check if museum exists and proceed
        corrected_name = find_correct_match(text, vectorize, trained_vectors, trained_names)
    
        if corrected_name:
            user_session["museum"] = corrected_name  # Store corrected name
            print(f"✅ Corrected museum name: {corrected_name}")  # Debugging
            museum_details = get_museum_details(corrected_name)
            user_session["state"] = "ASK_TYPE"
            return museum_details  # Proceed directly to ticket type selection
    
        return "❌ I couldn't find that museum. Do you want me to recommend one or try entering again?"

    elif user_session["state"] == "ASK_TICKET_MUSEUM":
        corrected_name = find_correct_match(text, vectorize, trained_vectors, trained_names)
    
        if corrected_name:
            user_session["museum"] = corrected_name  # Store corrected name in session
            print(f"✅ Corrected museum name for ticket inquiry: {corrected_name}")  # Debugging
    
            # ✅ Proceed to ask user type (Indian/Foreigner)
            user_session["state"] = "ASK_TYPE"
            return f"Are you an **Indian** or a **Foreigner** for ticket pricing of {corrected_name}?"
        
        return "❌ I couldn't find that museum. Please enter a valid museum name."

    elif user_session["state"] == "ASK_RECOMMEND_TYPE":
        if "location" in text.lower():
            user_session["state"] = "ASK_LOCATION_NAME"
            return "Please enter your location (e.g., 'New Delhi')."
        elif "category" in text.lower():
            user_session["state"] = "ASK_CATEGORY"
            return (
                "Here are the available museum categories:\n" +
                "\n".join([f"🔹 {category}" for category in MUSEUM_CATEGORIES]) +
                "\n\nPlease enter the category you are interested in."
            )
        return "Please enter either 'location' or 'category'."

    elif user_session["state"] == "ASK_LOCATION_NAME":
        recommendations = recommend_nearby_museums(text)
        
        if isinstance(recommendations, str):  # If an error message is returned
            return recommendations  # Show error (e.g., "Sorry, I couldn't find that location.")
        
        if not recommendations:  # If no museums are found
            return "❌ No museums found near this location. Please enter a different location."
        
        # Store the list of recommended museums in the session
        user_session["recommended_museums"] = [m["name"] for m in recommendations]
        
        # If valid recommendations exist, ask for selection
        response = "Here are some museums near you:\n"
        response += "\n".join([f"🏛️ {m['name']} - {m['distance_km']} km away" for m in recommendations])
        response += "\nWhich one would you like to visit?"
        
        user_session["state"] = "ASK_MUSEUM"  # Transition to ASK_MUSEUM state
        return response

    elif user_session["state"] == "ASK_CATEGORY":
        selected_category = text.strip()  # Remove leading/trailing whitespace
        print(f"DEBUG: User entered category: {selected_category}")  # Debugging
    
        # Normalize the user's input
        normalized_input = normalize_input(selected_category)
        print(f"DEBUG: Normalized input: {normalized_input}")  # Debugging
    
        # Normalize categories for case-insensitive matching
        normalized_categories = {cat.lower(): cat for cat in MUSEUM_CATEGORIES}
        print(f"DEBUG: Normalized categories: {normalized_categories}")  # Debugging
    
        # Find the best match for the selected category
        matched_category = None
        for category_key, category_value in normalized_categories.items():
            if normalized_input in category_key:
                matched_category = category_value
                break
    
        print(f"DEBUG: Matched category: {matched_category}")  # Debugging
    
        if matched_category:
            user_session["category"] = matched_category  # Store category in session
    
            # Fetch museums for the selected category
            museums = fetch_museum_by_category(matched_category)
            print(f"DEBUG: Fetched museums: {museums}")  # Debugging
    
            if museums:
                # Store the list of recommended museums in the session
                user_session["recommended_museums"] = [m["name"] for m in museums]
    
                # Format the response with museum details
                response = f"Here are the museums under **{matched_category}**:\n\n"
                for museum in museums:
                    response += f"🏛️ {museum['name']} 📍 {museum['location']}\n"
                response += "\nEnter the museum name to get more details or type 'back' to choose another category."
    
                user_session["state"] = "ASK_MUSEUM"  # Move to museum selection
            else:
                response = f"❌ Sorry, no museums found under **{matched_category}**. Try another category."
    
        else:
            # Provide a list of valid categories if the input is invalid
            response = (
                "❌ Invalid category. Please select from:\n\n"
                + "\n".join([f"🔹 {cat}" for cat in MUSEUM_CATEGORIES])
            )
    
        print(f"DEBUG: Response sent: {response}")  # Debugging
        return response

    elif user_session["state"] == "ASK_MUSEUM":
        if text.lower() in ["no", "not now", "cancel", "exit"]:
            user_session["state"] = None  # Reset session state
            return "Alright! If you need any help, feel free to ask. 😊"
        
        # Check if the user responded with "yes"
        if text.lower() == "yes":
            # Check if the selected museum is in the recommended list
            if "recommended_museums" in user_session and user_session.get("selected_museum"):
                corrected_name = find_correct_match(user_session["selected_museum"], vectorize, trained_vectors, trained_names)
                
                if corrected_name:
                    user_session["museum"] = corrected_name  # Store the corrected name
                    print(f"✅ Corrected museum name: {corrected_name}")  # Debugging
                    
                    # Fetch and display museum details
                    museum_details = get_museum_details(corrected_name)
                    
                    # Proceed to ask for the visitor category (Indian/Foreigner)
                    user_session["state"] = "ASK_TYPE"
                    return f"{museum_details}\n\nAre you an **Indian** or a **Foreigner** for ticket pricing?"
            
            return "❌ I couldn't find that museum. Please enter a valid name or type 'No' to exit."
        
        # If the user didn't respond with "yes", store the selected museum and ask for confirmation
        else:
            # Check if the selected museum is in the recommended list
            if "recommended_museums" in user_session and text in user_session["recommended_museums"]:
                user_session["selected_museum"] = text  # Store the selected museum name
                return f"Do you want to proceed with **{text}**? (Yes/No)"
            
            return "❌ I couldn't find that museum. Please enter a valid name or type 'No' to exit."
        
    elif user_session["state"] == "ASK_TYPE":
        if text.lower() in ["no", "not now", "cancel", "exit"]:
            user_session["state"] = None  # Reset session state
            return "No worries! If you need anything else, just let me know. 😊"
        
        if text.lower() in ["indian", "foreigner"]:
            user_session["type"] = text.lower()
            
            # Fetch ticket prices using the corrected museum name
            ticket_prices = fetch_price_of_ticket_by_type(user_session["museum"], text.lower())
            
            if ticket_prices:
                response = (
                    f"Here are the ticket prices for {text} visitors at **{user_session['museum']}**:\n"
                    f"🎟️ **Adult:** ₹{ticket_prices['adult_price']}\n"
                    f"👦 **Children:** ₹{ticket_prices['children_price']}\n"
                    f"🎓 **Student:** ₹{ticket_prices['student_fee']}\n"
                    f"📷 **Photography:** ₹{ticket_prices['photography_fee']}\n\n"
                    "How many Adult tickets do you need?"
                )
                user_session["state"] = "ASK_ADULT_TICKETS"  # Proceed to ask for adult tickets
            else:
                response = "❌ Sorry, ticket prices are unavailable for this category. Please try another option."
                user_session["state"] = "ASK_TYPE"  # Stay in the same state
            
            return response
        
        return "Please enter either 'Indian' or 'Foreigner' or type 'No' to exit."

    elif user_session["state"] == "ASK_ADULT_TICKETS":
        if text.isdigit():
            user_session["tickets"]["adult"] = int(text)
            if int(text) > 0:
                user_session["adult_names"] = []  # Initialize list to store names
                user_session["state"] = "ASK_ADULT_NAMES"
                return f"Please enter the names of {text} adult(s), separated by commas."
            else:
                user_session["state"] = "ASK_CHILDREN_TICKETS"
                return "How many Children tickets do you need?"
        elif text.lower() == "no" or int(text)==0:
            # Ask for confirmation if the user means 0 tickets
            user_session["state"] = "ASK_ADULT_TICKETS_CONFIRM_ZERO"
            return "Do you mean **0** adult tickets? (Yes/No)"
        else:
            return "Please enter a valid number or type 'No' to cancel."
    
    elif user_session["state"] == "ASK_ADULT_TICKETS_CONFIRM_ZERO":
        if text.lower() == "yes":
            user_session["tickets"]["adult"] = 0  # Set adult tickets to 0
            user_session["state"] = "ASK_CHILDREN_TICKETS"
            return "How many Children tickets do you need?"
        elif text.lower() == "no":
            user_session["state"] = None  # Reset session state
            return "No worries! If you need anything else, just let me know. 😊"
        else:
            return "Please enter 'Yes' or 'No'."
    
    elif user_session["state"] == "ASK_ADULT_NAMES":
        if text.lower() == "no":
            # If the user says "no," confirm if they mean 0 tickets
            user_session["state"] = "ASK_ADULT_TICKETS_CONFIRM_ZERO"
            return "Do you mean **0** adult tickets? (Yes/No)"
        
        # Split the input by commas and strip whitespace
        names = [name.strip() for name in text.split(",")]
        
        # Validate each name to ensure it contains only alphabets and spaces
        for name in names:
            if not re.match(r"^[A-Za-z\s]+$", name):
                return f"❌ Invalid name: '{name}'. Names should contain only alphabets and spaces. Please re-enter the names."
        
        # Check if the number of names matches the number of adult tickets
        if len(names) == user_session["tickets"]["adult"]:
            user_session["adult_names"] = names  # Store valid names
            user_session["state"] = "ASK_CHILDREN_TICKETS"
            return "How many Children tickets do you need?"
        else:
            return f"❌ You entered {len(names)} names but booked {user_session['tickets']['adult']} adult tickets. Please re-enter correctly."
    
    elif user_session["state"] == "ASK_CHILDREN_TICKETS":
        if text.isdigit():
            user_session["tickets"]["children"] = int(text)
            if int(text) > 0:
                user_session["children_names"] = []  # Initialize list to store names
                user_session["state"] = "ASK_CHILDREN_NAMES"
                return f"Please enter the names of {text} child(ren), separated by commas."
            else:
                user_session["state"] = "ASK_PHOTOGRAPHY_TICKETS"
                return "How many Photography tickets do you need?"
        elif text.lower() == "no" or int(text)==0:
            # Ask for confirmation if the user means 0 tickets
            user_session["state"] = "ASK_CHILDREN_TICKETS_CONFIRM_ZERO"
            return "Do you mean **0** children tickets? (Yes/No)"
        else:
            return "Please enter a valid number or type 'No' to cancel."
    
    elif user_session["state"] == "ASK_CHILDREN_TICKETS_CONFIRM_ZERO":
        if text.lower() == "yes":
            user_session["tickets"]["children"] = 0  # Set children tickets to 0
            user_session["state"] = "ASK_PHOTOGRAPHY_TICKETS"
            return "How many Photography tickets do you need?"
        elif text.lower() == "no":
            user_session["state"] = None  # Reset session state
            return "No worries! If you need anything else, just let me know. 😊"
        else:
            return "Please enter 'Yes' or 'No'."
    
    elif user_session["state"] == "ASK_CHILDREN_NAMES":
        if text.lower() == "no":
            # If the user says "no," confirm if they mean 0 tickets
            user_session["state"] = "ASK_CHILDREN_TICKETS_CONFIRM_ZERO"
            return "Do you mean **0** children tickets? (Yes/No)"
        
        # Split the input by commas and strip whitespace
        names = [name.strip() for name in text.split(",")]
        
        # Validate each name to ensure it contains only alphabets and spaces
        for name in names:
            if not re.match(r"^[A-Za-z\s]+$", name):
                return f"❌ Invalid name: '{name}'. Names should contain only alphabets and spaces. Please re-enter the names."
        
        # Check if the number of names matches the number of children tickets
        if len(names) == user_session["tickets"]["children"]:
            user_session["children_names"] = names  # Store valid names
            user_session["state"] = "ASK_PHOTOGRAPHY_TICKETS"
            return "How many Photography tickets do you need?"
        else:
            return f"❌ You entered {len(names)} names but booked {user_session['tickets']['children']} children tickets. Please re-enter correctly."
    
    # Add a new state for asking about student passes
    elif user_session["state"] == "ASK_PHOTOGRAPHY_TICKETS":
        if text.isdigit():
            user_session["tickets"]["photography"] = int(text)
            user_session["state"] = "ASK_STUDENT_PASSES"  # Transition to the new state
            return "Do you need any student passes? (Yes/No)"
        elif text.lower() == "no" or int(text)==0:
            # Ask for confirmation if the user means 0 tickets
            user_session["state"] = "ASK_PHOTOGRAPHY_TICKETS_CONFIRM_ZERO"
            return "Do you mean **0** photography tickets? (Yes/No)"
        else:
            return "Please enter a valid number or type 'No' to cancel."

    elif user_session["state"] == "ASK_PHOTOGRAPHY_TICKETS_CONFIRM_ZERO":
        if text.lower() == "yes":
            user_session["tickets"]["photography"] = 0  # Set photography tickets to 0
            user_session["state"] = "ASK_DATE_TIME"
            return "Enter the date and time of visit (YYYY-MM-DD HH:MM)"
        elif text.lower() == "no":
            user_session["state"] = None  # Reset session state
            return "No worries! If you need anything else, just let me know. 😊"
        else:
            return "Please enter 'Yes' or 'No'."

    elif user_session["state"] == "ASK_STUDENT_PASSES":
        if text.lower() == "yes":
            user_session["state"] = "ASK_STUDENT_PASSES_COUNT"
            return "How many student passes do you need?"
        
        elif text.lower() == "no" or int(text)==0:
            # Set student passes to 0
            user_session["tickets"]["student_passes"] = 0
            
            # Calculate total tickets
            total_tickets = (
                user_session["tickets"].get("adult", 0) +
                user_session["tickets"].get("children", 0) +
                user_session["tickets"].get("photography", 0) +
                user_session["tickets"].get("student_passes", 0)
            )
            
            # Check if total tickets is <= 0
            if total_tickets <= 0:
                user_session["state"] = None  # Reset session
                return "❌ Total tickets must be greater than 0. Booking canceled. How can I assist you?"
            
            # Proceed to ask for date and time
            user_session["state"] = "ASK_DATE_TIME"
            return "Enter the date and time of visit (YYYY-MM-DD HH:MM)"
        
        else:
            return "Please enter 'Yes' or 'No'."
    
    elif user_session["state"] == "ASK_STUDENT_PASSES_COUNT":
        if text.isdigit():
            user_session["tickets"]["student_passes"] = int(text)
            
            # Calculate total tickets
            total_tickets = (
                user_session["tickets"].get("adult", 0) +
                user_session["tickets"].get("children", 0) +
                user_session["tickets"].get("photography", 0) +
                user_session["tickets"].get("student_passes", 0)
            )
            
            # Check if total tickets is <= 0
            if total_tickets <= 0:
                user_session["state"] = None  # Reset session
                return "❌ Total tickets must be greater than 0. Booking canceled. How can I assist you?"
            
            # Proceed to ask for date and time
            user_session["state"] = "ASK_DATE_TIME"
            return "Enter the date and time of visit (YYYY-MM-DD HH:MM) in 24hours format"
        
        elif text.lower() == "no" or int(text)==0:
            user_session["tickets"]["student_passes"] = 0
            
            # Calculate total tickets
            total_tickets = (
                user_session["tickets"].get("adult", 0) +
                user_session["tickets"].get("children", 0) +
                user_session["tickets"].get("photography", 0) +
                user_session["tickets"].get("student_passes", 0)
            )
            
            # Check if total tickets is <= 0
            if total_tickets <= 0:
                user_session["state"] = None  # Reset session
                return "❌ Total tickets must be greater than 0. Booking canceled. How can I assist you?"
            
            # Proceed to ask for date and time
            user_session["state"] = "ASK_DATE_TIME"
            return "Enter the date and time of visit (YYYY-MM-DD HH:MM)"
        
        else:
            return "Please enter a valid number or type 'No' to cancel."

    elif user_session["state"] == "ASK_DATE_TIME":
        museum_name = user_session["museum"]
        visit_datetime = text.strip()
    
        try:
            visit_date, visit_time = visit_datetime.split(" ")  
        except ValueError:
            return "❌ Invalid format. Please enter the date and time in **YYYY-MM-DD HH:MM** format."
        
        if not is_future_date(visit_date, visit_time):
            print(f"DEBUG: User entered past date/time: {visit_datetime}")
            user_session["state"] = "ASK_DATE_TIME"
            return "❌ You cannot book tickets for past dates. Please enter a valid future date and time.📅 Enter the date and time in **YYYY-MM-DD HH:MM** format. Try again:"
            print(f"DEBUG: User entered past date/time: {visit_datetime}")
            user_session["state"] = "ASK_DATE_TIME"
            return "❌ Past dates are not allowed! Please provide a future date and time.📅 Format: **YYYY-MM-DD HH:MM**. Try again:"
    
        # Debugging: Print the visit date and time
        print(f"DEBUG: Visit Date = {visit_date}, Visit Time = {visit_time}")
    
        is_open, message = is_museum_open(museum_name, visit_date, visit_time)
        print(f"DEBUG: is_open = {is_open}, message = {message}")
    
        if is_open:
            user_session["visit_datetime"] = visit_datetime
            user_session["state"] = None
    
            # Calculate total amount
            total_amount = calculate_total_amount(
                museum_name=museum_name,
                visitor_type=user_session["type"],  # Ensure type is passed
                adult_tickets=user_session["tickets"]["adult"],
                children_tickets=user_session["tickets"]["children"],
                photography_tickets=user_session["tickets"]["photography"],
                student_passes=user_session["tickets"].get("student_passes", 0)
            )
            print(f"DEBUG: total_amount = {total_amount}")
    
            if total_amount is None:
                return "❌ Failed to calculate the total amount. Please try again."
    
            booking_success = insert_booking(
                user_id=user_id,
                museum_name=museum_name,
                visit_datetime=visit_datetime,
                type=user_session["type"],  # Store user category
                adult_tickets=user_session["tickets"]["adult"],
                children_tickets=user_session["tickets"]["children"],
                photography_tickets=user_session["tickets"]["photography"],
                student_passes=user_session["tickets"].get("student_passes", 0),
                adult_names=user_session.get("adult_names", []),
                children_names=user_session.get("children_names", []),
                amount_paid=total_amount
            )
            print(f"DEBUG: Booking success = {booking_success}")
    
            if booking_success:
                user_session["booking_id"] = fetch_last_booking_id(user_id)  # Store booking ID
                
                booking_confirmation = (
                    f"✅ **Booking confirmed for {user_session['museum']} on {visit_datetime}!**\n"
                    f"🎟️ **Adult:** {user_session['tickets']['adult']}\n"
                    f"👦 **Children:** {user_session['tickets']['children']}\n"
                    f"📷 **Photography:** {user_session['tickets']['photography']}"
                )
                if "student_passes" in user_session["tickets"]:
                    booking_confirmation += f"\n🎓 **Student Passes:** {user_session['tickets']['student_passes']}"
    
                # Handle free bookings
                if total_amount == 0:
                    booking_confirmation += "\n\n💰 **Total Amount:** ₹0 (Free Booking)"
                    booking_confirmation += "\nNo payment is required. Your booking is confirmed!"
                    booking_id = user_session.get("booking_id")
                    update_booking_with_no_total(booking_id, status="Paid")
                    user_session["state"] = "ASK_EMAIL"
                    return f"{booking_confirmation}\n\nPlease enter your email address to receive the tickets."
                
                else:
                    # Generate payment link for non-free bookings
                    payment_link = create_payment_link(user_session, total_amount)
                    print(f"DEBUG: payment_link = {payment_link}")
    
                    if not payment_link:
                        return "❌ Failed to generate payment link. Please try again."
    
                    booking_confirmation += f"\n\n💳 **Total Amount:** ₹{total_amount}\n"
                    booking_confirmation += f"🔗 **Payment Link:** {payment_link}\n\n"
                    booking_confirmation += "Please complete the payment to finalize your booking."
    
                return booking_confirmation
            else:
                return "❌ Failed to save booking details. Please try again."

    elif user_session["state"] == "ASK_EMAIL":
        email = text.strip()
    
        # Validate the email address
        if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
            return "❌ Invalid email address. Please enter a valid email."
    
        # Store the email in the session
        user_session["email"] = email
    
        booking_id = user_session.get("booking_id")
        update_booking_with_email(booking_id, email)
        
        # Send the ticket via email
        email_success = send_email_ticket(email, booking_id)
        if email_success:
            user_session["state"] = "ASK_FEEDBACK"
            return "🎟️ Your tickets have been sent to your email. Thank you for booking with us! Please share your feedback about the chatbot and any improvements you would like to see."
        else:
            user_session["state"]="ASK_EMAIL"
            return "❌ Failed to send the email. Please try again."
    
    return f"❌ {message} Please enter another date and time."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")
@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/admin_send_otp", methods=["POST"])
def admin_send_otp():
    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"message": "Email is required."}), 400

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        # Check if email exists in the database
        cursor.execute("SELECT email FROM admin WHERE email = %s", (email,))
        result = cursor.fetchone()

        if not result:
            return jsonify({"message": "Email not found. Only admins can log in."}), 401

        # Generate OTP
        otp = generate_otp()

        # Update the OTP in the database
        cursor.execute("UPDATE admin SET otp = %s WHERE email = %s", (otp, email))
        conn.commit()

        # Send OTP via email
        sender_email = "korukoppulamohanapriya@gmail.com"  # Replace with your Gmail
        sender_password = "oace ajek woxx szwu"  # Replace with your App Password
        send_otp_email(sender_email, sender_password, email, otp)

        return jsonify({'message': 'OTP sent successfully.'})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred while processing your request."}), 500

    finally:
        cursor.close()
        conn.close()

@app.route("/verify_admin_otp", methods=["POST"])
def verify_admin_otp():
    try:
        # Get JSON data from the request
        data = request.get_json()
        email = data.get("email")
        user_otp = data.get("otp")

        if not email or not user_otp:
            return jsonify({'message': 'Email and OTP are required.'}), 400

        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Fetch the stored OTP for the given email
        cursor.execute("SELECT otp FROM admin WHERE email = %s", (email,))
        result = cursor.fetchone()

        # Verify the OTP
        if result and str(result[0]) == user_otp:
            session['user_email'] = email  # Store user email in session
            return jsonify({'message': 'OTP verified successfully', 'redirect': url_for('admin_dashboard')})
        else:
            return jsonify({'message': 'Invalid OTP'}), 400

    except Exception as e:
        print(f"Error in verify_otp: {e}")
        return jsonify({'message': 'Server error while verifying OTP'}), 500

    finally:
        # Close the database connection
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/admin_dashboard')
def admin_dashboard():
    # Render the admin dashboard page
    return render_template("admin_dashboard.html")

@app.route('/add_event', methods=['POST'])
def add_event():
    try:
        data = request.get_json()
        event_name = data.get('name')
        event_date = data.get('date')
        event_time = data.get('time')
        museum_name = data.get('museum')
        event_description = data.get('description')

        # Validate required fields
        if not event_name or not event_date or not event_time or not museum_name or not event_description:
            return jsonify({'message': 'All fields are required.'}), 400

        # Validate date and time
        event_datetime = datetime.strptime(f"{event_date} {event_time}", "%Y-%m-%d %H:%M")
        if event_datetime < datetime.now():
            return jsonify({'message': 'Event date and time must be in the future.'}), 400

        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        print("Database connection successful")

        # Check for duplicate event
        cursor.execute("""
            SELECT COUNT(*) FROM events
            WHERE name = %s AND date = %s AND time = %s AND museum = %s
        """, (event_name, event_date, event_time, museum_name))
        result = cursor.fetchone()
        if result[0] > 0:
            return jsonify({'message': 'An event with the same name, date, time, and museum already exists.'}), 400

        # Insert the event into the database
        cursor.execute("""
            INSERT INTO events (name, date, time, museum, description)
            VALUES (%s, %s, %s, %s, %s)
        """, (event_name, event_date, event_time, museum_name, event_description))
        conn.commit()
        print("Event inserted successfully")

        cursor.close()
        conn.close()

        return jsonify({'message': 'Event added successfully!', 'redirect': url_for('admin_dashboard')})

    except Exception as e:
        print(f"Error in add_event: {e}")
        return jsonify({'message': 'Server error while adding event.'}), 500

@app.route('/add_event')
def add_event_page():
    return render_template("add_event.html")

@app.route('/view_tickets')
def view_tickets():
    # Render the add event page
    return render_template("view_tickets.html")

@app.route('/view_tickets_result', methods=['GET', 'POST'])
def view_tickets_result():
    try:
        data = request.get_json()
        museum_name = data.get('museum_name')
        date = data.get('date')

        if not museum_name or not date:
            return jsonify({'message': 'Museum name and date are required.'}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Fetch the ticket count for the specified museum and date
        cursor.execute("""
            SELECT (adult_tickets+children_tickets) FROM ticket_booking
            WHERE museum_name = %s AND visit_date = %s
        """, (museum_name, date))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        ticket_count = result[0] if result else 0
        return jsonify({'ticket_count': ticket_count})

    except Exception as e:
        print(f"Error in view_tickets: {e}")
        return jsonify({'message': 'Server error while fetching ticket count.'}), 500

@app.route('/dynamic_pricing')
def dynamic_pricing():
    return render_template("dynamic_pricing.html")

@app.route("/start_booking")
def start_booking():
    if "user_email" in session:  # ✅ Check if user is logged in
        return redirect(url_for("success"))  # ✅ Redirect to success page
    else:
        return redirect(url_for("login"))  # ✅ Redirect to login page if not logged in

# Route to check if an email is registered
@app.route("/check_email", methods=["POST"])
def check_email():
    try:
        data = request.get_json()
        email = data["email"]

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        return jsonify({'registered': bool(result)})
    except Exception as e:
        print(f"Error in check_email: {e}")
        return jsonify({'error': 'Server error while checking email'}), 500

@app.route("/send_otp", methods=["POST"])
def send_otp():
    try:
        data = request.get_json()
        email = data.get("email")
        name = data.get("name", "").strip()  # Get name, but default to ""

        if not email:
            return jsonify({'message': 'Email is required'}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Check if email exists in database
        cursor.execute("SELECT name FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()

        if result:  # If email exists (Login case)
            name = result[0]  # Use existing name
        elif not name:  # If Signup and name is missing
            return jsonify({'message': 'Name is required for Signup'}), 400

        otp = generate_otp()

        # Insert/update OTP
        if result:
            cursor.execute("UPDATE users SET otp=%s WHERE email=%s", (otp, email))
            message = "OTP sent for login"
        else:
            cursor.execute("INSERT INTO users (name, email, otp) VALUES (%s, %s, %s)", (name, email, otp))
            message = "OTP sent for signup"

        conn.commit()
        cursor.close()
        conn.close()

        sender_email = "korukoppulamohanapriya@gmail.com"  # Replace with your Gmail
        sender_password = "oace ajek woxx szwu"  # Replace with your App Password
        send_otp_email(sender_email, sender_password, email, otp)

        return jsonify({'message': message})

    except Exception as e:
        print(f"Error in send_otp: {e}")
        return jsonify({'message': 'Server error while sending OTP'}), 500

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    try:
        data = request.get_json()
        email = data["email"]
        user_otp = data["otp"]

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT otp FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and str(result[0]) == user_otp:
            session['user_email'] = email  # Store user email in session
            return jsonify({'message': 'OTP verified successfully', 'redirect': url_for('index')})
        else:
            return jsonify({'message': 'Invalid OTP'}), 400
    except Exception as e:
        print(f"Error in verify_otp: {e}")
        return jsonify({'message': 'Server error while verifying OTP'}), 500

@app.route("/my_bookings")
def my_bookings():
    if "user_email" not in session:
        return redirect(url_for("login"))  # ✅ Redirect to login if user is not logged in

    user_email = session["user_email"]  # Get logged-in user's email

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    try:
        # ✅ Fetch user's name using their email
        cursor.execute("SELECT name FROM users WHERE email = %s", (user_email,))
        user = cursor.fetchone()

        if not user:
            return "User not found", 404

        user_name = user["name"]  # Get the user's name

        # ✅ Fetch bookings using user_name from ticket_booking table
        cursor.execute("SELECT museum_name, visit_date, visit_time FROM ticket_booking WHERE user_name = %s", (user_name,))
        bookings = cursor.fetchall()

        return render_template("my_bookings.html", bookings=bookings)

    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return "Database error occurred", 500

    finally:
        cursor.close()
        conn.close()

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('index'))

@app.route("/success")
def success():
    return render_template("language.html")

@app.route('/language_selection', methods=['POST'])
def language_selection():
    """Handle language selection and redirect to the main menu."""
    return redirect(url_for('main_menu'))

@app.route('/main_menu', methods=['GET', 'POST'])
def main_menu():
    """Render the main menu with options for ticket booking and booking management."""
    return render_template('main_menu.html')

@app.route("/crowd_prediction", methods=["GET", "POST"])
def crowd_prediction():
    prediction = None
    if request.method == "POST":
        museum_name = request.form["museum_name"]
        date = request.form["date"]
        time = request.form["time"]
        
        # Pass df as an argument
        prediction = predict_crowd(museum_name, date, time, df)

    return render_template("crowd_prediction.html", prediction=prediction)

@app.route('/options', methods=['GET', 'POST'])
def options():
    """Display options after language selection."""
    return render_template('options.html')

@app.route('/recommend')
def recommend_museums():
    """Ask the user for the type of recommendation they want."""
    return render_template('recommend_options.html')

@app.route('/recommend/category')
def recommend_by_category():
    """Display museum categories for recommendation."""
    categories = [
        "Arts", 
        "Historical Museums", 
        "Science and Technology",
        "Museum-house",
        "Archeology Museum", 
        "General"
    ]
    return render_template('recommend.html', categories=categories)

@app.route('/recommend/<category>')
def display_museums_by_category(category):
    """Fetch and display museums by the selected category."""
    museums = fetch_museum_data_by_category(category)
    if museums:
        return render_template('category_museums.html', category=category, museums=museums)
    else:
        return render_template('error.html', message="No museums found in this category.")

@app.route('/recommend/location')
def recommend_by_location():
    """Render the location recommendation form."""
    return render_template('recommend_location.html')

@app.route('/recommend_location_user_type', methods=['POST'])
def recommend_location_user_type():
    """Handle user type selection and redirect to location recommendation options."""
    user_type = request.form.get('user_type')
    if not user_type:
        return render_template('error.html', message="Please select a user type.")
    session['user_type'] = user_type  # Store in session
    return render_template('recommend_location_options.html', user_type=user_type)

@app.route('/recommend_near_me_form')
def recommend_near_me_form():
    """Render the form to recommend museums near the user."""
    return render_template('recommend_near_me_form.html')

@app.route('/recommend_specific_location_form')
def recommend_specific_location_form():
    """Render the form to recommend museums at a specific location."""
    return render_template('recommend_specific_location.html')

@app.route('/recommend_near_me', methods=['POST'])
def recommend_near_me():
    try:
        user_lat = float(request.json['latitude'])
        user_lon = float(request.json['longitude'])

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, Name, coordinates FROM museumdetails")
            data = cursor.fetchall()
        conn.close()

        df = pd.DataFrame(data)
        df[['latitude', 'longitude']] = df['coordinates'].str.strip('()').str.split(',', expand=True).astype(float)

        # Calculate distances using Haversine formula
        df['distance_km'] = df.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']), axis=1)

        # Sort by distance and get the top 10 recommendations
        recommendations = df.sort_values(by='distance_km').head(10)

        response = recommendations[['id', 'Name', 'latitude', 'longitude', 'distance_km']].to_dict(orient='records')
        return jsonify({'status': 'success', 'data': response})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/recommend_near_location', methods=['POST'])
def recommend_near_location():
    try:
        location_name = request.json['location']
        location = geolocator.geocode(location_name)

        if location is None:
            return jsonify({'status': 'error', 'message': 'Location not found'})

        user_lat = location.latitude
        user_lon = location.longitude

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, Name, coordinates FROM museumdetails")
            data = cursor.fetchall()
        conn.close()

        df = pd.DataFrame(data)
        df[['latitude', 'longitude']] = df['coordinates'].str.strip('()').str.split(',', expand=True).astype(float)

        # Calculate distances using Haversine formula
        df['distance_km'] = df.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']), axis=1)

        # Sort by distance and get the top 10 recommendations
        recommendations = df.sort_values(by='distance_km').head(10)

        response = recommendations[['id', 'Name', 'latitude', 'longitude', 'distance_km']].to_dict(orient='records')
        return jsonify({'status': 'success', 'data': response})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/museum/<museum_name>')
def display_museum_details(museum_name):
    museum_name_form = request.form.get('museum_name')
    museum_name_url = museum_name
    museum_name_combined = museum_name_form or museum_name_url
    user_type = request.args.get('user_type') or session.get('user_type') or request.form.get('user_type')

    if not museum_name_combined:
        return render_template('error.html', message="Museum name is missing.")

    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            query = "INSERT INTO selected_museum (museum_name, user_type) VALUES (%s, %s)"
            cursor.execute(query, (museum_name, user_type))
            connection.commit()
        except Error as e:
            print(f"Database Error: {e}")
        finally:
            cursor.close()
            connection.close()
    
    museum_details = fetch_museum_data_by_name_with_prices(museum_name_combined, user_type)

    if museum_details:
        print(museum_details)  # Debugging line
        return render_template('museum_details.html', museum_details=museum_details)
    else:
        return render_template('error.html', message="Museum details not found.")

# Route for search functionality
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.json.get('query')  # Get search query
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Load the model
        vectorizer, train_names, train_vectors = load_model()
        if vectorizer is None or train_names is None or train_vectors is None:
            return jsonify({'error': 'Model failed to load'}), 500

        # Find the best match
        best_match = find_best_match(query, vectorizer, train_vectors, train_names)
        
        if best_match is None:
            return jsonify({'error': 'No museum found with the provided name.'}), 404
        
        return jsonify({'best_match': best_match})
    
    return render_template('search.html')

# Route for displaying search results
@app.route('/search_results', methods=['POST'])
def search_results():
    museum_name = request.form.get('museum_name')
    user_type = request.form.get('user_type')

    if not museum_name or not user_type:
        return render_template('error.html', message="Please provide both museum name and user type.")

    # Load model and refine museum name
    vectorizer, train_names, train_vectors = load_model()
    refined_museum_name = find_best_match(museum_name, vectorizer, train_vectors, train_names)

    if refined_museum_name is None:
        return render_template('error.html', message="No museum found with the provided name.")

    # Fetch museum details based on the refined name
    museum_details = fetch_museum_data_by_name_with_prices(refined_museum_name, user_type)
    
    if not museum_details:
        return render_template('error.html', message="Museum or ticket price not found.")
    
    # Store the user type and refined museum name in the selected_museum table
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            query = "INSERT INTO selected_museum (museum_name, user_type) VALUES (%s, %s)"
            cursor.execute(query, (refined_museum_name, user_type))
            connection.commit()
        except Error as e:
            print(f"Database Error: {e}")
            return render_template('error.html', message="An error occurred while saving the museum selection.")
        finally:
            cursor.close()
            connection.close()

    return render_template('museum_details.html', museum_details=museum_details)

# Route to display the ticket booking form
@app.route('/book_ticket/<museum_name>', methods=['GET'])
def display_booking_form(museum_name):
    """Render the ticket booking form."""
    return render_template('book_ticket.html', museum_name=museum_name)

@app.route('/book_ticket', methods=['GET', 'POST'])
def book_ticket():
    """Handle ticket booking for a selected museum."""
    if request.method == 'POST':
        user_name = request.form.get('user_name')
        museum_name = request.form.get('museum_name')
        adult_tickets = int(request.form.get('adult_tickets', 0))
        children_tickets = int(request.form.get('children_tickets', 0))
        photography_tickets = int(request.form.get('photography_tickets', 0))

        if not user_name or (adult_tickets < 0 or children_tickets < 0 or photography_tickets < 0 or (adult_tickets + children_tickets + photography_tickets) <= 0):
            flash("Please provide all required information.", "error")
            return redirect(url_for('display_booking_form', museum_name=museum_name))

        # Store the user_name in session
        session['user_name'] = user_name

        # Database connection
        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("SELECT museum_name, user_type FROM selected_museum ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            if not result or len(result) != 2:
                flash('No museum selected. Please try again.', 'error')
                return redirect(url_for('display_booking_form', museum_name=museum_name))

            museum_name, user_type = result

            # Load the model
            vectorizer, train_names, train_vectors = load_model()
            if vectorizer is None or train_names is None or train_vectors is None:
                flash("Model or vectorizer not found.", "error")
                return redirect(url_for('display_booking_form', museum_name=museum_name))

            # Find the refined museum name
            refined_museum_name = find_best_match(museum_name, vectorizer, train_vectors, train_names)
            if not refined_museum_name:
                flash("Museum name not found in database.", "error")
                return redirect(url_for('display_booking_form', museum_name=museum_name))

            # Insert booking details into the database
            visit_date = 'None'  # Placeholder for date
            visit_time = 'None'  # Placeholder for time

            query = """
                INSERT INTO ticket_booking (museum_name, category, adult_tickets, children_tickets, photography_tickets, visit_date, visit_time, user_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (refined_museum_name, user_type, adult_tickets, children_tickets, photography_tickets, visit_date, visit_time, user_name))
            connection.commit()

            return redirect(url_for('enter_date'))
        else:
            flash('Database connection failed.', 'error')
            return redirect(url_for('display_booking_form', museum_name=museum_name))
    else:
        return render_template('book_ticket.html', error='Invalid request method.')
    
@app.route('/enter_date', methods=['GET', 'POST'])
def enter_date():
    """Render the date and time entry form and handle the submission."""
    # Fetch user_name from session
    user_name = session.get('user_name')
    if not user_name:
        return render_template('error.html', message="User not logged in.")
    
    # Fetch the latest booking_id based on the user_name
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT id, museum_name FROM ticket_booking
            WHERE user_name = %s
            ORDER BY id DESC
            LIMIT 1
        """
        cursor.execute(query, (user_name,))
        result = cursor.fetchone()
        if result:
            booking_id = result['id']
            museum_name = result['museum_name']
        else:
            cursor.close()  # Close the cursor after fetching result
            connection.close()  # Close the connection after fetching result
            return render_template('error.html', message="No bookings found for this user.")
        
        cursor.close()  # Close the cursor after fetching result
        connection.close()  # Close the connection after fetching result
    else:
        return render_template('error.html', message="Database connection failed.")
    
    # Fetch museum details from the database for the specific museum
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT name, opening_hours, holidays, required_time 
            FROM museumdetails 
            WHERE name = %s
        """
        cursor.execute(query, (museum_name,))
        museum_data = cursor.fetchone()
        cursor.close()  # Close the cursor after fetching result
        connection.close()  # Close the connection after fetching result

        if not museum_data:
            return render_template('error.html', message="Museum data not available.")
    else:
        return render_template('error.html', message="Database connection failed.")
    
    if request.method == 'POST':
        booking_date = request.form.get('booking_date')
        booking_time = request.form.get('booking_time')
        print(f"Form submitted with date: {booking_date}, time: {booking_time}")  # Debugging statement
        if not (booking_date and booking_time):
            flash("Please provide both booking date and time.", "error")
            return redirect(url_for('enter_date'))
        selected_date = datetime.strptime(booking_date, "%Y-%m-%d").date()
        today = datetime.today().date()
        if selected_date < today:
            flash("Entered date is in the past. Please select a future date.", "error")
            return redirect(url_for('enter_date'))
        
        # Check if the museum is open on the selected date and time
        is_open, message = is_museum_open(museum_name, booking_date, booking_time)
        if not is_open:
            flash(message, "error")
            return redirect(url_for('enter_date'))
        
        # Update the booking with the selected date and time
        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            query = """
                UPDATE ticket_booking
                SET visit_date = %s, visit_time = %s
                WHERE id = %s
            """
            cursor.execute(query, (booking_date, booking_time, booking_id))
            connection.commit()  # Commit the transaction
            print(f"Booking updated with date: {booking_date}, time: {booking_time}")  # Debugging statement
    
            cursor.close()  # Close the cursor after query execution
            connection.close()  # Close the connection after query execution
        return redirect(url_for('payment', booking_id=booking_id))
        
    return render_template('enter_date.html', museum_data=museum_data)

@app.route('/save_pricing', methods=['POST'])
def save_pricing():
    try:
        data = request.json
        print("Received Data:", data)  # Debugging step
        
        museum_name = data.get('museum_name')
        pricing_factor = data.get('pricing_factor')
        factor_status = data.get('factor_status')

        if not museum_name or pricing_factor is None or factor_status is None:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        connection = create_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO museum_pricing (museum_name, pricing_factor, factor_status)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            pricing_factor = VALUES(pricing_factor),
            factor_status = VALUES(factor_status)
        """
        cursor.execute(query, (museum_name, pricing_factor, factor_status))
        connection.commit()

        cursor.close()
        connection.close()

        return jsonify({"success": True})
    
    except Exception as e:
        print("Error:", str(e))  # Debugging step
        return jsonify({"success": False, "error": str(e)}), 500

def get_connection():
    return mysql.connector.connect(
        host="museum-db.czc02syc8orz.ap-south-1.rds.amazonaws.com",
        user="root",
        password="Heysiri1207",
        database="museum",
    )


@app.route('/payment/<int:booking_id>', methods=['GET', 'POST'])
def payment(booking_id):
    user_name = session.get('user_name')
    if not user_name:
        return render_template('error.html', message="User not logged in.")
    
    connection = get_connection()
    if connection is None:  # ✅ Fix: Ensure connection is established
        return render_template('error.html', message="Database connection failed.")

    cursor = connection.cursor(dictionary=True)

    try:
        # ✅ Fetch booking details
        query = """
            SELECT id, museum_name, category, adult_tickets, children_tickets, photography_tickets, visit_date, visit_time
            FROM ticket_booking
            WHERE id = %s
        """
        cursor.execute(query, (booking_id,))
        booking_data = cursor.fetchone()

        if not booking_data:
            return render_template('error.html', message="Booking details not found.")

        museum_name = booking_data["museum_name"]

        # ✅ Fix: Ensure the connection is still open before executing query
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)  # Ensure cursor is still valid
        else:
            connection = get_connection()
            cursor = connection.cursor(dictionary=True)

        # ✅ Fetch ticket prices
        query = """
            SELECT adult_price, children_price, photography_fee
            FROM ticketprices
            WHERE museum_id = (SELECT id FROM museumdetails WHERE name = %s) AND type = %s
        """
        cursor.execute(query, (museum_name, booking_data["category"]))  # ✅ Fix: Ensure connection is active
        price_data = cursor.fetchone()

        if not price_data:
            return render_template('error.html', message="Pricing details not found.")

        # ✅ Get Dynamic Pricing Factor
        museum_pricing = get_museum_pricing(museum_name)
        print(f"🔍 Debug: Pricing Data for {museum_name}: {museum_pricing}")  # Debugging line

        if museum_pricing and museum_pricing["factor_status"] == 1:
            pricing_factor = museum_pricing["pricing_factor"]
            print(f"✅ Applying Dynamic Pricing: {pricing_factor}x")  # Debugging line
        else:
            pricing_factor = 1.0  # Normal price
            print("❌ No Dynamic Pricing Applied")  # Debugging line

        # ✅ Apply Dynamic Pricing Only When `factor_status == 1`
        adult_price = convert_to_int(price_data["adult_price"]) * pricing_factor
        children_price = convert_to_int(price_data["children_price"]) * pricing_factor
        photography_price = convert_to_int(price_data["photography_fee"]) * pricing_factor

        # ✅ Corrected Total Amount Calculation
        total_amount = (
            (int(booking_data["adult_tickets"]) * adult_price) +
            (int(booking_data["children_tickets"]) * children_price) +
            (int(booking_data["photography_tickets"]) * photography_price)
        )

        print(f"🔢 Final Total Amount: ₹{total_amount}")  # Debugging line
        
        if request.method == 'POST':
            if total_amount == 0:
                # ✅ Fix: Directly mark free bookings as paid
                connection = get_connection()
                cursor = connection.cursor()
                cursor.execute("UPDATE ticket_booking SET payment_status = 'Paid' WHERE id = %s", (booking_id,))
                connection.commit()
                cursor.close()
                connection.close()
                return render_template('payment_success.html', booking_id=booking_id)

            # ✅ Process Payment (Razorpay)
            order_data = {
                "amount": total_amount*100,  # Convert to paise
                "currency": "INR",
                "payment_capture": "1"
            }
            order = razorpay_client.order.create(order_data)
            return render_template('razorpay_payment.html', total_amount=total_amount, order_id=order["id"], booking_id=booking_id)

        return render_template('payment.html', booking_data=booking_data, total_amount=total_amount, booking_id=booking_id)

    finally:
        cursor.close()
        connection.close()

@app.route("/payment_success", methods=["POST"])
def payment_success():
    """
    Handles Razorpay payment success callback and updates the database.
    """
    try:
        data = request.get_json()
        payment_id = data.get("razorpay_payment_id")
        booking_id = data.get("booking_id")

        if not payment_id or not booking_id:
            return jsonify({"message": "Invalid request"}), 400

        # Update database to mark the payment as 'Paid'
        connection = create_connection()
        cursor = connection.cursor()
        update_query = """
            UPDATE ticket_booking 
            SET payment_status = 'Paid', razorpay_payment_id = %s
            WHERE id = %s
        """
        cursor.execute(update_query, (payment_id, booking_id))
        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({"message": "Payment successful and database updated"})

    except Exception as e:
        print(f"Error updating payment status: {e}")
        return jsonify({"message": "Server error while updating payment"}), 500

@app.route('/enter_email/<int:booking_id>', methods=['GET', 'POST'])
def enter_email(booking_id):
    if request.method == 'POST':
        user_email = request.form.get('email')
        if not user_email:
            flash("Please enter a valid email address.", "error")
            return redirect(url_for('enter_email', booking_id=booking_id))
        # Generate QR Code
        qr_data = f"http://localhost:5001/ticket/{booking_id}"  # Encode booking ID in QR
        qr_path = f"{QR_DIR}/{booking_id}.png"
        qr = qrcode.make(qr_data)
        qr.save(qr_path)

        # Store QR code path in the database
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = "UPDATE ticket_booking SET payment_status='Paid', user_email=%s, qr_code_path=%s WHERE id=%s"
            cursor.execute(sql, (user_email, qr_path, booking_id))
            connection.commit()
        connection.close()
        
        # Send tickets to the entered email
        send_ticket_email(user_email, booking_id)
        
        flash(f"Tickets have been sent to {user_email}.", "success")
        return redirect(url_for('my_bookings', user_name=session.get('user_name')))

    return render_template('enter_email.html', booking_id=booking_id)

@app.route('/booking_management', methods=['GET', 'POST'])
def booking_management():
    """Fetch booking details and validate the booking ID."""
    if request.method == 'POST':
        booking_id = request.form.get('booking_id')

        connection = create_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)
        cursor.execute("SELECT id, museum_name, visit_date, visit_time FROM ticket_booking WHERE id = %s", (booking_id,))
        booking = cursor.fetchone()
        cursor.close()
        connection.close()

        if not booking:
            flash("❌ Booking ID not found. Please enter a valid ID.", "error")
            return redirect(url_for('booking_management'))

        return redirect(url_for('manage_booking', booking_id=booking_id))

    return render_template('booking_management.html')

@app.route('/management', methods=['GET', 'POST'])
def management():
    """Fetch booking details and validate the booking ID."""
    if request.method == 'POST':
        booking_id = request.form.get('booking_id')

        connection = create_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)
        cursor.execute("SELECT id, museum_name, visit_datetime FROM bookings WHERE id = %s", (booking_id,))
        booking = cursor.fetchone()
        cursor.close()
        connection.close()

        if not booking:
            flash("❌ Booking ID not found. Please enter a valid ID.", "error")
            return redirect(url_for('management'))

        return redirect(url_for('booking_manage', booking_id=booking_id))

    return render_template('management.html')

@app.route('/manage_booking/<int:booking_id>', methods=['GET'])
def manage_booking(booking_id):
    """Display booking details and museum opening hours."""
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)

    cursor.execute("SELECT * FROM ticket_booking WHERE id = %s", (booking_id,))
    booking = cursor.fetchone()

    if not booking:
        flash("❌ Booking ID not found.", "error")
        return redirect(url_for('booking_management'))

    cursor.execute("SELECT opening_hours, holidays FROM museumdetails WHERE name = %s", (booking['museum_name'],))
    museum_info = cursor.fetchone()

    cursor.close()
    connection.close()

    return render_template('manage_booking.html', booking=booking, museum_info=museum_info)

@app.route('/booking_manage/<int:booking_id>', methods=['GET'])
def booking_manage(booking_id):
    """Display booking details and museum opening hours."""
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)

    cursor.execute("SELECT * FROM bookings WHERE id = %s", (booking_id,))
    booking = cursor.fetchone()

    if not booking:
        flash("❌ Booking ID not found.", "error")
        return redirect(url_for('management'))

    cursor.execute("SELECT opening_hours, holidays FROM museumdetails WHERE name = %s", (booking['museum_name'],))
    museum_info = cursor.fetchone()

    cursor.close()
    connection.close()

    return render_template('booking_manage.html', booking=booking, museum_info=museum_info)

@app.route('/change_time_slot/<int:booking_id>/<museum_name>', methods=['GET', 'POST'])
def change_time_slot(booking_id, museum_name):
    """Change booking time ensuring it is within opening hours and not in the past."""
    
    # Fetch existing booking details
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)
    cursor.execute("SELECT visit_date, visit_time FROM ticket_booking WHERE id = %s", (booking_id,))
    booking = cursor.fetchone()
    cursor.close()
    connection.close()

    if not booking:
        flash("❌ Booking not found.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    visit_date = booking['visit_date']
    visit_time = booking['visit_time']

    # Ensure visit_date is in the correct format
    try:
        visit_date_obj = datetime.strptime(visit_date, '%Y-%m-%d')
        visit_date_str = visit_date_obj.strftime('%Y-%m-%d')
    except ValueError:
        flash("❌ Invalid date format in booking details.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    # Check if the existing booking date and time are in the past
    current_datetime = datetime.now()
    visit_datetime_str = f"{visit_date} {visit_time}"
    visit_datetime_obj = datetime.strptime(visit_datetime_str, '%Y-%m-%d %H:%M')
    if visit_datetime_obj < current_datetime:
        flash("❌ Cannot change the time for past events.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    # Fetch museum details (Opening Hours)
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)
    cursor.execute("SELECT opening_hours FROM museumdetails WHERE name = %s", (museum_name,))
    museum_info = cursor.fetchone()
    cursor.close()
    connection.close()

    if not museum_info:
        flash("❌ Museum details not available.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    if request.method == 'POST':
        new_time = request.form.get('new_time')

        if not new_time:
            flash("❌ Please enter a new time.", "error")
            return redirect(url_for('change_time_slot', booking_id=booking_id, museum_name=museum_name))

        # Combine new date and time into a datetime object
        new_datetime_str = f"{visit_date_str} {new_time}"
        new_datetime_obj = datetime.strptime(new_datetime_str, '%Y-%m-%d %H:%M')

        # Check if the new date and time are in the past
        if new_datetime_obj < current_datetime:
            flash("❌ Cannot change the time to a past event.", "error")
            return redirect(url_for('change_time_slot', booking_id=booking_id, museum_name=museum_name))

        # Check if the selected time is within opening hours
        is_open, message = is_museum_open(museum_name, visit_date_str, new_time)
        if not is_open:
            flash(message, "error")
            return redirect(url_for('change_time_slot', booking_id=booking_id, museum_name=museum_name))

        # Update booking time
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("UPDATE ticket_booking SET visit_time = %s WHERE id = %s", (new_time, booking_id))
        connection.commit()
        cursor.close()
        connection.close()

        flash("✅ Time slot changed successfully.", "success")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    return render_template('change_time_slot.html', booking_id=booking_id, museum_name=museum_name, museum_info=museum_info)

@app.route('/change_date_slot/<int:booking_id>/<museum_name>', methods=['GET', 'POST'])
def change_date_slot(booking_id, museum_name):
    """Change booking date ensuring it is not a holiday and not in the past."""
    
    # Fetch existing booking details
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)
    cursor.execute("SELECT visit_date, visit_time FROM ticket_booking WHERE id = %s", (booking_id,))
    booking = cursor.fetchone()
    cursor.close()
    connection.close()

    if not booking:
        flash("❌ Booking not found.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    visit_date = booking['visit_date']
    visit_time = booking['visit_time']

    # Ensure visit_date is in the correct format
    try:
        visit_date_obj = datetime.strptime(visit_date, '%Y-%m-%d')
        visit_date_str = visit_date_obj.strftime('%Y-%m-%d')
    except ValueError:
        flash("❌ Invalid date format in booking details.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    # Check if the existing booking date and time are in the past
    current_datetime = datetime.now()
    visit_datetime_str = f"{visit_date} {visit_time}"
    visit_datetime_obj = datetime.strptime(visit_datetime_str, '%Y-%m-%d %H:%M')
    if visit_datetime_obj < current_datetime:
        flash("❌ Cannot change the date for past events.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    # Fetch museum details (Opening Hours and Holidays)
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)
    cursor.execute("SELECT opening_hours, holidays FROM museumdetails WHERE name = %s", (museum_name,))
    museum_info = cursor.fetchone()
    cursor.close()
    connection.close()

    if not museum_info:
        museum_info = {'opening_hours': 'Not Available', 'holidays': 'Not Available'}

    if request.method == 'POST':
        new_date = request.form.get('new_date')
        new_time = request.form.get('new_time')

        if not new_date or not new_time:
            flash("❌ Please enter both a new date and time.", "error")
            return redirect(url_for('change_date_slot', booking_id=booking_id, museum_name=museum_name))

        # Combine new date and time into a datetime object
        new_datetime_str = f"{new_date} {new_time}"
        new_datetime_obj = datetime.strptime(new_datetime_str, '%Y-%m-%d %H:%M')

        # Check if the new date and time are in the past
        if new_datetime_obj < current_datetime:
            flash("❌ Cannot change the date and time to a past event.", "error")
            return redirect(url_for('change_date_slot', booking_id=booking_id, museum_name=museum_name))

        # Check if the selected date and time are within opening hours and not a holiday
        is_open, message = is_museum_open(museum_name, new_date, new_time)
        if not is_open:
            flash(message, "error")
            return redirect(url_for('change_date_slot', booking_id=booking_id, museum_name=museum_name))

        # Update booking date and time
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("UPDATE ticket_booking SET visit_date = %s, visit_time = %s WHERE id = %s", (new_date, new_time, booking_id))
        connection.commit()
        cursor.close()
        connection.close()

        flash("✅ Date and time slot changed successfully.", "success")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    return render_template('change_date_slot.html', booking_id=booking_id, museum_name=museum_name, museum_info=museum_info)

@app.route('/change_date/<int:booking_id>/<museum_name>', methods=['GET', 'POST'])
def change_date(booking_id, museum_name):
    """Change booking date ensuring it is not a holiday and not in the past."""
    
    # Fetch existing booking details
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)
    cursor.execute("SELECT visit_datetime FROM bookings WHERE id = %s", (booking_id,))
    booking = cursor.fetchone()
    cursor.close()
    connection.close()

    if not booking:
        flash("❌ Booking not found.", "error")
        return redirect(url_for('booking_manage', booking_id=booking_id))

    visit_datetime = booking['visit_datetime']
    visit_datetime_obj = visit_datetime  # No need to parse again
    visit_date_str = visit_datetime_obj.strftime('%Y-%m-%d')

    # Check if the existing booking date is in the past
    current_datetime = datetime.now()
    if visit_datetime_obj < current_datetime:
        flash("❌ Cannot change the date for past events.", "error")
        return redirect(url_for('booking_manage', booking_id=booking_id))

    # Fetch museum details (Opening Hours and Holidays)
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)
    cursor.execute("SELECT opening_hours, holidays FROM museumdetails WHERE name = %s", (museum_name,))
    museum_info = cursor.fetchone()
    cursor.close()
    connection.close()

    if not museum_info:
        museum_info = {'opening_hours': 'Not Available', 'holidays': 'Not Available'}

    if request.method == 'POST':
        new_date = request.form.get('new_date')
        new_time = request.form.get('new_time')

        if not new_date or not new_time:
            flash("❌ Please enter both a new date and time.", "error")
            return redirect(url_for('change_date', booking_id=booking_id, museum_name=museum_name))

        # Combine new date and time into a datetime object
        new_datetime_str = f"{new_date} {new_time}"
        new_datetime_obj = datetime.strptime(new_datetime_str, '%Y-%m-%d %H:%M')

        # Check if the new date and time are in the past
        if new_datetime_obj < current_datetime:
            flash("❌ Cannot change the date and time to a past event.", "error")
            return redirect(url_for('change_date', booking_id=booking_id, museum_name=museum_name))

        # Check if the selected date and time are within opening hours and not a holiday
        is_open, message = is_museum_open(museum_name, new_date, new_time)
        if not is_open:
            flash(message, "error")
            return redirect(url_for('change_date', booking_id=booking_id, museum_name=museum_name))

        # Update booking datetime
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("UPDATE bookings SET visit_datetime = %s WHERE id = %s", (new_datetime_str, booking_id))
        connection.commit()
        cursor.close()
        connection.close()

        flash("✅ Date and time slot changed successfully.", "success")
        return redirect(url_for('booking_manage', booking_id=booking_id))

    return render_template('change_date.html', booking_id=booking_id, museum_name=museum_name, museum_info=museum_info)


@app.route('/cancel_booking/<int:booking_id>/<museum_name>', methods=['GET', 'POST'])
def cancel_booking(booking_id, museum_name):
    """Cancel booking with refund logic."""
    print("🔹 Cancel booking function called")  # Debugging

    # Fetch booking details
    connection = create_connection()
    cursor = connection.cursor(dictionary=True, buffered=True)
    cursor.execute("SELECT visit_date, visit_time, adult_tickets, children_tickets, photography_tickets FROM ticket_booking WHERE id = %s", (booking_id,))
    booking = cursor.fetchone()
    cursor.close()
    connection.close()

    if not booking:
        print("❌ Booking not found.")
        flash("❌ Booking not found.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    visit_datetime = datetime.strptime(f"{booking['visit_date']} {booking['visit_time']}", '%Y-%m-%d %H:%M')
    current_datetime = datetime.now()

    # Prevent past event cancellations
    if visit_datetime < current_datetime:
        print("❌ Cannot cancel past events.")
        flash("❌ Cannot cancel past events.", "error")
        return redirect(url_for('manage_booking', booking_id=booking_id))

    if request.method == 'POST':
        cancel_option = request.form.get('cancel_option')
        print(f"✅ Received cancel option: {cancel_option}")  # Debugging

        if cancel_option == 'all':
            print("🔹 Cancelling all tickets...")  # Debugging
            try:
                connection = create_connection()
                cursor = connection.cursor()
                cursor.execute("SET FOREIGN_KEY_CHECKS=0;")  # Temporarily disable FK checks
                cursor.execute("DELETE FROM ticket_booking WHERE id = %s", (booking_id,))
                deleted_rows = cursor.rowcount
                cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
                connection.commit()
                cursor.close()
                connection.close()

                if deleted_rows == 0:
                    print("❌ No rows deleted. Booking ID not found.")
                    flash("❌ Booking cancellation failed. No matching booking found.", "error")
                else:
                    print(f"✅ Deleted {deleted_rows} booking(s). Redirecting...")  # Debugging
                    flash("✅ Booking cancelled successfully!", "success")

            except Exception as e:
                print(f"❌ Error deleting booking: {e}")  # Debugging
                flash("❌ An error occurred while cancelling the booking.", "error")

            print("✅ Redirecting to manage_booking...")  # Debugging
            return redirect(url_for('manage_booking', booking_id=booking_id))  # Ensure this route exists

        elif cancel_option == 'some':
            adult_tickets = int(request.form.get('adult_tickets', 0))
            children_tickets = int(request.form.get('children_tickets', 0))
            photography_tickets = int(request.form.get('photography_tickets', 0))
            print(f"✅ Cancelling some tickets - Adults: {adult_tickets}, Children: {children_tickets}, Photography: {photography_tickets}")

            try:
                connection = create_connection()
                cursor = connection.cursor()
                cursor.execute("""
                    UPDATE ticket_booking 
                    SET adult_tickets = adult_tickets - %s, 
                        children_tickets = children_tickets - %s, 
                        photography_tickets = photography_tickets - %s 
                    WHERE id = %s
                """, (adult_tickets, children_tickets, photography_tickets, booking_id))
                connection.commit()
                cursor.close()
                connection.close()

                print("✅ Booking updated with new ticket counts")  # Debugging
                flash("✅ Selected tickets cancelled successfully.", "success")

            except Exception as e:
                print(f"❌ Error updating booking: {e}")  # Debugging
                flash("❌ An error occurred while updating the booking.", "error")

            return redirect(url_for('manage_booking', booking_id=booking_id))

    return render_template('cancel_booking.html', booking_id=booking_id, museum_name=museum_name, booking=booking)

@app.route('/booking_cancel/<int:booking_id>/<museum_name>', methods=['GET', 'POST'])
def booking_cancel(booking_id, museum_name):
    """Cancel booking with refund logic."""
    print("🔹 Booking Cancel function called")

    connection = create_connection()
    if not connection:
        print("❌ Database connection failed.")
        flash("❌ Unable to connect to the database.", "error")
        return redirect(url_for('booking_manage', booking_id=booking_id))

    cursor = connection.cursor(dictionary=True, buffered=True)
    cursor.execute("""
        SELECT user_id, visit_datetime, amount_paid, payment_id, 
               adult_tickets, children_tickets, photography_tickets 
        FROM bookings WHERE id = %s
    """, (booking_id,))
    booking = cursor.fetchone()
    cursor.close()
    connection.close()

    if not booking:
        print("❌ Booking not found.")
        flash("❌ Booking not found.", "error")
        return redirect(url_for('booking_manage', booking_id=booking_id))

    user_id = booking['user_id']
    visit_datetime = booking['visit_datetime']
    total_amount_paid = float(booking['amount_paid'])
    payment_id = booking['payment_id']

    # ✅ Fix: Initialize refund_message
    refund_message = ""

    if request.method == 'POST':
        cancel_option = request.form.get('cancel_option')
        print(f"✅ Received cancel option: {cancel_option}")

        if cancel_option == 'all':
            print("🔹 Cancelling all tickets...")

            # Process refund if eligible
            refund_amount, refund_message = get_refund_amount(visit_datetime, total_amount_paid)
            if refund_amount > 0:
                process_razorpay_refund(payment_id, refund_amount, booking_id, user_id)

            try:
                connection = create_connection()
                cursor = connection.cursor()
                cursor.execute("DELETE FROM bookings WHERE id = %s", (booking_id,))
                connection.commit()
                cursor.close()
                connection.close()

                flash("✅ Booking cancelled successfully!", "success")

            except Exception as e:
                print(f"❌ Error deleting booking: {e}")
                flash("❌ An error occurred while cancelling the booking.", "error")

            return redirect(url_for('booking_manage', booking_id=booking_id))

        elif cancel_option == 'some':
            adult_cancel = int(request.form.get('adult_tickets', 0))
            children_cancel = int(request.form.get('children_tickets', 0))
            photography_cancel = int(request.form.get('photography_tickets', 0))
            print(f"✅ Cancelling some tickets - Adults: {adult_cancel}, Children: {children_cancel}, Photography: {photography_cancel}")

            # Calculate the refund for only the canceled tickets
            price_per_adult = total_amount_paid / max(1, booking['adult_tickets'])
            price_per_child = total_amount_paid / max(1, booking['children_tickets'])
            price_per_photo = total_amount_paid / max(1, booking['photography_tickets'])

            refund_for_cancelled_tickets = (adult_cancel * price_per_adult) + \
                                            (children_cancel * price_per_child) + \
                                            (photography_cancel * price_per_photo)

            # Determine refund eligibility
            refund_amount, refund_message = get_refund_amount(visit_datetime, refund_for_cancelled_tickets)

            # Process refund if eligible
            if refund_amount > 0:
                process_razorpay_refund(payment_id, refund_amount, booking_id, user_id)

            try:
                connection = create_connection()
                cursor = connection.cursor()
                cursor.execute("""
                    UPDATE bookings 
                    SET adult_tickets = adult_tickets - %s, 
                        children_tickets = children_tickets - %s, 
                        photography_tickets = photography_tickets - %s 
                    WHERE id = %s
                """, (adult_cancel, children_cancel, photography_cancel, booking_id))
                connection.commit()
                cursor.close()
                connection.close()

                flash("✅ Selected tickets cancelled successfully.", "success")

            except Exception as e:
                print(f"❌ Error updating booking: {e}")
                flash("❌ An error occurred while updating the booking.", "error")

            return redirect(url_for('booking_manage', booking_id=booking_id))

    return render_template('booking_cancel.html', booking_id=booking_id, museum_name=museum_name, booking=booking, refund_message=refund_message)        

@app.route('/chatbot')
def chatbot():
    user_email = session.get('user_email')  # Check if user email is stored in session
    if user_email:
        return render_template('chatbot.html')
    else:
        return render_template('login.html')

@app.route("/chat", methods=["POST"])
def chat():
    global user_languages
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON data missing"}), 400

        user_id = data.get("user_id", "default_user")
        user_message = data.get("message", "").strip()
        user_lang = data.get("language", "en")

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        supported_languages = ["en", "hi", "es", "zh", "ru"]
        if user_lang not in supported_languages:
            user_lang = "en"

        user_languages[user_id] = user_lang

        # Translate incoming message to English
        translated_text = translate_text(user_message, "en") if user_lang != "en" else user_message

        # Call the chatbot response
        bot_response = chatbot_response(user_id, translated_text)

        # Translate back to user's selected language
        translated_response = translate_text(bot_response, user_lang) if user_lang != "en" else bot_response

        return jsonify({"response": translated_response})
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/register-complaint', methods=['POST'])
def register_complaint():
    """Handles complaint submission, stores it in DB, and sends an email to admin."""
    name = request.form.get("name")
    email = request.form.get("email")
    complaint_text = request.form.get("complaint")

    if not name or not email or not complaint_text:
        return jsonify({"success": False, "message": "❌ All fields are required."})

    try:
        # Store complaint in database
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO complaints (name, email, complaint_text, status) 
            VALUES (%s, %s, %s, 'Pending')
        """, (name, email, complaint_text))
        connection.commit()
        cursor.close()
        connection.close()

        # Send email notification to admin
        admin_email = "admin@example.com"  # Change to actual admin email
        send_complaint_email(admin_email, name, email, complaint_text)

        return jsonify({"success": True, "message": "✅ Complaint registered successfully!"})

    except Exception as e:
        print("❌ Error storing complaint:", e)
        return jsonify({"success": False, "message": "❌ Failed to register complaint. Try again."})

def send_complaint_email(admin_email, user_name, user_email, complaint_text):
    """Sends the complaint details to the admin via email."""
    try:
        sender_email = "your-email@gmail.com"  # Use your system email
        sender_password = "your-app-password"  # Use App Password

        subject = f"New Complaint from {user_name}"
        body = f"User: {user_name}\nEmail: {user_email}\n\nComplaint:\n{complaint_text}"

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = admin_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, admin_email, msg.as_string())
        server.quit()

        print("✅ Complaint email sent to admin.")

    except Exception as e:
        print("❌ Error sending complaint email:", e)
        
@app.route('/subscribe', methods=['POST'])
def subscribe():
    """Handles user subscription, stores it in DB, and sends a confirmation email."""
    email = request.form.get("email")

    if not email:
        return jsonify({"success": False, "message": "❌ Please enter a valid email address."})

    try:
        # Store email in database
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO subscribers (email) VALUES (%s)
        """, (email,))
        connection.commit()
        cursor.close()
        connection.close()

        # Send confirmation email
        send_subscription_email(email)

        return jsonify({"success": True, "message": "✅ Subscription successful!"})

    except Exception as e:
        print("❌ Error storing email:", e)
        return jsonify({"success": False, "message": "❌ Subscription failed. Try again."})

def send_subscription_email(user_email):
    """Sends a subscription confirmation email."""
    try:
        sender_email = "your-email@gmail.com"  # Use your system email
        sender_password = "your-app-password"  # Use App Password

        subject = "Subscription Confirmed!"
        body = f"Thank you for subscribing to our newsletter!\n\nYou will now receive the latest updates."

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = user_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, user_email, msg.as_string())
        server.quit()

        print(f"✅ Confirmation email sent to {user_email}")

    except Exception as e:
        print("❌ Error sending confirmation email:", e)

if __name__ == "__main__":
    app.run(debug=True)