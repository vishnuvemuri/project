from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from apscheduler.schedulers.background import BackgroundScheduler
import mysql.connector
import smtplib
import random
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database import get_db_connection,create_connection,fetch_museum_data_by_category, fetch_museum_data_by_name_with_prices, fetch_ticket_prices_by_type, fetch_museum_data, is_museum_open,update_booking_with_date_time,is_museumopen
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
import numpy as np
import os
import razorpay
import qrcode
from email.mime.image import MIMEImage
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

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
    'password': 'Mohana@04',
    'host': 'localhost',
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

def load_pricing_model():
    return joblib.load("dynamic_pricing_model.pkl")

import mysql.connector
from mysql.connector import Error, cursor

def get_museum_pricing(museum_name):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Mohana@04",
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

model = joblib.load('crowd_prediction.pkl')

# User session storage (to track multiple user interactions)
user_sessions = {}

# Function to convert 12-hour time format to 24-hour format
def convert_to_24_hour(time_str):
    try:
        return datetime.strptime(time_str.strip(), "%I:%M %p").time()  # Converts '10:00 a.m.' → 10:00
    except ValueError:
        return None  # Handles cases where conversion fails

# Function to predict crowd level
def predict_crowd(museum_name, date_input, time_str, user_id):
    try:
        # ✅ Parse user input date & time
        selected_datetime = datetime.strptime(f"{date_input} {time_str}", "%Y-%m-%d %H:%M")

        if selected_datetime < datetime.now():
            return "Past times cannot be predicted. Please enter a future date and time."

        # ✅ Load dataset
        df = pd.read_csv(r"C:\\Users\\hp\\OneDrive\\Desktop\\dataset for crowd prediction.csv", encoding="ISO-8859-1")

        # ✅ Find museum data
        museum = df[df['Name'].str.lower().str.strip() == museum_name.lower().strip()]
        if museum.empty:
            return "Museum not found. Please provide a valid museum name."

        # ✅ Check if the selected date is a holiday
        museum_holidays = str(museum.iloc[0]['Holidays']).lower()  # Convert to lowercase for consistency
        weekday_name = selected_datetime.strftime("%A").lower()  # Get weekday name (e.g., "monday")

        if date_input in museum_holidays or weekday_name in museum_holidays:
            return f"{museum_name} is closed on {date_input} due to a holiday."

        # ✅ Convert opening & closing hours from 12-hour to 24-hour format
        opening_hours = museum.iloc[0]['Opening_hours']  # Example: "10:00 a.m. to 5:00 p.m."

        match = re.match(r"(\d{1,2}:\d{2} [apAP][mM]) to (\d{1,2}:\d{2} [apAP][mM])", opening_hours)
        if not match:
            return "Museum opening/closing times are not formatted correctly."

        opening_time = convert_to_24_hour(match.group(1))  # Convert to time object
        closing_time = convert_to_24_hour(match.group(2))  # Convert to time object

        if not opening_time or not closing_time:
            return "Museum opening/closing times are invalid."

        # ✅ Check if the selected time is within museum hours
        selected_time = selected_datetime.time()

        if selected_time < opening_time:
            return f"Museum is not open yet. It opens at {opening_time.strftime('%H:%M')}."
        if selected_time > closing_time:
            return f"Museum is closed for the day. It closed at {closing_time.strftime('%H:%M')}."

        # ✅ Prepare input features
        input_features = np.array([[selected_datetime.weekday(), selected_datetime.hour]])

        # ✅ Load model and predict crowd level
        model = joblib.load('crowd_prediction.pkl')
        predicted_crowd_numeric = int(round(model.predict(input_features)[0]))
        predicted_crowd_numeric = np.clip(predicted_crowd_numeric, 0, 2)

        # ✅ Map numerical prediction to labels
        crowd_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}

        # ✅ Reset session after prediction
        user_sessions[user_id] = {}

        return f"Predicted crowd level at {museum_name} on {date_input} at {time_str} is {crowd_mapping[predicted_crowd_numeric]}"

    except Exception as e:
        return f"Error: {str(e)}"
    
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

@app.route('/crowd_prediction', methods=['GET', 'POST'])
def crowd_prediction():
    if request.method == 'POST':
        museum_name = request.form['museum_name']
        date_input = request.form['date']
        time_str = request.form['time']
        user_id = request.remote_addr  # Use IP as a temporary user identifier

        prediction_result = predict_crowd(museum_name, date_input, time_str, user_id)

        return render_template("crowd_prediction.html", prediction=prediction_result)

    return render_template("crowd_prediction.html")

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
        host="localhost",
        user="root",
        password="Mohana@04",
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

@app.route('/process_chat', methods=['POST'])
def process_chat():
    data = request.get_json()
    user_message = data.get('message', '')

    # Here you can call your second Python file or function to process the message
    # For example, you can use subprocess to run another Python script
    import subprocess

    # Assuming you have a second Python file called `chatbot_processor.py`
    result = subprocess.run(['python', 'chatbot_processor.py', user_message], capture_output=True, text=True)
    
    # Get the output from the second Python file
    chatbot_response = result.stdout.strip()

    return jsonify({'response': chatbot_response})

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
      app.run(host='0.0.0.0', port=8000, debug=False)