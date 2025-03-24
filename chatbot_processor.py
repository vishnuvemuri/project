import json
import pickle
from flask import Flask, request, jsonify, render_template,redirect,url_for
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from database1 import create_connection,get_db_connection,update_booking_with_payment_id,fetch_booking_details,insert_booking,fetch_museum_by_category,fetch_data_of_museum,is_museum_open,fetch_ticket_prices_by_type,update_booking_status,update_booking_with_qr_code,update_booking_with_email,fetch_last_booking_id,update_booking_with_no_total,update_booking_with_feedback,is_future_date
import random
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from rapidfuzz import process
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import razorpay
import qrcode
from io import BytesIO
from email.mime.image import MIMEImage
import smtplib,ssl
from flask_cors import CORS
from translation import detect_language, translate_text
import numpy as np
import pandas as pd
from datetime import datetime
app = Flask(__name__)
CORS(app)

# Store user-selected languages
user_languages = {}

# Initialize Razorpay client
client = razorpay.Client(auth=("rzp_test_Vc1dMULkCvrbi2", "AbFrPLAmRPAQoo4039F79LVq"))

def create_payment_link(user_session, amount, currency="INR", description="Museum Ticket Booking"):
    try:
        print(f"Creating payment link for amount: {amount}, currency: {currency}, description: {description}")

        payment_link = client.payment_link.create({
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
        payment_link_details = client.payment_link.fetch(payment_link_id)
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
    ticket_prices = fetch_ticket_prices_by_type(museum_name, visitor_type)
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

def send_ticket_email(recipient_email, booking_id):
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
vectorizer = joblib.load("vectorizer.pkl")
train_names = joblib.load("train_names.pkl")
train_vectors = joblib.load("train_vectors.pkl")

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

# Initialize geolocator
geolocator = Nominatim(user_agent="museum_locator")

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

def find_best_match(query, vectorizer, train_vectors, train_names, threshold=0.5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, train_vectors)
    best_match_index = similarities.argmax()
    best_match_name = train_names[best_match_index]
    
    # Use difflib to refine the best match
    difflib_scores = [difflib.SequenceMatcher(None, query, name).ratio() for name in train_names]
    best_difflib_match_index = difflib_scores.index(max(difflib_scores))
    best_difflib_match_name = train_names[best_difflib_match_index]
    
    # Combine scores
    combined_scores = [(similarity + difflib_score) / 2 for similarity, difflib_score in zip(similarities[0], difflib_scores)]
    best_combined_match_index = combined_scores.index(max(combined_scores))
    best_combined_match_name = train_names[best_combined_match_index]
    
    # Check if the best match score is above the threshold
    if combined_scores[best_combined_match_index] < threshold:
        return None
    
    return best_combined_match_name

model = joblib.load('crowd_prediction.pkl')

# Function to predict crowd level using the saved model
def predict_crowd(museum_name, date_input, time_str, user_id):
    try:
        selected_datetime = datetime.strptime(f"{date_input} {time_str}", "%d-%m-%Y %H:%M")
        if selected_datetime < datetime.now():
            return "Past times cannot be predicted. Please enter a future date and time."
        
        # Load the dataset for reference
        df = pd.read_csv(r"C:\\Users\\hp\\OneDrive\\Desktop\\dataset for crowd prediction.csv", encoding="ISO-8859-1")
        
        museum = df[df['Name'].str.lower().str.strip() == museum_name.lower().strip()]
        if museum.empty:
            return "Museum not found. Please provide a valid museum name."
        
        if museum.shape[0] == 0:
            return "Error: Museum data is missing. Please check the dataset."
        
        # Convert category to numeric if needed
        category = museum.iloc[0]['Category']
        if isinstance(category, str):
            category = category.split(',')[0].strip()  # Extract first category
            category_mapping = {cat: idx for idx, cat in enumerate(df['Category'].unique())}  # Create mapping
            category = category_mapping.get(category, -1)  # Convert to number, -1 if not found
        
        # Convert location to numeric if needed
        location = museum.iloc[0]['Location']
        if isinstance(location, str):
            location_mapping = {loc: idx for idx, loc in enumerate(df['Location'].unique())}  # Create mapping
            location = location_mapping.get(location, -1)  # Convert to number, -1 if not found
        
        # Extract numerical value from "Required Time"
        required_time = museum.iloc[0]['Required Time']
        if isinstance(required_time, str):
            match = re.search(r'\d+', required_time)  # Extract numeric part
            required_time = float(match.group()) if match else -1  # Convert to float, -1 if not found
        
        input_features = np.array([[selected_datetime.weekday(), selected_datetime.hour, category, location, required_time]])
        predicted_crowd_numeric = int(round(model.predict(input_features)[0]))
        predicted_crowd_numeric = np.clip(predicted_crowd_numeric, 0, 2)
        
        crowd_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
        
        # Reset session after prediction
        user_sessions[user_id] = {}
        
        return f"Predicted crowd level at {museum_name} on {date_input} at {time_str} is {crowd_mapping[predicted_crowd_numeric]}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_museum_details(museum_name):
    corrected_name = find_best_match(museum_name, vectorizer, train_vectors, train_names)

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
        corrected_name = find_best_match(museum_name, vectorizer, train_vectors, train_names)
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
            corrected_museum = find_best_match(museum_query, vectorizer, train_vectors, train_names)
    
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
        corrected_name = find_best_match(museum_name, vectorizer, train_vectors, train_names)
        
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
        corrected_name = find_best_match(text, vectorizer, train_vectors, train_names)
    
        if corrected_name:
            user_session["museum"] = corrected_name  # Store corrected name
            print(f"✅ Corrected museum name: {corrected_name}")  # Debugging
            museum_details = get_museum_details(corrected_name)
            user_session["state"] = "ASK_TYPE"
            return museum_details  # Proceed directly to ticket type selection
    
        return "❌ I couldn't find that museum. Do you want me to recommend one or try entering again?"

    elif user_session["state"] == "ASK_TICKET_MUSEUM":
        corrected_name = find_best_match(text, vectorizer, train_vectors, train_names)
    
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
                corrected_name = find_best_match(user_session["selected_museum"], vectorizer, train_vectors, train_names)
                
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
            ticket_prices = fetch_ticket_prices_by_type(user_session["museum"], text.lower())
            
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
        email_success = send_ticket_email(email, booking_id)
        if email_success:
            user_session["state"] = "ASK_FEEDBACK"
            return "🎟️ Your tickets have been sent to your email. Thank you for booking with us! Please share your feedback about the chatbot and any improvements you would like to see."
        else:
            user_session["state"]="ASK_EMAIL"
            return "❌ Failed to send the email. Please try again."
    
    return f"❌ {message} Please enter another date and time."


@app.route('/')
def home():
    return render_template('chatbot.html')

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
            return jsonify({"error": "Message is required"}), 400  # Bad Request

        # Ensure supported languages, default to English if unsupported
        supported_languages = ["en", "hi", "es", "zh", "ru"]
        if user_lang not in supported_languages:
            user_lang = "en"

        # Store selected language for the user
        user_languages[user_id] = user_lang

        # Only translate if the selected language is different from English
        translated_text = translate_text(user_message, "en") if user_lang != "en" else user_message

        bot_response = chatbot_response(user_id, translated_text)
        translated_response = translate_text(bot_response, user_lang) if user_lang != "en" else bot_response

        return jsonify({"response": translated_response})
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")  # Log error
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True,port=5001)
    #get_museum_details("salar jung museum")
    