import mysql.connector
from mysql.connector import Error
import requests
from datetime import datetime,timedelta
import calendar
import sqlite3
import re
import pymysql
import time
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG)

API_KEY = 'yi8U3ni7qxsREArm1ME1ZyMr9lU5liRl'

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # Seconds

def get_db_connection():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            connection = pymysql.connect(
                host='localhost',
                user='root',
                password='Mohana@04',
                database='museum',
                cursorclass=pymysql.cursors.DictCursor
            )
            logging.info("✅ Database connection successful.")
            return connection
        except pymysql.MySQLError as e:
            logging.error(f"❌ Database connection failed: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY)
            retries += 1

    raise Exception("Database connection failed after multiple attempts.")

def get_public_holidays_for_india(year=datetime.now().year):
    url = f"https://calendarific.com/api/v2/holidays"
    
    params = {
        "api_key": API_KEY,
        "country": "IN",  # Country code for India
        "year": year      # Year for which you want to fetch the holidays
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        # Check if the request was successful
        if response.status_code == 200 and data.get("meta", {}).get("code") == 200:
            holidays = data.get("response", {}).get("holidays", [])
            holiday_dates = [holiday["date"]["iso"] for holiday in holidays]
            return holiday_dates
        else:
            print("Error fetching holidays:", data.get("meta", {}).get("error_detail"))
            return []
    except Exception as e:
        print(f"Error fetching public holidays: {e}")
        return []

def is_public_holiday(booking_date):
    # Get the list of public holidays for the current year
    public_holidays = get_public_holidays_for_india()

    # Check if the entered date is in the public holidays list
    if booking_date in public_holidays:
        return True
    return False

def create_connection():
    """Create and return a connection to the MySQL database with retry logic."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            connection = mysql.connector.connect(
                host='localhost',  # Replace with your MySQL host
                user='root',  # Replace with your MySQL username
                password='Mohana@04',  # Replace with your MySQL password
                database='museum'  # Your database name
            )
            if connection.is_connected():
                logging.info("✅ Database connection successful.")
                return connection

        except Error as e:
            logging.error(f"❌ Database connection failed: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY)
            retries += 1

    raise Exception("❌ Database connection failed after multiple attempts.")

def fetch_museum_data_by_name_with_prices(museum_name, user_type):
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        
        # Ensure that museum_name and user_type are properly formatted
        museum_name = museum_name.strip()
        
        query = """
            SELECT m.name, m.address, m.location, m.opening_hours, m.holidays, m.description, m.required_time,
                   tp.adult_price, tp.children_price, tp.photography_fee, tp.student_fee
            FROM museumdetails m
            JOIN ticketprices tp ON tp.museum_id = m.id
            WHERE m.name = %s AND tp.type = %s
        """
        print(f"Query: {query}")
        print(f"Parameters: {museum_name}, {user_type}")
        
        try:
            cursor.execute(query, (museum_name, user_type))
            result = cursor.fetchone()
            print(f"Fetched data: {result}")  # Debugging line
            
            if result:
                museum_details = {
                    'name': result['name'],
                    'address':result['address'],
                    'location': result['location'],
                    'opening_hours': result['opening_hours'],
                    'holidays': result['holidays'],
                    'description': result['description'],
                    'required_time':result['required_time'],
                    'prices': {
                        'Adult': result['adult_price'],
                        'Children': result['children_price'],
                        'Photography Fee': result['photography_fee'],
                        'Student': result['student_fee']
                    }
                }
                # Fetch all remaining results to clear the cursor
                cursor.fetchall()
                return museum_details
            else:
                return None
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

def fetch_museum_data_by_category(category):
    """
    Fetch museums by category.
    Args:
        category (str): Museum category (e.g., Arts, History).
    Returns:
        list: A list of museums in the category.
    """
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        # Use LIKE with wildcards to match the category in a comma-separated string
        query = "SELECT name, location FROM museumdetails WHERE category LIKE %s"
        try:
            # Add wildcards for partial matching
            cursor.execute(query, (f"%{category}%",))
            museums = cursor.fetchall()
            return museums
        except Error as e:
            print(f"Error fetching museums by category: {e}")
            return []
        finally:
            cursor.close()
            connection.close()
    return []


def fetch_ticket_prices_by_type(museum_name, user_type):
    """
    Fetch ticket prices for a museum and user type.
    Args:
        museum_name (str): Name of the museum.
        user_type (str): Type of user (e.g., Indian/Foreigner).
    Returns:
        dict: Ticket price details.
    """
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT tp.adult_price, tp.children_price, tp.photography_fee, tp.student_fee
            FROM ticketprices tp
            JOIN museumdetails m ON tp.museum_id = m.id
            WHERE m.name = %s AND tp.type = %s
        """
        try:
            cursor.execute(query, (museum_name, user_type))
            ticket_prices = cursor.fetchone()
            return ticket_prices
        except Error as e:
            print(f"Error fetching ticket prices: {e}")
            return None
        finally:
            cursor.close()
            connection.close()
    return None

def fetch_museum_data(museum_name):
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT opening_hours, holidays, required_time FROM museumdetails WHERE name = %s"
        try:
            cursor.execute(query, (museum_name,))
            museum_data = cursor.fetchone()
            # Fetch all remaining results to clear the cursor
            cursor.fetchall()
            return museum_data
        except mysql.connector.Error as e:
            print(f"Error fetching museum data: {e}")
            return None
        finally:
            cursor.close()
            connection.close()
    return None

def is_museum_open(museum_name, booking_date, booking_time):
    """Check if the museum is open at the given date and time."""
    try:
        # Validate the booking date (check if it's a valid date format)
        try:
            # Attempt to parse the booking date
            booking_date_obj = datetime.strptime(booking_date, '%Y-%m-%d')
        except ValueError:
            return False, "Invalid date format. Please enter the date in YYYY-MM-DD format."

        # Fetch museum opening hours and holidays
        connection = create_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            query = """
                SELECT opening_hours, holidays FROM museumdetails WHERE name = %s
            """
            cursor.execute(query, (museum_name,))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            if result:
                opening_hours = result['opening_hours']
                holidays = result['holidays'].split(' and ') if result['holidays'] else []

                # Convert booking date to day of the week (e.g., 'Monday', 'Tuesday')
                day_of_week = calendar.day_name[booking_date_obj.weekday()]

                # Normalize holidays to handle different cases
                holidays = [holiday.strip().lower() for holiday in holidays]

                # Check if the museum is closed on the selected day
                if day_of_week.lower() in holidays:
                    return False, f"Sorry, the museum is closed on {day_of_week}. Please select another date."

                # Check if the museum observes public holidays
                if "public holidays" in holidays:
                    # Check if the booking date is a public holiday
                    if is_public_holiday(booking_date):
                        return False, f"Sorry, the museum is closed on {booking_date} due to a public holiday."

                # Handle opening hours format
                if 'to' in opening_hours:
                    opening_hours = opening_hours.replace('to', ' - ')
                
                if ' - ' not in opening_hours:
                    return False, f"Invalid opening hours format for the museum: {opening_hours}"

                # Extract opening and closing times
                open_time_str, close_time_str = opening_hours.split(' - ')
                
                try:
                    open_time = datetime.strptime(open_time_str.strip(), '%I:%M %p')
                    close_time = datetime.strptime(close_time_str.strip(), '%I:%M %p')
                except ValueError:
                    open_time = datetime.strptime(open_time_str.strip(), '%H:%M')
                    close_time = datetime.strptime(close_time_str.strip(), '%H:%M')

                # Parse booking time
                try:
                    booking_time_obj = datetime.strptime(booking_time, '%I:%M %p')
                except ValueError:
                    booking_time_obj = datetime.strptime(booking_time, '%H:%M')
                
                # Check if the booking time falls within the opening hours
                if open_time <= booking_time_obj < close_time:
                    return True, "Museum is open during this time."
                else:
                    return False, f"Sorry, the museum is closed at {booking_time}."
            else:
                return False, "Museum data not available."
        else:
            return False, "Database connection failed."
    except Exception as e:
        return False, f"Error processing opening hours: {str(e)}"

def is_museumopen(museum_name, visit_date, visit_time):
    """Check if the museum is open at the selected date and time."""
    conn, cursor = create_connection()

    # Fetch museum opening and closing hours
    cursor.execute("SELECT opening_time, closing_time FROM museum WHERE name = ?", (museum_name,))
    museum_hours = cursor.fetchone()

    if not museum_hours:
        conn.close()
        return False, "Museum not found"

    opening_time, closing_time = museum_hours

    # Convert times to datetime objects
    visit_time_obj = datetime.strptime(visit_time, "%H:%M")
    opening_time_obj = datetime.strptime(opening_time, "%H:%M")
    closing_time_obj = datetime.strptime(closing_time, "%H:%M")

    # Validate time
    if not (opening_time_obj <= visit_time_obj <= closing_time_obj):
        conn.close()
        return False, "Visit time is outside museum hours"

    # Check if it's a holiday
    cursor.execute("SELECT 1 FROM holidays WHERE date = ?", (visit_date,))
    if cursor.fetchone():
        conn.close()
        return False, "Museum is closed on this date (holiday)"

    conn.close()
    return True, "Booking allowed"

def execute_query(query, params=None):
    """Execute a database query with optional parameters."""
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            print("Query executed successfully.")
        except mysql.connector.Error as e:
            print(f"Error executing query: {e}")
        finally:
            cursor.close()
            conn.close()
    else:
        print("Failed to establish a database connection.")

def store_booking_in_db(booking_details):
    """Insert complete booking details into the database and return the booking ID."""
    query = """
    INSERT INTO ticket_booking (user_name, museum_name, category, adult_tickets, 
    children_tickets, photography_tickets, visit_date, visit_time)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        booking_details['user_name'],
        booking_details['museum_name'],
        booking_details['category'],
        booking_details['adult_tickets'],
        booking_details['children_tickets'],
        booking_details['photography_tickets'],
        booking_details['visit_date'],
        booking_details['visit_time']
    )
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            booking_id = cursor.lastrowid
            return booking_id
        except mysql.connector.Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    return None

def update_booking_with_date_time(ticket_id, visit_date=None, visit_time=None):
    try:
        connection = create_connection()
        cursor = connection.cursor()

        if visit_date is not None:
            update_query = """
            UPDATE ticket_booking
            SET visit_date = %s, visit_time = %s
            WHERE id = %s
            """
            cursor.execute(update_query, (visit_date, visit_time, ticket_id))
        else:
            update_query = """
            UPDATE ticket_booking
            SET visit_time = %s
            WHERE id = %s
            """
            cursor.execute(update_query, (visit_time, ticket_id))

        connection.commit()
        return cursor.rowcount > 0  # Return True if one row was updated
    except Error as e:
        print(f"Error: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

def store_user_selection_in_db(user_type, museum_name):
    query = """
    INSERT INTO user_selection (user_type, museum_name)
    VALUES (%s, %s)
    """
    params = (user_type, museum_name)
    execute_query(query, params)

def fetch_data_by_user_name(user_name):
    connection = create_connection()
    cursor = connection.cursor()
    query = "SELECT * FROM ticket_booking WHERE user_name = %s"
    cursor.execute(query, (user_name,))
    bookings = cursor.fetchall()  # Fetch all matching bookings
    cursor.close()
    connection.close()
    return bookings


# Test the database functions
if __name__ == "__main__":
    print("Testing database module...")

    # Test fetching museum by category
    print("Museums in the category 'Arts':")
    print(fetch_museum_data_by_category("Arts"))

    # Test fetching museum details by name
    print("\nMuseum details for 'Victoria Memorial Hall':")
    print(fetch_museum_data_by_name_with_prices("Victoria Memorial Hall", "Indian"))

    print(fetch_museum_data("Victoria Memorial Hall"))
    # Test fetching ticket prices
    print("\nTicket prices for 'Victoria Memorial' (Indian):")
    print(fetch_ticket_prices_by_type("Victoria Memorial Hall", "Indian"))
