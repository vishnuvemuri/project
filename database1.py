import mysql.connector
from mysql.connector import Error
import requests
from datetime import datetime
import calendar
import pymysql
import pandas as pd
import logging
import json
import time

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
    connection = get_db_connection()
    if connection:
        cursor = connection.cursor()
        # Use LIKE with wildcards to match the category in a comma-separated string
        query = "SELECT name, location FROM museumdetails WHERE category LIKE %s"
        try:
            # Add wildcards for partial matching
            cursor.execute(query, (f"%{category}%",))
            museums = cursor.fetchall()
            print(f"✅ Fetched museums for category '{category}': {museums}")  # Debugging
            return museums
        except Error as e:
            print(f"❌ Error fetching museums by category: {e}")
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
        dict: Ticket price details, including adult_price, children_price, student_fee, and photography_fee.
              Returns None if no data is found or an error occurs.
    """
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT tp.adult_price, tp.children_price, tp.student_fee, tp.photography_fee
            FROM ticketprices tp
            JOIN museumdetails m ON tp.museum_id = m.id
            WHERE m.name = %s AND tp.type = %s
        """
        try:
            print(f"🔍 Fetching ticket prices for Museum: '{museum_name}', Type: '{user_type}'")  # Debugging

            cursor.execute(query, (museum_name, user_type))
            ticket_prices = cursor.fetchone()

            print(f"✅ Query Result: {ticket_prices}")  # Debugging

            if ticket_prices:
                # Ensure all keys are present in the result, even if some values are NULL
                ticket_prices.setdefault("adult_price", None)
                ticket_prices.setdefault("children_price", None)
                ticket_prices.setdefault("student_fee", None)
                ticket_prices.setdefault("photography_fee", None)

            return ticket_prices

        except Exception as e:
            print(f"⚠️ Error fetching ticket prices: {e}")
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

                normalized_day = day_of_week.lower().rstrip('s')  # Removes 's' at the end if present
                if any(normalized_day == holiday.lower().rstrip('s') for holiday in holidays):
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
                    return False, f"Sorry, the museum is closed at {booking_time}. Museum is opened only from {opening_hours}"
            else:
                return False, "Museum data not available."
        else:
            return False, "Database connection failed."
    except Exception as e:
        return False, f"Error processing opening hours: {str(e)}"

def is_future_date(visit_date, visit_time):
    """Check if the selected visit date and time are in the future."""
    print(f"DEBUG: Checking if date is in the future - Date: {visit_date}, Time: {visit_time}")
    
    try:
        visit_datetime = datetime.strptime(f"{visit_date} {visit_time}", "%Y-%m-%d %H:%M")
        current_datetime = datetime.now()
        is_future = visit_datetime > current_datetime
        print(f"DEBUG: Computed future date status: {is_future}")
        return is_future
    except ValueError as e:
        print(f"ERROR: Invalid date format - {e}")
        return False

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

def fetch_museum_by_category(category):
    """
    Fetch museums by category.
    Args:
        category (str): Museum category (e.g., Arts, History).
    Returns:
        list: A list of dictionaries containing museum details.
    """
    connection = get_db_connection()
    if not connection:
        logging.error("❌ Failed to connect to the database.")
        return []

    try:
        with connection.cursor() as cursor:
            # Use LIKE with wildcards to match the category in a comma-separated string
            query = "SELECT name, location FROM museumdetails WHERE category LIKE %s"
            cursor.execute(query, (f"%{category}%",))
            museums = cursor.fetchall()
            return museums

            # Debug: Print raw data fetched from the database
            print(f"DEBUG: Raw data from database: {museums}")

            # Convert tuples to dictionaries for easier access
            museums_list = [{"name": name, "location": location} for name, location in museums]
            logging.debug(f"✅ Fetched museums for category '{category}': {museums_list}")
            return museums_list

    except Error as e:
        logging.error(f"❌ Error fetching museums by category: {e}")
        return []
    finally:
        connection.close()

# Fetch museum data from database
def fetch_data_of_museum():
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, name, address, location, opening_hours, holidays, coordinates FROM museumdetails")
        data = cursor.fetchall()
    connection.close()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df[['latitude', 'longitude']] = df['coordinates'].str.strip('()').str.split(',', expand=True).astype(float)
    return df

def insert_booking(user_id, museum_name, visit_datetime, type, adult_tickets, children_tickets,
                   photography_tickets, student_passes, adult_names, children_names, amount_paid):
    """Insert a new booking into the database."""
    try:
        conn = get_db_connection()
        if conn is None:
            print("DEBUG: Failed to connect to the database.")
            return False

        cursor = conn.cursor()

        # Convert lists to JSON strings for proper storage
        adult_names_str = json.dumps(adult_names)
        children_names_str = json.dumps(children_names)

        # Convert amount_paid to string
        amount_paid_str = str(amount_paid)

        # SQL query with placeholders for all columns
        query = """
        INSERT INTO bookings 
        (user_id, museum_name, visit_datetime, type, adult_tickets, children_tickets, 
         photography_tickets, student_passes, adult_names, children_names, amount_paid)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Values to be inserted into the database
        values = (user_id, museum_name, visit_datetime, type, adult_tickets, children_tickets,
                  photography_tickets, student_passes, adult_names_str, children_names_str, amount_paid_str)

        # Print debug info before execution
        print(f"DEBUG: Executing query: {query}")
        print(f"DEBUG: Values: {values}")

        # Execute the query
        cursor.execute(query, values)
        conn.commit()  # Commit the transaction
        print("DEBUG: Booking successfully inserted into the database.")
        
        cursor.close()
        conn.close()
        return True  # Return True if the insertion is successful

    except Exception as e:
        print(f"DEBUG: Database Insertion Error - {e}")
        if conn:
            conn.rollback()  # Rollback transaction in case of failure
        return False


def update_booking_with_payment_id(booking_id, payment_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            query = """
                UPDATE bookings
                SET payment_id = %s
                WHERE id = %s
            """
            cursor.execute(query, (payment_id, booking_id))
            connection.commit()
            return True
    except Exception as e:
        print(f"Error updating payment ID: {e}")
        return False
    finally:
        connection.close()

def update_booking_status(booking_id, status):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            query = """
                UPDATE bookings
                SET payment_status = %s
                WHERE id = %s
            """
            cursor.execute(query, (status, booking_id))
            connection.commit()
            return True
    except Exception as e:
        print(f"Error updating booking status: {e}")
        return False
    finally:
        connection.close()
        
def update_booking_with_no_total(booking_id, status):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            query = """
                UPDATE bookings
                SET payment_status = %s
                WHERE id = %s
            """
            cursor.execute(query, (status, booking_id))
            connection.commit()
            return True
    except Exception as e:
        print(f"Error updating booking status: {e}")
        return False
    finally:
        connection.close()
     
def update_booking_with_email(booking_id, user_email):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            query = """
                UPDATE bookings
                SET user_email = %s
                WHERE id = %s
            """
            cursor.execute(query, (user_email, booking_id))
            connection.commit()
            return True
    except Exception as e:
        print(f"Error updating booking email: {e}")
        return False
    finally:
        connection.close()

def update_booking_with_qr_code(booking_id, qr_code_path):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        query = """
            UPDATE bookings
            SET qr_code_path = %s
            WHERE id = %s
        """
        cursor.execute(query, (qr_code_path, booking_id))
        connection.commit()
        cursor.close()
        connection.close()
        return True
    return False

def fetch_booking_details(booking_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            query = """
                SELECT * FROM bookings
                WHERE id = %s
            """
            cursor.execute(query, (booking_id,))
            result = cursor.fetchone()
            return result
    except Exception as e:
        print(f"Error fetching booking details: {e}")
        return None
    finally:
        connection.close()
        
def fetch_last_booking_id(user_id):
    """Fetch the most recent booking ID for the given user."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            query = """
                SELECT id FROM bookings
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            return result["id"] if result else None
    except Exception as e:
        print(f"Error fetching last booking ID: {e}")
        return None
    finally:
        connection.close()

def store_user_email_in_bookings(user_id):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        query = """
        INSERT INTO bookings (user_id)
        VALUES (%s)
        ON DUPLICATE KEY UPDATE user_id = VALUES(user_id)
        """
        cursor.execute(query, (user_id,))
    connection.commit()
    connection.close()
    
def update_booking_with_feedback(booking_id, feedback):
    connection = create_connection()
    cursor = connection.cursor()
    query = "UPDATE bookings SET feedback = %s WHERE id = %s"
    cursor.execute(query, (feedback, booking_id))
    connection.commit()
    cursor.close()
    connection.close()

# Test the database functions
if __name__ == "__main__":
    print("Testing database module...")

    # Test fetching museum by category
    print("Museums in the category 'Arts':")
    print(fetch_museum_data_by_category("Archeology Museum"))

    # Test fetching museum details by name
    print("\nMuseum details for 'Victoria Memorial Hall':")
    print(fetch_museum_data_by_name_with_prices("Victoria Memorial Hall", "Indian"))

    print(fetch_museum_data("Victoria Memorial Hall"))
    # Test fetching ticket prices
    print("\nTicket prices for 'Victoria Memorial' (Indian):")
    print(fetch_ticket_prices_by_type("Victoria Memorial Hall", "Indian"))
    
    print("Museum details by category")
    print(fetch_museum_by_category("Historical Museums"))
