import mysql.connector
from mysql.connector import Error, cursor
import joblib
from database1 import create_connection

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
    
print(calculate_total_amount("Salar Jung Museum", "Indian", 1, 0, 0, 0))  # Should apply a factor of 1.5
