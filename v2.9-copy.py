import os
import sys
import random
import time
import logging
import json
# import subprocess # No longer explicitly used, playwright handles browser launch
import threading
import datetime
import re
from typing import Optional, List, Dict, Tuple, Union, Callable # Added Callable
import socket # Needed for proxy check

# --- GUI Libraries ---
try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                 QLineEdit, QPushButton, QTextEdit, QCheckBox, QSpinBox,
                                 QDoubleSpinBox, QFileDialog, QMessageBox, QTabWidget, QGroupBox, QFormLayout,
                                 QTableWidget, QTableWidgetItem, QComboBox, QProgressBar, QSizePolicy) # Added QProgressBar, QSizePolicy
    from PyQt6.QtCore import QThread, pyqtSignal, QSettings, QTime, Qt, QTimer, QObject # Added QObject
    from PyQt6.QtGui import QIcon, QPixmap, QPalette, QColor # Added QPalette, QColor
except ImportError:
    print("Missing required PyQt6 library. Please run 'pip install PyQt6'")
    sys.exit(1)


# --- Other Libraries ---
try:
    # Use pysocks for low-level SOCKS check if available, playwright for full check
    import socks
    HAS_PYSOCKS = True
except ImportError:
    HAS_PYSOCKS = False
    print("Optional: 'pip install PySocks' for faster basic SOCKS proxy checks.")

try:
    from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext, Playwright, Error as PlaywrightError
    import google.generativeai as genai  # For Gemini API integration
    # Import Pillow here for logo creation if needed later
    from PIL import Image, ImageDraw
except ImportError:
    # Improved error message for missing core libraries
    missing = []
    try: from playwright.sync_api import sync_playwright
    except ImportError: missing.append("playwright")
    try: import google.generativeai
    except ImportError: missing.append("google-generativeai")
    try: import PIL
    except ImportError: missing.append("Pillow")

    err_msg = f"Missing required libraries: {', '.join(missing)}. Please run:\n\n" \
              f"1. pip install {' '.join(missing)}\n" \
              f"2. playwright install\n\n" \
              f"Then restart the application."
    print(err_msg)
    # Show a GUI message box if PyQt is available but others are missing
    try:
        app_temp = QApplication.instance() or QApplication(sys.argv)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Dependency Error")
        msg_box.setText(f"Missing required Python libraries ({', '.join(missing)}) or Playwright browsers.")
        msg_box.setInformativeText("Please install them by running:\n\n"
                                  f"1. pip install {' '.join(missing)}\n"
                                  "2. playwright install\n\n"
                                  "Then restart the application.")
        msg_box.exec()
    except NameError: # PyQt6 not imported yet
        pass # Message already printed to console
    sys.exit(1)

# --- Project Structure ---
def _ensure_structure():
    """Ensures necessary directories and empty files exist."""
    for dir_path in ("core", "config", "logs", "data", "resources"):
        os.makedirs(dir_path, exist_ok=True)
    for file_path in ("config/settings.py", "config/proxies.txt", "logs/bot_activity.log", "logs/errors.log",
                      "data/user_agents.json", "data/important_words.json", "data/generated_user_agents.json"):
        if not os.path.exists(file_path):
            try:
                with open(file_path, "w", encoding='utf-8') as f:
                    if file_path.endswith(".json"):
                        f.write("{}")  # Empty JSON object
                    elif file_path.endswith(".py"):
                         f.write("# Bot Settings\n") # Empty Python file
                    else:
                        f.write("")  # Empty text file
            except IOError as e:
                print(f"Warning: Could not create file {file_path}: {e}")
                logging.warning(f"Could not create file {file_path}: {e}")

_ensure_structure()

# --- Logo Creation (Optional) ---
LOGO_PATH = "resources/chromium_blue.png"
if not os.path.exists(LOGO_PATH):
    try:
        # Create a basic blue logo (replace with your actual logo file)
        img = Image.new('RGB', (64, 64), color=(30, 144, 255)) # DodgerBlue
        draw = ImageDraw.Draw(img)
        # Simple 'A' for Aladdin - Requires a font file for better look, basic text here
        # For simplicity, we draw a simple shape or skip text if font handling is complex
        draw.rectangle([(15, 15), (49, 49)], fill=(255, 255, 255))
        # draw.text((20, 15), "A", fill=(255, 255, 255)) # Basic text
        img.save(LOGO_PATH)
        logging.info(f"Created default logo at {LOGO_PATH}")
    except NameError:
         logging.warning(f"Pillow library was needed but not fully imported (likely due to earlier error). Cannot create default logo if {LOGO_PATH} is missing.")
    except Exception as e:
        print(f"Error creating simple blue logo: {e}. Please provide a valid logo file at {LOGO_PATH}.")
        logging.error(f"Error creating simple blue logo: {e}")


# --- Logging Setup ---
# Close existing handlers to avoid duplication if script is re-run in some environments
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    handler.close() # Explicitly close file handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s", # Added thread name
    handlers=[
        logging.FileHandler("logs/bot_activity.log", encoding='utf-8', mode='a'), # Specify encoding, append mode
        logging.StreamHandler(sys.stdout), # Keep console output
    ],
)
# Configure error logger separately
error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)
error_logger.propagate = False # Prevent duplicate logging to root
error_log_path = "logs/errors.log"
error_handler = logging.FileHandler(error_log_path, encoding='utf-8', mode='a') # Specify encoding, append mode
error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(threadName)s] - %(filename)s:%(lineno)d - %(message)s") # More details
error_handler.setFormatter(error_formatter)

# Ensure handler is not added multiple times if logger already exists
if not error_logger.hasHandlers():
     error_logger.addHandler(error_handler)

# --- Configuration (config/settings.py) ---
DEFAULT_SETTINGS = {
    "PROXY_ENABLED": False,
    "PROXY_TYPE": "socks5",
    "PROXY_STRING": "", # Store the raw proxy string here as well
    "HEADLESS": True,
    "MIN_DELAY": 2.0, # Use floats for delays
    "MAX_DELAY": 5.0,
    "GEMINI_API_KEY": "",
    "LICENSE_KEY": "",
    "USER_AGENTS_FILE": "data/user_agents.json",
    "IMPORTANT_WORDS_FILE": "data/important_words.json",
    "VIEWPORT_MIN_WIDTH": 1024, # Slightly larger default min
    "VIEWPORT_MAX_WIDTH": 1920,
    "VIEWPORT_MIN_HEIGHT": 768,
    "VIEWPORT_MAX_HEIGHT": 1080,
    "CHROMIUM_BLUE_ENABLED": False,
    "CHROMIUM_BLUE_PATH": "",
    "CHROMIUM_BLUE_ARGS": "",
    "MOUSE_MOVEMENT_ENABLED": True,
    "CONCURRENT_BROWSERS": 1, # Default for UI, actual value from spinbox
    "SCROLL_DURATION_MIN": 500, # ms
    "SCROLL_DURATION_MAX": 1500, # ms
    "FORM_FILL_ENABLED": False,
    "IMPRESSION_ENABLED": False,
    "NEXT_PAGE_SELECTOR": ".next-page, .next, a[rel='next'], [aria-label*='next'], [class*='pagination'] a:last-child", # Expanded selectors
    "AD_CLICK_ENABLED": False,
    "AD_SELECTOR": ".ad-link, .advertisement, [id*='ad'], [class*='ad'], iframe[src*='googleads']", # Expanded selectors
    "AD_CLICK_PROBABILITY": 0.1, # Float
    "TOTAL_RUNS": 1,
    "RUN_GROUP_SIZE": 1,
    "USER_AGENT_GENERATION_ENABLED": False,
    "GENERATED_USER_AGENTS_FILE": "data/generated_user_agents.json",
    "USER_AGENT_GENERATION_COUNT": 10,
    "PROXY_CHECK_TIMEOUT": 15, # Seconds
    "PROXY_CHECK_URL": "http://httpbin.org/ip", # Simple IP check URL
}


def load_settings() -> Dict:
    """Loads settings from config/settings.py, using defaults if necessary."""
    settings_path = "config/settings.py"
    loaded_settings = DEFAULT_SETTINGS.copy()
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding='utf-8') as f:
                settings_code = f.read()
                # Execute in a specific scope to avoid polluting globals
                settings_module = {}
                # Add builtins for safety, although eval/exec is inherently risky with external files
                # A safer approach would be parsing, but exec is common for simple config files
                exec(settings_code, {"__builtins__": __builtins__}, settings_module)

                for k, v in settings_module.items():
                    if k.isupper() and k in loaded_settings:
                         # Basic type validation/conversion based on default type
                         default_type = type(DEFAULT_SETTINGS.get(k))
                         if default_type is not type(None) and not isinstance(v, default_type):
                             try:
                                 # Attempt conversion (e.g., "True" -> True, "5.0" -> 5.0)
                                 if default_type == bool:
                                     v = str(v).lower() in ('true', '1', 'yes')
                                 elif default_type == list and isinstance(v, str): # Handle list stored as comma-sep string? (Better to store as list)
                                      v = [item.strip() for item in v.split(',') if item.strip()]
                                 else:
                                     v = default_type(v)
                                 logging.info(f"Converted setting '{k}' from type {type(settings_module[k]).__name__} to {default_type.__name__}")
                             except (ValueError, TypeError):
                                 logging.warning(f"Setting '{k}' in {settings_path} has incorrect type ({type(v).__name__}), expected {default_type.__name__}. Using default.")
                                 v = DEFAULT_SETTINGS[k] # Use default if conversion fails
                         loaded_settings[k] = v
                    elif not k.startswith('_'): # Avoid loading private variables
                         logging.warning(f"Ignoring unknown setting '{k}' from {settings_path}")

            logging.info(f"Loaded settings from {settings_path}")
        else:
            logging.warning(f"{settings_path} not found. Using default settings and creating the file.")
            save_settings_to_file(loaded_settings) # Create file with defaults
    except SyntaxError as e:
        error_logger.error(f"Syntax error loading settings from {settings_path}: {e}. Using default settings.")
        loaded_settings = DEFAULT_SETTINGS.copy()
    except Exception as e:
        error_logger.error(f"Error loading settings from {settings_path}: {e}. Using default settings.", exc_info=True)
        loaded_settings = DEFAULT_SETTINGS.copy()
    return loaded_settings

def save_settings_to_file(settings_dict: Dict, filepath: str = "config/settings.py"):
    """Saves a settings dictionary to config/settings.py"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
             f.write("# Aladdin Traffic Bot Settings\n")
             f.write("# Auto-generated by the application\n\n")
             # Save only keys present in DEFAULT_SETTINGS to avoid saving temporary state
             for key in DEFAULT_SETTINGS.keys():
                 value = settings_dict.get(key, DEFAULT_SETTINGS[key]) # Use current or default
                 # Represent value correctly in Python syntax
                 if isinstance(value, str):
                     # Escape backslashes and quotes in strings
                     safe_value = value.replace('\\', '\\\\').replace("'", "\\'").replace("\n", "\\n")
                     f.write(f"{key} = '{safe_value}'\n")
                 elif isinstance(value, bool):
                     f.write(f"{key} = {value}\n") # True / False
                 elif isinstance(value, (int, float)):
                     f.write(f"{key} = {value}\n")
                 elif isinstance(value, list):
                      # Save list as a Python list literal
                      list_repr = "[" + ", ".join(repr(item) for item in value) + "]"
                      f.write(f"{key} = {list_repr}\n")
                 else:
                      logging.warning(f"Skipping settings key '{key}' with unhandled type {type(value).__name__} during save.")
        logging.info(f"Settings saved successfully to {filepath}")
        return True
    except Exception as e:
        error_logger.error(f"Failed to save settings to {filepath}: {e}", exc_info=True)
        # Optionally show message box if GUI is running
        if QApplication.instance():
            QMessageBox.critical(None, "Error", f"Failed to save settings to {filepath}: {e}")
        return False

# --- Initial Load ---
settings = load_settings()


# --- Load Data Files ---
def load_json_data(file_path: str, key_name: str) -> List[str]:
    """Loads a list of strings from a JSON file under a specific key."""
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"Data file not found or path invalid: '{file_path}'. Returning an empty list.")
        return []
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            items = data.get(key_name, [])
            if not isinstance(items, list):
                logging.warning(f"Data under key '{key_name}' in {file_path} is not a list. Returning empty list.")
                return []
            # Ensure all items are strings and filter out empty ones
            return [str(item).strip() for item in items if str(item).strip()]
    except json.JSONDecodeError as e:
        error_logger.error(f"Error decoding JSON from {file_path}: {e}. Returning an empty list.")
        return []
    except Exception as e:
        error_logger.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)
        return []

def save_json_data(file_path: str, key_name: str, data_list: List[str]):
    """Saves a list of strings to a JSON file under a specific key."""
    try:
        # Ensure directory exists before trying to save
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = {key_name: data_list}
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved {len(data_list)} items to {file_path} under key '{key_name}'")
    except Exception as e:
        error_logger.error(f"Error saving data to {file_path}: {e}", exc_info=True)


user_agents = load_json_data(settings.get("USER_AGENTS_FILE", "data/user_agents.json"), "user_agents")
generated_user_agents = load_json_data(settings.get("GENERATED_USER_AGENTS_FILE", "data/generated_user_agents.json"), "generated_user_agents")
important_words = load_json_data(settings.get("IMPORTANT_WORDS_FILE", "data/important_words.json"), "important_words")

def save_generated_user_agents(user_agents_list: List[str]):
    """Saves generated user agents to the specified JSON file."""
    save_json_data(settings.get("GENERATED_USER_AGENTS_FILE", "data/generated_user_agents.json"), "generated_user_agents", user_agents_list)

# --- Core Classes and Functions ---

# --- Proxy Validator ---
class ProxyValidator(QObject):
    """Validates proxy format and checks connectivity."""
    check_complete_signal = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Store settings relevant at init time
        self.timeout_sec = settings.get("PROXY_CHECK_TIMEOUT", 15)
        self.check_url = settings.get("PROXY_CHECK_URL", "http://httpbin.org/ip")

    def validate_proxy_format(self, proxy_string: str, proxy_type: str) -> bool:
        """Validates proxy string format based on type."""
        if not proxy_string: return False
        proxy_type = proxy_type.lower()
        # Basic patterns (allow hostnames and IPs)
        host_pattern = r"[a-zA-Z0-9.-]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}" # Simplified host
        port_pattern = r"\d{1,5}"
        user_pass_pattern = r"[^@:]+" # Allow most chars except @ and :

        if proxy_type == "socks5":
            # host:port
            pattern_noauth = rf"^{host_pattern}:{port_pattern}$"
            # user:pass@host:port (Note: Playwright often uses user:pass:host:port internally, but input might be URL style)
            pattern_auth = rf"^{user_pass_pattern}:{user_pass_pattern}@{host_pattern}:{port_pattern}$"
             # user:pass:host:port (Less common input format, but support?)
            pattern_pw_style = rf"^{user_pass_pattern}:{user_pass_pattern}:{host_pattern}:{port_pattern}$"
            return bool(re.match(pattern_noauth, proxy_string) or re.match(pattern_auth, proxy_string) or re.match(pattern_pw_style, proxy_string))
        elif proxy_type == "socks4":
            # host:port
            pattern_noauth = rf"^{host_pattern}:{port_pattern}$"
            # userid@host:port
            pattern_auth = rf"^{user_pass_pattern}@{host_pattern}:{port_pattern}$"
            return bool(re.match(pattern_noauth, proxy_string) or re.match(pattern_auth, proxy_string))
        elif proxy_type in ["http", "https"]:
             # host:port
            pattern_noauth = rf"^{host_pattern}:{port_pattern}$"
             # user:pass@host:port
            pattern_auth = rf"^{user_pass_pattern}:{user_pass_pattern}@{host_pattern}:{port_pattern}$"
            return bool(re.match(pattern_noauth, proxy_string) or re.match(pattern_auth, proxy_string))
        else:
            logging.error(f"Unknown proxy type for validation: {proxy_type}")
            return False

    def parse_proxy_string(self, proxy_string: str, proxy_type: str) -> Optional[Dict]:
        """Parses the proxy string into a dictionary for Playwright or requests."""
        proxy_type = proxy_type.lower()
        if not self.validate_proxy_format(proxy_string, proxy_type):
             error_logger.error(f"Invalid proxy format for type {proxy_type}: {proxy_string}")
             return None

        proxy_info = {"type": proxy_type}
        user = None
        pwd = None
        host = None
        port = None

        try:
            # Handle user:pass@host:port format first for HTTP/S/SOCKS5
            if '@' in proxy_string:
                auth_part, host_part = proxy_string.split('@', 1)
                if ':' in auth_part:
                    user, pwd = auth_part.split(':', 1)
                else: # Only user ID for SOCKS4a
                    user = auth_part
                host_port_part = host_part
            else: # No '@', format is host:port or user:pass:host:port (SOCKS5 special case)
                host_port_part = proxy_string
                # Check for user:pass:host:port SOCKS5 format
                if proxy_type == "socks5" and host_port_part.count(':') == 3:
                    try:
                        user, pwd, host_maybe, port_str = host_port_part.split(':', 3)
                        # Very basic check if host_maybe looks like host or IP
                        if re.match(r"(\d{1,3}\.){3}\d{1,3}|[a-zA-Z0-9.-]+", host_maybe):
                             host = host_maybe
                             port = int(port_str)
                        else: # Parsing failed, treat as host:port probably
                             user, pwd = None, None # Reset creds
                             # Fallthrough to host:port parsing below
                    except ValueError:
                        pass # Fallthrough if split fails

            # Parse host:port part (if not already parsed by SOCKS5 special case)
            if host is None or port is None:
                 if ':' in host_port_part:
                     # Handle potential IPv6 bracket notation (basic)
                     if host_port_part.startswith('[') and ']:' in host_port_part:
                          host, port_str = host_port_part[1:].split(']:', 1)
                     else:
                          host, port_str = host_port_part.rsplit(':', 1) # Split from right
                     port = int(port_str)
                 else: # Should not happen if validation passed
                     raise ValueError(f"Cannot split host and port in '{host_port_part}'")


            proxy_info['hostname'] = host
            proxy_info['port'] = port
            if user: proxy_info['username'] = user
            if pwd: proxy_info['password'] = pwd

            # --- Determine Playwright 'server' field ---
            server_address = f"{host}:{port}" # Base address

            if proxy_type == "socks5":
                # Playwright SOCKS5 needs protocol prefix ONLY if NO auth is used
                if user and pwd:
                    proxy_info['playwright_server'] = server_address # No prefix for SOCKS5 with auth
                else:
                    proxy_info['playwright_server'] = f"socks5://{server_address}" # Prefix for SOCKS5 without auth
            elif proxy_type == "socks4":
                # Playwright SOCKS4 needs protocol prefix
                proxy_info['playwright_server'] = f"socks4://{server_address}"
            elif proxy_type in ["http", "https"]:
                # Playwright HTTP/S needs protocol prefix
                # Auth is handled via username/password keys, not usually in server URL for Playwright proxy dict
                proxy_info['playwright_server'] = f"{proxy_type}://{server_address}"
            else: # Should not be reached
                 raise ValueError(f"Unsupported proxy type for Playwright server formatting: {proxy_type}")

            # Add generic server URL for other potential uses (e.g., PySocks)
            auth_str = ""
            if user and pwd: auth_str = f"{user}:{pwd}@"
            elif user: auth_str = f"{user}@" # For SOCKS4a user ID
            proxy_info['server_url_full'] = f"{proxy_type}://{auth_str}{server_address}"

            return proxy_info

        except ValueError as e:
             error_logger.error(f"Error parsing proxy string '{proxy_string}' for type {proxy_type}: {e}", exc_info=True)
             return None
        except Exception as e:
             error_logger.error(f"Unexpected error parsing proxy string '{proxy_string}': {e}", exc_info=True)
             return None


    def run_check_in_thread(self, proxy_info: Dict):
        """Runs the connectivity check in a separate thread."""
        if not proxy_info:
            self.check_complete_signal.emit(False, "Status: Invalid Proxy Info")
            return

        if not self.signalsBlocked():
             thread_name = f"ProxyCheck-{proxy_info.get('hostname', 'Unknown')}"
             checker_thread = threading.Thread(target=self._check_connectivity_worker, args=(proxy_info,), name=thread_name, daemon=True)
             checker_thread.start()
        else:
             logging.error("Proxy check signal not ready. Cannot start check thread.")
             self.check_complete_signal.emit(False, "Error: GUI Signal Error")


    def _check_connectivity_worker(self, proxy_info: Dict):
        """Worker function to perform the actual proxy check."""
        is_connected = False
        status_message = "Status: Error during check" # Default message
        try:
            # Update instance variables from global settings *inside the thread* if needed
            # This allows settings changes between checks, though risky if check is long
            self.timeout_sec = settings.get("PROXY_CHECK_TIMEOUT", 15)
            self.check_url = settings.get("PROXY_CHECK_URL", "http://httpbin.org/ip")

            is_connected = self.check_proxy_connectivity(proxy_info)
            if is_connected:
                status_message = "Status: Connection Successful!"
            else:
                # Check if a specific reason was logged (e.g., content mismatch)
                status_message = "Status: Connection Failed"
        except Exception as e:
            log_func = error_logger.error if 'error_logger' in globals() else logging.error
            log_func(f"Exception in proxy check worker for {proxy_info.get('hostname')}: {e}", exc_info=True)
            status_message = f"Status: Error ({type(e).__name__})"
        finally:
            self.check_complete_signal.emit(is_connected, status_message)


    def check_proxy_connectivity(self, proxy_info: Dict) -> bool:
        """Checks proxy connectivity using Playwright. Returns True on success."""
        proxy_type = proxy_info.get("type", "unknown").lower()
        hostname = proxy_info.get('hostname', 'N/A')
        port = proxy_info.get('port', 'N/A')
        check_url = self.check_url
        timeout_ms = self.timeout_sec * 1000

        logging.info(f"Checking {proxy_type.upper()} proxy connectivity to {hostname}:{port} using Playwright ({check_url})...")

        playwright_instance = None
        browser = None
        try:
            playwright_instance = sync_playwright().start()
            proxy_config_pw = {}

            pw_server = proxy_info.get('playwright_server')
            if not pw_server:
                 raise ValueError("Internal error: Missing 'playwright_server' in parsed proxy info.")

            proxy_config_pw['server'] = pw_server
            if 'username' in proxy_info:
                 proxy_config_pw['username'] = proxy_info['username']
            if 'password' in proxy_info:
                 proxy_config_pw['password'] = proxy_info['password']

            # Launch args - minimal for check
            launch_args = ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage']

            # Ensure global user_agents list is available or provide a fallback
            ua_list = globals().get("user_agents", [])
            if not ua_list: ua_list = globals().get("generated_user_agents", [])
            ua_to_use = random.choice(ua_list) if ua_list else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"


            browser = playwright_instance.chromium.launch(
                 headless=True,
                 proxy=proxy_config_pw,
                 args=launch_args,
                 timeout=timeout_ms + 10000 # Increased buffer for launch
            )
            context = browser.new_context(
                ignore_https_errors=True,
                user_agent=ua_to_use
            )
            page = context.new_page()

            logging.info(f"Playwright Check: Navigating to {check_url} via proxy...")
            # Use wait_until='domcontentloaded' for potentially faster check
            response = page.goto(check_url, timeout=timeout_ms, wait_until="domcontentloaded")

            status = response.status if response else 'N/A'
            logging.info(f"Playwright Check: Response status: {status}")

            if response and response.ok:
                 # SUCCESS: Received a 2xx status code through the proxy.
                 logging.info(f"Playwright Check: Proxy connection successful (Status OK) for {hostname}:{port}")
                 # Optional: Verify content briefly
                 # try:
                 #     content = response.text(timeout=3000)
                 #     if proxy_info.get('hostname') in content or "origin" in content: # Check if proxy IP or origin appears
                 #         logging.info("Playwright Check: Content verification passed.")
                 #     else:
                 #         logging.warning(f"Playwright Check: Content verification failed (Proxy IP/Origin not found). Content: {content[:100]}...")
                 #         # return False # Uncomment for stricter check
                 # except Exception as text_err:
                 #     logging.warning(f"Playwright Check: Could not get response text for verification: {text_err}")

                 return True # Consider connected if status is OK
            else:
                 logging.warning(f"Playwright Check: Proxy connection failed (Status: {status}) for {hostname}:{port}.")
                 return False

        except PlaywrightError as e:
             error_message = str(e)
             log_func = error_logger.error
             # More detailed logging for common network errors
             if "Target page, context or browser has been closed" in error_message:
                  log_func(f"Playwright proxy check failed (Page/Context Closed Prematurely): {hostname}:{port} - {error_message}")
             elif "proxy" in error_message.lower() or "timed out" in error_message.lower() or "net::ERR" in error_message:
                  log_func(f"Playwright proxy check failed (Network/Proxy Error): {hostname}:{port} ({proxy_type}) - {error_message}")
             else: # Log full trace for other Playwright errors
                  log_func(f"Playwright proxy check failed: {hostname}:{port} ({proxy_type}) - {e}", exc_info=True)
             return False
        except ValueError as e: # Catch parsing errors passed up
             log_func = error_logger.error
             log_func(f"Proxy check aborted due to parsing error: {e}")
             return False
        except Exception as e:
             log_func = error_logger.error
             log_func(f"Unexpected error during Playwright proxy check for {hostname}:{port}: {e}", exc_info=True)
             return False
        finally:
            # Ensure cleanup happens reliably
            # Use try-except for each close operation
            if 'page' in locals() and page and not page.is_closed():
                try: page.close()
                except Exception as close_e: logging.debug(f"Non-critical error closing page: {close_e}")
            if 'context' in locals() and context:
                try: context.close()
                except Exception as close_e: logging.debug(f"Non-critical error closing context: {close_e}")
            if browser:
                try: browser.close()
                except Exception as close_e: logging.debug(f"Non-critical error closing browser: {close_e}")
            if playwright_instance:
                try: playwright_instance.stop()
                except Exception as stop_e: logging.debug(f"Non-critical error stopping playwright: {stop_e}")
            logging.debug("Playwright instance stopped after proxy check.")


# --- Browser Manager ---
class BrowserManager:
    """Manages Chromium/Chromium Blue browser instances."""

    def __init__(self, playwright: Playwright, proxy_config: Optional[Dict] = None,
                 headless: bool = True,
                 use_chromium_blue: bool = False,
                 chromium_blue_path: str = "",
                 chromium_blue_args: str = ""):
        self.playwright = playwright
        self.proxy_config = proxy_config # Store the pre-parsed config
        self.headless = headless
        self.use_chromium_blue = use_chromium_blue
        self.chromium_blue_path = chromium_blue_path
        # Split args safely, handling quotes if needed in future
        self.chromium_blue_args = [arg for arg in chromium_blue_args.split() if arg] if chromium_blue_args else []
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def start_browser(self, user_agent: Optional[str] = None, viewport_size: Optional[Dict] = None):
        """Starts a new browser instance (Chromium or Chromium Blue)."""
        launch_options = {
            "headless": self.headless,
            "args": list(self.chromium_blue_args) # Start with custom args if any
        }
        # Add some default useful args if not overridden by user
        default_args = [
            '--no-sandbox', # Often needed in Docker/Linux environments
            '--disable-infobars', # Deprecated but sometimes still useful
            '--disable-notifications',
            '--disable-popup-blocking',
            '--disable-dev-shm-usage', # Important in Docker/limited memory
            '--disable-blink-features=AutomationControlled', # Basic stealth
            '--disable-features=site-per-process', # Can help with stability/memory
            '--force-color-profile=srgb', # Consistent color profile
            '--mute-audio', # Mute audio from browser
            # '--enable-features=NetworkService,NetworkServiceInProcess' # Sometimes needed
        ]
        # Headless-specific args
        if self.headless:
             default_args.extend([
                 '--headless', # Ensure headless arg is passed explicitly
                 '--disable-gpu', # Often recommended for headless
                 '--window-size=1920,1080', # Define size for headless
             ])
        else:
             # Non-headless args
             default_args.append('--start-maximized') # Maximize window if not headless


        # Add default args only if not already present in custom args
        current_args_set = set(arg.split('=')[0] for arg in launch_options["args"]) # Check arg name before '='
        for arg in default_args:
            arg_name = arg.split('=')[0]
            if arg_name not in current_args_set:
                launch_options["args"].append(arg)
                current_args_set.add(arg_name) # Track added arg names


        # --- Proxy Configuration ---
        if self.proxy_config:
            launch_options["proxy"] = self.proxy_config
            proxy_server_display = self.proxy_config.get("server", "N/A")
            if '@' in proxy_server_display:
                 proxy_server_display = proxy_server_display.split('@')[-1] # Hide user:pass
            logging.info(f"Attempting to launch browser with proxy: {proxy_server_display} (Type: {self.proxy_config.get('type','N/A')})")
        else:
             logging.info("Launching browser without proxy.")


        # --- Chromium Blue Configuration ---
        if self.use_chromium_blue:
            if not self.chromium_blue_path or not os.path.exists(self.chromium_blue_path):
                error_logger.error(f"Chromium Blue enabled but executable not found or path not set: {self.chromium_blue_path}")
                raise FileNotFoundError(f"Custom browser executable not found at '{self.chromium_blue_path}'. Please check Settings.")
            launch_options["executable_path"] = self.chromium_blue_path
            logging.info(f"Using custom executable: {self.chromium_blue_path}")
        else:
             if "executable_path" in launch_options:
                 del launch_options["executable_path"]


        # --- Launch Browser ---
        try:
            browser_type = self.playwright.chromium
            logging.info(f"Launching browser with options: Headless={launch_options['headless']}, Args={launch_options['args']}, Proxy={bool(launch_options.get('proxy'))}, Exec={launch_options.get('executable_path', 'Default')}")
            # Increased timeout for launch, especially with proxies
            self.browser = browser_type.launch(**launch_options, timeout=90000) # 90s timeout

            # --- Context Configuration ---
            context_options = {
                # Use provided UA or fallback to a default plausible one
                'user_agent': user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
                'viewport': viewport_size or {'width': 1366, 'height': 768}, # Default viewport if none provided
                'bypass_csp': True, # Helps with interactions on some sites
                'accept_downloads': False, # Disable downloads unless needed
                'locale': 'en-US',
                'timezone_id': 'America/New_York', # Common timezone
                'ignore_https_errors': True, # Often needed with proxies or test environments
                # Reduce automation flags further (experimental)
                'java_script_enabled': True,
                # Geolocation (optional, set to specific coords or default)
                # 'geolocation': {'longitude': -74.0060, 'latitude': 40.7128}, # Example: NYC
                # 'permissions': ['geolocation'],
            }
            # Handle non-headless viewport/screen sizing
            if not self.headless:
                 context_options['viewport'] = None # Let maximized window handle viewport
                 # context_options['screen'] = {'width': 1920, 'height': 1080} # Optional: Specify screen size if needed


            self.context = self.browser.new_context(**context_options)

            # Add stealth scripts if desired (requires external js files)
            # try:
            #     with open("stealth.min.js", "r") as f:
            #         stealth_js = f.read()
            #     self.context.add_init_script(stealth_js)
            #     logging.info("Applied stealth.min.js init script.")
            # except FileNotFoundError:
            #     logging.warning("stealth.min.js not found, skipping init script.")
            # except Exception as e:
            #     logging.error(f"Error adding init script: {e}")

            self.page = self.context.new_page()

             # Add/Modify headers - more realistic set
            headers = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br', # Standard encodings
                'Sec-Ch-Ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"', # Example Client Hints (Update periodically)
                'Sec-Ch-Ua-Mobile': '?0', # Usually ?0 for desktop
                'Sec-Ch-Ua-Platform': '"Windows"', # Example platform
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none', # Or 'same-origin', 'cross-site' depending on context
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
                # User-Agent is set via context, no need to set here again
            }
            self.page.set_extra_http_headers(headers)

            logging.info(
                f"Browser context created. Headless: {self.headless}, "
                f"Proxy: {'Enabled' if self.proxy_config else 'Disabled'}, "
                f"Custom Exec: {self.use_chromium_blue}, "
                f"User Agent: {context_options['user_agent'][:60]}..., " # Log truncated UA
                f"Viewport: {context_options['viewport']}"
            )
        except PlaywrightError as e:
            error_msg = f"Playwright Error: Failed to start browser: {e}"
            error_logger.error(error_msg, exc_info=True) # Log traceback for Playwright errors
            if "Executable doesn't exist" in str(e):
                 error_msg += "\n\nEnsure the browser executable path is correct (standard Playwright or Custom). Standard path issues can sometimes be fixed by running 'playwright install'."
            elif "proxy" in str(e).lower() or "net::ERR" in str(e):
                 error_msg += "\n\nThis might be due to an invalid or unreachable proxy configuration. Check proxy settings and connectivity."
            # Raise a more informative error to be caught by the calling thread/GUI
            raise Exception(error_msg) from e
        except FileNotFoundError as e:
            error_logger.error(str(e), exc_info=True)
            raise FileNotFoundError(f"File Not Found Error: {e}. Check paths in settings.") from e
        except Exception as e:
            error_logger.exception("Unexpected error occurred during browser start:")
            raise Exception(f"An unexpected error occurred during browser startup: {e}") from e


    def navigate_to(self, url: str):
        """Navigates to the given URL with improved error handling."""
        if not self.page or self.page.is_closed():
            logging.error("Browser page not available or closed. Cannot navigate.")
            raise Exception("Browser page not available. Cannot navigate.")
        try:
            logging.info(f"Navigating to: {url}")
            response = self.page.goto(
                url,
                timeout=90000, # Increased navigation timeout (90s)
                wait_until="domcontentloaded" # Wait for DOM, generally faster than 'load' or 'networkidle'
            )
            status = response.status if response else "N/A"
            logging.info(f"Navigation to {url} finished with status: {status}")

            if response and not response.ok:
                # Log non-OK status but don't necessarily raise exception unless it's critical (e.g., 4xx, 5xx)
                logging.warning(f"Navigation resulted in non-OK status: {status} for URL: {url}. Content (first 200): {response.text(timeout=2000)[:200]}")
                # Optionally take screenshot on non-OK status
                # self.take_screenshot(f"navigation_error_{status}_{datetime.datetime.now():%H%M%S}.png")
                if status >= 400: # Treat client/server errors as navigation failures
                     raise Exception(f"Navigation failed with HTTP status {status}")

            # Short wait after DOM loaded for basic scripts to potentially run
            self.page.wait_for_timeout(random.randint(500, 1500))

        except PlaywrightError as e:
            error_msg = f"Playwright navigation error to {url}: {e}"
            error_logger.error(error_msg, exc_info=True)
            self.take_screenshot(f"navigation_playwright_error_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
            # Re-raise a more specific error if needed, or let the bot thread handle it
            raise Exception(f"Navigation failed: {e}") from e
        except Exception as e:
            # Catch exceptions possibly raised from the status check above
            error_logger.exception(f"Unexpected error during navigation or status check for {url}:")
            self.take_screenshot(f"navigation_unexpected_error_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
            raise Exception(f"Unexpected navigation error: {e}") from e


    def close_browser(self):
        """Closes the browser instance gracefully."""
        closed_something = False
        # Close page first
        if self.page and not self.page.is_closed():
            try:
                self.page.close()
                logging.debug("Page closed.")
                closed_something = True
            except PlaywrightError as e:
                error_logger.warning(f"Error closing page: {e}")
            self.page = None
        # Then close context
        if self.context and not getattr(self.context, '_closed', False): # Check if closed attribute exists
            try:
                self.context.close()
                logging.debug("Context closed.")
                closed_something = True
            except PlaywrightError as e:
                error_logger.warning(f"Error closing context: {e}")
            self.context = None
         # Finally close browser
        if self.browser and self.browser.is_connected():
            try:
                self.browser.close()
                logging.debug("Browser closed.")
                closed_something = True
            except PlaywrightError as e:
                error_logger.warning(f"Error closing browser: {e}")
            self.browser = None

        if closed_something:
             logging.info("Browser resources closed.")
        else:
             logging.debug("No active browser resources needed closing.")


    def take_screenshot(self, filename: str = "screenshot.png"):
        """Takes a screenshot of the current page."""
        if self.page and not self.page.is_closed():
            try:
                # Ensure filename is safe and create logs dir if needed
                safe_filename = re.sub(r'[^\w\-_\.]', '_', filename) # Basic sanitization
                # Add timestamp to avoid overwrites during long runs
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(safe_filename)
                final_filename = f"{base}_{timestamp}{ext or '.png'}"

                logs_dir = "logs"
                os.makedirs(logs_dir, exist_ok=True)
                screenshot_path = os.path.join(logs_dir, final_filename)

                self.page.screenshot(path=screenshot_path, full_page=False) # Capture viewport by default
                logging.info(f"Screenshot saved to {screenshot_path}")
            except PlaywrightError as e:
                 # Ignore errors if page closes during screenshot attempt
                 if "Target page, context or browser has been closed" not in str(e):
                    error_logger.error(f"Playwright error taking screenshot {final_filename}: {e}")
            except Exception as e:
                error_logger.exception(f"Unexpected error while taking screenshot {final_filename}:")
        else:
            logging.warning("Screenshot requested, but page is not available or closed.")


# --- Fingerprint/User Agent Management ---
class FingerprintManager:
    """Manages User Agents, Viewports, and potentially other fingerprint aspects."""

    def __init__(self, initial_user_agents: List[str], generated_user_agents: List[str], gemini_api_key: Optional[str]):
        # Combine and ensure unique on init
        self.user_agents = list(set(ua for ua in initial_user_agents if ua)) # Filter empty
        self.generated_user_agents = list(set(ua for ua in generated_user_agents if ua)) # Filter empty
        self.gemini_api_key = gemini_api_key
        self.used_user_agents_in_group = set() # Track used user agents within the current group run

        if not self.user_agents and not self.generated_user_agents and not (settings.get("USER_AGENT_GENERATION_ENABLED") and self.gemini_api_key):
             logging.warning("No user agents loaded and generation is disabled or API key missing. Bot may fail or use Playwright default.")


    def get_random_viewport_size(self) -> Dict:
        """Returns a random viewport size within configured bounds."""
        try:
             min_w = int(settings.get("VIEWPORT_MIN_WIDTH", 1024))
             max_w = int(settings.get("VIEWPORT_MAX_WIDTH", 1920))
             min_h = int(settings.get("VIEWPORT_MIN_HEIGHT", 768))
             max_h = int(settings.get("VIEWPORT_MAX_HEIGHT", 1080))
             # Swap if min > max
             if min_w > max_w: min_w, max_w = max_w, min_w
             if min_h > max_h: min_h, max_h = max_h, min_h
             # Ensure minimum size
             min_w = max(320, min_w)
             min_h = max(240, min_h)

             width = random.randint(min_w, max_w)
             height = random.randint(min_h, max_h)
             return {"width": width, "height": height}
        except (ValueError, TypeError): # Handle case where settings might be invalid types
            error_logger.error("Invalid viewport dimensions in settings. Using default 1366x768.", exc_info=True)
            return {"width": 1366, "height": 768}


    def get_user_agent(self, browser_id, update_signal) -> str:
        """Selects a unique random user agent for the current group, generates more if needed."""
        all_available_agents = list(set(self.user_agents + self.generated_user_agents))
        selectable_agents = [ua for ua in all_available_agents if ua and ua not in self.used_user_agents_in_group]

        if not selectable_agents:
            update_signal.emit(f"Bot {browser_id + 1}: All unique UAs used. Trying to replenish...", browser_id)
            logging.info(f"Bot {browser_id + 1}: Ran out of unique user agents for this group run.")

            generation_successful = False
            if settings.get("USER_AGENT_GENERATION_ENABLED") and self.gemini_api_key:
                update_signal.emit(f"Bot {browser_id + 1}: Attempting Gemini UA generation...", browser_id)
                generation_successful = self.generate_user_agents_gemini(browser_id, update_signal)
                if generation_successful:
                    self._reload_all_user_agents() # Reload lists from files
                    all_available_agents = list(set(self.user_agents + self.generated_user_agents))
                    selectable_agents = [ua for ua in all_available_agents if ua and ua not in self.used_user_agents_in_group]
                    if selectable_agents:
                        update_signal.emit(f"Bot {browser_id + 1}: Successfully generated and reloaded UAs.", browser_id)
                    else:
                        logging.warning(f"Bot {browser_id + 1}: Generation successful but yielded no usable *new* UAs for this group.")
                        # Fallthrough to reset logic

            # If generation failed, was disabled, or yielded no new usable UAs, reset the pool for this group
            if not selectable_agents:
                if not (settings.get("USER_AGENT_GENERATION_ENABLED") and self.gemini_api_key):
                    update_signal.emit(f"Bot {browser_id + 1}: UA generation disabled/unavailable. Resetting used UA pool.", browser_id)
                elif not generation_successful:
                     update_signal.emit(f"Bot {browser_id + 1}: UA generation failed. Resetting used UA pool.", browser_id)
                else: # Generation worked but no new usable UAs
                     update_signal.emit(f"Bot {browser_id + 1}: No new UAs usable. Resetting used UA pool.", browser_id)

                logging.info(f"Bot {browser_id + 1}: Resetting used user agent pool for this group.")
                self.used_user_agents_in_group = set()
                # Make all non-empty agents available again (including newly generated if any)
                selectable_agents = [ua for ua in all_available_agents if ua]

            # Final check: If still no agents after all attempts, use a hardcoded default
            if not selectable_agents:
                error_logger.critical(f"Bot {browser_id + 1}: CRITICAL: No user agents available after all attempts! Using hardcoded default.")
                update_signal.emit(f"Bot {browser_id + 1}: CRITICAL: No User Agents available! Using default.", browser_id)
                return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"

        # Select a random agent from the available pool
        chosen_agent = random.choice(selectable_agents)
        self.used_user_agents_in_group.add(chosen_agent) # Mark as used for this group run
        logging.info(f"Bot {browser_id + 1}: Selected User Agent: ...{chosen_agent[-60:]}") # Log end part
        return chosen_agent


    def _reload_all_user_agents(self):
         """Helper to reload both standard and generated user agents."""
         logging.debug("Reloading user agents from data files.")
         # Use settings values directly here
         ua_file = settings.get("USER_AGENTS_FILE", "data/user_agents.json")
         gen_ua_file = settings.get("GENERATED_USER_AGENTS_FILE", "data/generated_user_agents.json")
         self.user_agents = list(set(ua for ua in load_json_data(ua_file, "user_agents") if ua))
         self.generated_user_agents = list(set(ua for ua in load_json_data(gen_ua_file, "generated_user_agents") if ua))


    def generate_user_agents_gemini(self, browser_id, update_signal) -> bool:
        """Generates new user agents using Gemini API and saves them. Returns True on success."""
        if not self.gemini_api_key:
            update_signal.emit(f"Bot {browser_id + 1}: Gemini API key missing. Cannot generate.", browser_id)
            logging.warning(f"Bot {browser_id + 1}: Cannot generate UAs, Gemini API key missing.")
            return False

        count = settings.get("USER_AGENT_GENERATION_COUNT", 10)
        prompt = f"""
        Generate a list of exactly {count} diverse and realistic browser user agent strings.
        Focus on recent versions (last 1-2 years) of major browsers (Chrome, Firefox, Safari, Edge) on common desktop and mobile operating systems (Windows 10/11, macOS, Android, iOS). Avoid very old or obscure combinations.
        Ensure the formatting is correct for each user agent string (e.g., starts with 'Mozilla/5.0').
        Output *only* the user agent strings, each on a new line. Do not include any numbering, markdown, introductory text, explanations, or any surrounding characters like quotes or code blocks.
        """

        try:
            update_signal.emit(f"Bot {browser_id + 1}: Contacting Gemini API (model: gemini-1.5-pro)...", browser_id)
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-pro') # Use the specified model
            # Adjust safety settings - potentially less restrictive for UA generation
            safety_settings=[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]] # Example
            generation_config = genai.types.GenerationConfig(temperature=0.8) # Increase temperature slightly for diversity

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Handle potential content blocking or errors in response more robustly
            if not response.candidates or not hasattr(response.candidates[0], 'content') or not response.text:
                 feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'
                 finish_reason = response.candidates[0].finish_reason if response.candidates and hasattr(response.candidates[0], 'finish_reason') else 'N/A'
                 error_msg = f"Gemini API response invalid. Finish Reason: {finish_reason}, Feedback: {feedback}"
                 update_signal.emit(f"Bot {browser_id + 1}: {error_msg}", browser_id)
                 logging.warning(f"Bot {browser_id + 1}: {error_msg}")
                 return False

            generated_ua_text = response.text

            # Strict validation: Split lines, strip whitespace, check start, non-empty
            new_user_agents = [
                ua.strip() for ua in generated_ua_text.strip().splitlines()
                if ua.strip() and ua.strip().startswith("Mozilla/5.0")
            ]

            if not new_user_agents:
                 update_signal.emit(f"Bot {browser_id + 1}: Gemini API returned no valid UAs after filtering.", browser_id)
                 logging.warning(f"Bot {browser_id + 1}: Gemini API returned no valid UAs. Raw response snippet: '{generated_ua_text[:100]}...'")
                 return False

            if len(new_user_agents) < count:
                update_signal.emit(f"Bot {browser_id + 1}: Gemini generated fewer valid UAs than requested ({len(new_user_agents)}/{count}).", browser_id)
                logging.warning(f"Bot {browser_id + 1}: Gemini generated {len(new_user_agents)}/{count} valid UAs.")

            # Load existing generated agents, add new unique ones, and save
            generated_agents_path = settings.get("GENERATED_USER_AGENTS_FILE", "data/generated_user_agents.json")
            current_generated_agents = set(load_json_data(generated_agents_path, "generated_user_agents"))
            added_count = 0
            all_new_agents = list(current_generated_agents) # Start with existing

            for ua in new_user_agents:
                if ua not in current_generated_agents:
                    all_new_agents.append(ua) # Add to list for saving
                    current_generated_agents.add(ua) # Add to set for uniqueness tracking
                    added_count += 1

            if added_count > 0:
                 save_generated_user_agents(all_new_agents) # Save the combined list
                 update_signal.emit(f"Bot {browser_id + 1}: Saved {added_count} new unique UAs.", browser_id)
                 logging.info(f"Bot {browser_id + 1}: Saved {added_count} new unique UAs generated by Gemini.")
                 return True
            else:
                 update_signal.emit(f"Bot {browser_id + 1}: Generated UAs were duplicates. No new agents saved.", browser_id)
                 logging.info(f"Bot {browser_id + 1}: Gemini generated UAs were duplicates, none saved.")
                 return False # Didn't add *new* ones, so signal as such

        except ImportError: # Handle case where genai wasn't imported successfully earlier
            error_logger.error(f"Bot {browser_id + 1}: Google Generative AI library not available.", exc_info=True)
            update_signal.emit(f"Bot {browser_id + 1}: Gemini library error. Cannot generate.", browser_id)
            return False
        except Exception as e:
            error_logger.error(f"Bot {browser_id + 1}: Error generating UAs with Gemini API: {e}", exc_info=True)
            err_str = str(e).lower()
            if "api key not valid" in err_str:
                 update_signal.emit(f"Bot {browser_id + 1}: Gemini Error: Invalid API Key.", browser_id)
            elif "quota" in err_str or "resource has been exhausted" in err_str:
                 update_signal.emit(f"Bot {browser_id + 1}: Gemini Error: Quota Exceeded.", browser_id)
            elif "model not found" in err_str:
                 update_signal.emit(f"Bot {browser_id + 1}: Gemini Error: Model 'gemini-1.5-pro' not found.", browser_id)
            else:
                 update_signal.emit(f"Bot {browser_id + 1}: Gemini API Error: {type(e).__name__}", browser_id)
            return False

    def reset_used_agents(self):
        """Resets the set of used user agents for a new group run."""
        self.used_user_agents_in_group = set()
        logging.info("Resetting used user agent pool for the new group.")


# --- Interaction Managers (Scrolling, Mouse, Form Fill, etc.) ---

# Updated ScrollingManager Class (Incorporated from User's Input)
class ScrollingManager:
    """Handles human-like scrolling behavior, including Gemini-driven patterns."""

    def __init__(self, page: Page):
        # Page is now required at initialization
        if not page or page.is_closed():
             raise ValueError("ScrollingManager initialized with an invalid or closed page.")
        self.page = page
        # Access global settings directly
        self.scroll_duration_min = settings.get("SCROLL_DURATION_MIN", 500)
        self.scroll_duration_max = settings.get("SCROLL_DURATION_MAX", 1500)

    def smooth_scroll_to(self, scroll_to: int, duration: int):
        """Smoothly scrolls using JS requestAnimationFrame."""
        if not self.page or self.page.is_closed():
            logging.warning("Smooth scroll attempted but page is closed.")
            return
        try:
            current_scroll = self.page.evaluate("window.pageYOffset")
            max_y = self.page.evaluate("document.body.scrollHeight - window.innerHeight")
            clamped_scroll_to = max(0, min(max_y, scroll_to))

            if abs(clamped_scroll_to - current_scroll) < 10: # Don't scroll tiny amounts
                 return

            duration = max(100, min(10000, duration)) # 0.1s to 10s

            js_code = f"""
                (() => {{
                    const startY = window.pageYOffset;
                    const endY = {clamped_scroll_to}; // Use the pre-clamped value
                    const distance = endY - startY;
                    if (Math.abs(distance) < 1) return; // Avoid tiny scrolls
                    const duration = {duration};
                    let startTime = null;
                    const easeInOutCubic = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

                    const animation = (currentTime) => {{
                        if (startTime === null) startTime = currentTime;
                        const timeElapsed = currentTime - startTime;
                        const progress = Math.min(1, timeElapsed / duration);
                        const easedProgress = easeInOutCubic(progress);
                        const newY = startY + distance * easedProgress;
                        window.scrollTo(0, newY);
                        if (timeElapsed < duration && Math.abs(window.pageYOffset - endY) > 1) {{
                            requestAnimationFrame(animation);
                        }}
                    }};
                    requestAnimationFrame(animation);
                }})();
            """
            self.page.evaluate(js_code)
            self.page.wait_for_timeout(duration + random.uniform(100, 200))
        except PlaywrightError as e:
             if "Target page, context or browser has been closed" in str(e) or "execution context was destroyed" in str(e):
                  logging.warning(f"Smooth scroll failed because page/context was closed: {e}")
             else:
                  error_logger.warning(f"Smooth scroll using JS failed: {e}")
        except Exception as e:
             error_logger.exception(f"Unexpected error during smooth scroll:")

    def random_scroll(self, min_scrolls: int = 3, max_scrolls: int = 7):
        """Performs a series of random smooth scrolls."""
        if not self.page or self.page.is_closed(): return
        try:
            num_scrolls = random.randint(min_scrolls, max_scrolls)
            viewport_height = self.page.viewport_size["height"] if self.page.viewport_size else 800
            total_scroll_time_ms = 0
            max_total_duration_ms = 45 * 1000

            logging.debug(f"Performing {num_scrolls} random scrolls...")
            for i in range(num_scrolls):
                if not self.page or self.page.is_closed(): break
                if total_scroll_time_ms >= max_total_duration_ms: break

                scroll_fraction = random.uniform(0.2, 0.9)
                scroll_amount = int(viewport_height * scroll_fraction)
                direction = random.choice([1, 1, -1])

                current_y = self.page.evaluate("window.pageYOffset")
                max_y = self.page.evaluate("document.body.scrollHeight - window.innerHeight")
                target_y = max(0, min(max_y, current_y + scroll_amount * direction))

                if abs(target_y - current_y) < 50: continue

                scroll_duration = random.randint(self.scroll_duration_min, self.scroll_duration_max)
                self.smooth_scroll_to(target_y, scroll_duration)
                if not self.page or self.page.is_closed(): break
                pause_after = random.uniform(0.4, 1.2)
                self.page.wait_for_timeout(int(pause_after * 1000))
                total_scroll_time_ms += scroll_duration + (pause_after * 1000)
            logging.debug("Finished random scroll sequence.")
        except PlaywrightError as e:
             if "Target page, context or browser has been closed" in str(e) or "execution context was destroyed" in str(e):
                 logging.warning(f"Random scroll failed because page/context was closed: {e}")
             else:
                 error_logger.warning(f"Random scroll failed: {e}")
        except Exception as e:
             error_logger.exception(f"Unexpected error during random scroll:")

    # Modified to accept is_running_check
    def gemini_scroll(self, browser_id: int, update_signal: pyqtSignal, is_running_check: Callable):
    
        """Generates human-like scroll instructions using Gemini API."""
        page_height = self.page.evaluate("document.body.scrollHeight")

        if page_height <= 1000:  # Small page
            scroll_instruction_prompt = """
            Describe a very short, natural human-like scroll action on a small webpage (less than 1000 pixels high).
            Focus on quick glances and minimal movement.
            Give instructions *strictly* in the following format, and *only* this format:
            'Scroll down [pixels] pixels over [seconds] seconds, pause [seconds] seconds' OR
            'Scroll up [pixels] pixels over [seconds] seconds, pause [seconds] seconds'
            Provide a sequence of exactly 5 such instructions, separated by commas.
            Example: 'Scroll down 150 pixels over 0.6 seconds, pause 0.3 seconds, Scroll down 200 pixels over 0.9 seconds, pause 0.5 seconds, Scroll up 50 pixels over 0.2 seconds, pause 0.2 seconds, Scroll down 100 pixels over 0.4 seconds, pause 0.4 seconds, Scroll down 80 pixels over 0.3 seconds, pause 0.1 seconds'
            Be concise. Do not include any introductory text or explanations.
            """
        else:
            scroll_instruction_prompt = """Describe a highly realistic, varied, and nuanced human-like scroll action on a webpage for approximately 50-70 seconds (adjust the total time to align with typical human session durations on the target website). The key is to simulate a *conscious, goal-oriented* user engaged with the content, but who also exhibits natural browsing *idiosyncrasies*.

Go beyond basic speed and direction variations. Simulate the *micro-movements* and *hesitations* that characterize human scrolling, and simulate how a reader scans down the page. Incorporate realistic reactions to the *content* encountered, such as pausing longer on visually appealing elements or rereading confusing text.

This is not merely about avoiding detection; it's about convincingly *emulating* human browsing.

Include a complex and intertwined mix of:

    - **Variable Scrolling Speeds and Acceleration:** Start with gentle, slow acceleration from a complete stop. Use smaller scroll amounts and speeds at the beginning, gradually increasing them over the first few seconds. Don't instantly jump to high speeds. Similarly, simulate deceleration before pauses and direction changes.

    - **Micro-Movements (Subpixel Scrolling):** Humans rarely scroll in perfect, straight lines. Introduce tiny, almost imperceptible 'jitters' or corrections during longer scrolls by slightly altering the vertical scroll position (+/- 1-2 pixels) every few milliseconds. Note: This cannot be accomplished with a simple command, but MUST be accomplished with intermediate commands that simulate this movement.

    - **Content-Driven Pauses:** *This is crucial*. Adapt pause durations based on the content encountered:
        - *Headings/Subheadings:* Pause 1-3 seconds.
        - *Images/Videos:* Pause 2-5 seconds (simulate looking at them).
        - *Unfamiliar Words/Dense Text:* Pause 1-4 seconds (simulate rereading).
        - *Empty Areas/Whitespace:* Scroll quickly through (humans skip these areas).
        - *Advertisements:* A very brief pause (0.3 - 1 seconds) then scroll past as if ignoring (or very rarely, a slightly longer pause to look at the add).

    - **"Z-Pattern" Scanning:** Instead of scrolling straight down the page, simulate the natural human tendency to scan in a rough "Z" pattern across the screen. This can be achieved by adding very small horizontal mouse movements (even if the bot doesn't explicitly click) during scrolls, or by making quick pauses at the left and right edges of the viewport.

    - **Intentional Backtracking (Rereading):** After scrolling down a significant distance, occasionally scroll back *up* to re-read a paragraph or section that seemed important or confusing. Make this look purposeful, not random (e.g., scroll back to a specific heading).

    - **Error Correction:** Occasionally simulate a mis-scroll (scrolling too far or in the wrong direction) and then quickly correct it.

    - **Subconscious Hesitations and Interruptions:** Introduce brief, random pauses (0.2 - 0.8 seconds) during scrolling, as if the user was momentarily distracted or thinking. Vary the frequency of these interruptions to make them less predictable. Vary pause durations, so that they appear "real."

    - **Footer Avoidance and Exploration:** As before, avoid reaching the absolute footer. Instead, scroll back up and explore other parts of the page, or to click on a small number of links at the bottom.

    - **Reaction to Page Load Events:** If the webpage dynamically loads content as you scroll (common on social media or infinite scrolling sites), pause briefly after the new content appears to simulate waiting for it to load and then adjust scroll behavior accordingly.

Crucially, these actions should *flow together* seamlessly. The goal is to create a continuous, believable stream of browsing behavior, not a series of disjointed actions.

Give instructions *strictly* in the following format, and *only* this format:

    'Scroll down [pixels] pixels over [seconds] seconds, pause [seconds] seconds' OR
    'Scroll up [pixels] pixels over [seconds] seconds, pause [seconds] seconds' OR
    'Pause [seconds] seconds'

Provide a sequence of instructions that will take approximately 50-70 seconds to execute. Separate instructions with commas. Do *not* number the instructions. Do not include any introductory text or explanations. The output should be *only* the comma-separated instructions."""

        try:
            response = genai.GenerativeModel('gemini-2.0-flash').generate_content(scroll_instruction_prompt)
            scroll_instructions_text = response.text
            update_signal.emit(f"Browser {browser_id}: Gemini Scroll Instructions: {scroll_instructions_text}", browser_id)

            # Basic validation and instruction counting (important for longer sequences)
            if scroll_instructions_text:
                instructions = scroll_instructions_text.strip().split(',')
                num_instructions = len(instructions)
                if num_instructions < 5: #minimum instruction to run
                    update_signal.emit(f"Browser {browser_id}: Warning: Gemini returned fewer instructions than expected.  Using fallback.", browser_id)
                    self.random_scroll() # Fallback
                    return
                update_signal.emit(f"Browser {browser_id}: Received {num_instructions} scroll instructions from Gemini.", browser_id)

            self.execute_scroll_instructions(scroll_instructions_text, browser_id, update_signal) # Execute here


        except Exception as e:
            update_signal.emit(f"Browser {browser_id}: Error generating scroll instructions from Gemini API: {e}", browser_id)
            update_signal.emit(f"Browser {browser_id}: Gemini API Error Details: {e}", browser_id)
            self.random_scroll() # Fallback

    def execute_scroll_instructions(self, instructions_text, browser_id, update_signal):
        """Executes scroll instructions generated by Gemini."""
        if not instructions_text:
            self.random_scroll()
            return

        try:
            # Improved parsing with clearer error handling
            for instruction_set in instructions_text.split(','):  # Split into individual instructions
                instruction_set = instruction_set.strip()
                parts = instruction_set.split()

                if "scroll" in [p.lower() for p in parts]: # Check if it's a scroll instruction
                    try:
                        direction_index = [p.lower() for p in parts].index("scroll") + 1
                        direction = parts[direction_index].lower()
                        pixels_index = [p.lower() for p in parts].index("pixels") -1
                        pixels = int(parts[pixels_index])
                        duration_index = [p.lower() for p in parts].index("over") + 1
                        duration_seconds = float(parts[duration_index])
                    except (ValueError, IndexError) as e:
                        update_signal.emit(f"Browser {browser_id}: Error parsing scroll instruction: {instruction_set}. Skipping. Error: {e}", browser_id)
                        continue
                elif "pause" in [p.lower() for p in parts]:
                    try:
                        duration_index = [p.lower() for p in parts].index("pause") + 1
                        duration_seconds = float(parts[duration_index])
                        time.sleep(duration_seconds)
                        continue
                    except (ValueError, IndexError) as e:
                        update_signal.emit(f"Browser {browser_id}: Error parsing pause instruction: {instruction_set}. Skipping. Error {e}", browser_id)
                        continue

                else:
                    update_signal.emit(f"Browser {browser_id}: Unknown instruction: '{instruction_set}'. Skipping.", browser_id)
                    continue

                if direction == 'down':
                    self.page.evaluate(f"window.scrollBy(0, {pixels});")
                elif direction == 'up':
                    self.page.evaluate(f"window.scrollBy(0, -{pixels});")
                else:
                    update_signal.emit(f"Browser {browser_id}: Unknown scroll direction: {direction}", browser_id)
                    continue  # Skip invalid instruction

                time.sleep(duration_seconds)

        except Exception as e:
            update_signal.emit(f"Browser {browser_id}: Error executing scroll instructions: {e}", browser_id)
            self.random_scroll()  # Fallback on error



# --- Text Selection Manager --- (Assumed unchanged from original structure)
class TextSelectionManager:
    """Handles mouse movement on the page."""
    def __init__(self, page: Optional[Page]):
        self.page = page

    def set_page(self, page: Page):
        self.page = page

    def _check_page(self, operation_name: str, browser_id: int, update_signal) -> bool:
        if not self.page or self.page.is_closed():
            update_signal.emit(f"Bot {browser_id + 1}: Page closed/invalid before {operation_name}.", browser_id)
            logging.warning(f"Browser {browser_id}: Page closed/invalid before {operation_name}.")
            return False
        return True

    def interact_with_page(self, browser_id, update_signal):
        """Performs mouse movement."""
        if not self._check_page("mouse interaction", browser_id, update_signal): return

        if settings.get("MOUSE_MOVEMENT_ENABLED", True):
             api_key = settings.get("GEMINI_API_KEY")
             if api_key:
                 update_signal.emit(f"Bot {browser_id + 1}: Attempting Gemini mouse movement.", browser_id)
                 self.gemini_mouse_movement(browser_id, update_signal)
             else:
                 update_signal.emit(f"Bot {browser_id + 1}: Gemini key missing, using simple mouse movement.", browser_id)
                 self.simple_random_mouse_movement(browser_id, update_signal)
        else:
             update_signal.emit(f"Bot {browser_id + 1}: Mouse movement disabled.", browser_id)

    def simple_random_mouse_movement(self, browser_id, update_signal):
        """Simple fallback: move mouse to a few random points."""
        if not self._check_page("simple mouse movement", browser_id, update_signal): return
        try:
            update_signal.emit(f"Bot {browser_id + 1}: Performing simple random mouse moves.", browser_id)
            viewport = self.page.viewport_size
            if not viewport:
                 update_signal.emit(f"Bot {browser_id + 1}: Cannot get viewport size for mouse move.", browser_id)
                 logging.warning(f"Browser {browser_id}: Viewport size not available for mouse movement.")
                 return

            for _ in range(random.randint(2, 4)): # Move 2-4 times
                if not self._check_page(f"simple move {_ + 1}", browser_id, update_signal): break
                target_x = random.randint(0, viewport['width'] - 1)
                target_y = random.randint(0, viewport['height'] - 1)
                steps = random.randint(5, 20) # Fewer steps for simple moves
                self.page.mouse.move(target_x, target_y, steps=steps)
                random_delay(0.2, 0.6) # Short pause after move
        except PlaywrightError as e:
            if "Target page, context or browser has been closed" not in str(e):
                 update_signal.emit(f"Bot {browser_id + 1}: Playwright error during simple mouse move: {e}", browser_id)
                 error_logger.warning(f"Browser {browser_id}: Playwright error during simple mouse move: {e}")
        except Exception as e:
            update_signal.emit(f"Bot {browser_id + 1}: Unexpected error during simple mouse move: {e}", browser_id)
            error_logger.exception(f"Browser {browser_id}: Unexpected error during simple mouse movement:")

    def gemini_mouse_movement(self, browser_id, update_signal):
        """Generates and executes mouse path using Gemini API."""
        if not self._check_page("Gemini mouse movement", browser_id, update_signal): return
        api_key = settings.get("GEMINI_API_KEY")
        if not api_key:
            self.simple_random_mouse_movement(browser_id, update_signal)
            return

        try:
            viewport = self.page.viewport_size
            if not viewport:
                 update_signal.emit(f"Bot {browser_id + 1}: Cannot get viewport for Gemini mouse. Using simple.", browser_id)
                 self.simple_random_mouse_movement(browser_id, update_signal)
                 return

            update_signal.emit(f"Bot {browser_id + 1}: Requesting Gemini mouse instructions.", browser_id)
            prompt = f"""
            Generate a short, natural human-like mouse movement path within a {viewport['width']}x{viewport['height']} viewport.
            Simulate browsing: start near top-middle, move towards center or an edge, pause briefly, maybe one more short move.
            Include 2-4 steps total (moves + pauses). Keep durations short (0.5-1.5s per move/pause).
            Output *only* a semicolon-separated list of instructions in the *exact* format: 'Move to x=[x], y=[y] over [s]s' OR 'Pause [s]s'.
            Coordinates must be positive integers within the viewport [0-{viewport['width']-1}, 0-{viewport['height']-1}]. Durations should be positive floats (e.g., 1.2).
            Example: Move to x=600, y=300 over 0.8s; Pause 0.6s; Move to x=950, y=500 over 1.1s
            Do not include any other text, explanations, or code blocks.
            """
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            safety_settings=[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
            generation_config = genai.types.GenerationConfig(temperature=0.6)
            response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)

            if not response.candidates or not hasattr(response.candidates[0], 'content') or not response.text:
                feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'
                finish_reason = response.candidates[0].finish_reason if response.candidates and hasattr(response.candidates[0], 'finish_reason') else 'N/A'
                error_msg = f"Gemini mouse response invalid. Finish Reason: {finish_reason}, Feedback: {feedback}. Using simple movement."
                update_signal.emit(f"Bot {browser_id + 1}: {error_msg}", browser_id)
                logging.warning(f"Bot {browser_id + 1}: {error_msg}")
                self.simple_random_mouse_movement(browser_id, update_signal)
                return

            mouse_path_instructions_text = response.text.strip()
            if not mouse_path_instructions_text:
                 update_signal.emit(f"Bot {browser_id + 1}: Gemini returned empty mouse instructions. Using simple.", browser_id)
                 self.simple_random_mouse_movement(browser_id, update_signal)
                 return

            update_signal.emit(f"Bot {browser_id + 1}: Executing Gemini mouse instructions...", browser_id)
            logging.info(f"Bot {browser_id + 1}: Gemini mouse instructions: {mouse_path_instructions_text}")
            self.execute_mouse_path_instructions(mouse_path_instructions_text, browser_id, update_signal)

        except ImportError:
             update_signal.emit(f"Bot {browser_id + 1}: Gemini library error. Using simple movement.", browser_id)
             error_logger.error("Gemini library import failed during mouse attempt.")
             self.simple_random_mouse_movement(browser_id, update_signal)
        except Exception as e:
            update_signal.emit(f"Bot {browser_id + 1}: Error during Gemini mouse: {type(e).__name__}. Using simple.", browser_id)
            error_logger.error(f"Bot {browser_id + 1}: Error in gemini_mouse_movement: {e}", exc_info=True)
            self.simple_random_mouse_movement(browser_id, update_signal) # Fallback

    def execute_mouse_path_instructions(self, instructions_text: str, browser_id: int, update_signal):
        """Executes mouse path instructions generated by Gemini."""
        if not self._check_page("execute Gemini mouse path", browser_id, update_signal): return

        move_pattern = re.compile(r"Move\s+to\s+x=(\d+),\s*y=(\d+)\s+over\s+([\d.]+)\s*s", re.IGNORECASE)
        pause_pattern = re.compile(r"Pause\s+([\d.]+)\s*s", re.IGNORECASE)
        instructions = instructions_text.split(';')

        for instruction_part in instructions:
            if not self._check_page(f"execute mouse instruction '{instruction_part[:20]}...'", browser_id, update_signal): break
            instruction_part = instruction_part.strip()
            move_match = move_pattern.match(instruction_part)
            pause_match = pause_pattern.match(instruction_part)

            if move_match:
                try:
                    target_x = int(move_match.group(1))
                    target_y = int(move_match.group(2))
                    duration_sec = float(move_match.group(3))
                    if duration_sec <= 0: raise ValueError("Duration must be positive")
                    steps = max(5, int(duration_sec * 15))

                    viewport = self.page.viewport_size or {'width': 1920, 'height': 1080}
                    target_x = max(0, min(target_x, viewport['width'] - 1))
                    target_y = max(0, min(target_y, viewport['height'] - 1))

                    update_signal.emit(f"Bot {browser_id + 1}: Exec: Move mouse to ({target_x}, {target_y}) / {duration_sec:.1f}s", browser_id)
                    self.page.mouse.move(target_x, target_y, steps=steps)
                except (PlaywrightError, ValueError) as e:
                    if "Target page, context or browser has been closed" not in str(e):
                        update_signal.emit(f"Bot {browser_id + 1}: Error executing move instruction '{instruction_part}': {e}", browser_id)
                        error_logger.warning(f"Bot {browser_id + 1}: Error during Gemini mouse move execution: {e}")
                    break # Stop on error
                except Exception as e:
                     update_signal.emit(f"Bot {browser_id + 1}: Unexpected error executing move '{instruction_part}': {e}", browser_id)
                     error_logger.exception(f"Bot {browser_id + 1}: Unexpected error executing Gemini move:")
                     break

            elif pause_match:
                try:
                    pause_sec = float(pause_match.group(1))
                    if pause_sec <=0: raise ValueError("Pause must be positive")
                    update_signal.emit(f"Bot {browser_id + 1}: Exec: Pause mouse {pause_sec:.1f}s", browser_id)
                    time.sleep(pause_sec) # Use time.sleep for pause
                except ValueError as e:
                     logging.warning(f"Bot {browser_id + 1}: Skipping invalid pause instruction '{instruction_part}': {e}")
                     continue
                except Exception as e:
                     update_signal.emit(f"Bot {browser_id + 1}: Unexpected error during pause '{instruction_part}': {e}", browser_id)
                     error_logger.exception(f"Bot {browser_id + 1}: Unexpected error during Gemini pause:")
                     break
            elif instruction_part: # Only warn if it's not an empty string from splitting
                logging.warning(f"Bot {browser_id + 1}: Skipping unrecognized mouse instruction: '{instruction_part}'")

        logging.info(f"Bot {browser_id + 1}: Finished executing Gemini mouse instructions.")


# --- Next Page Navigator --- (Assumed unchanged)
class NextPageNavigator:
    """Handles navigation to the next page in pagination."""
    def __init__(self, page: Optional[Page]):
        self.page = page
        self.next_page_selector = settings.get("NEXT_PAGE_SELECTOR", ".next-page") # Get from settings

    def set_page(self, page: Page):
        self.page = page

    def _check_page(self, operation_name: str, browser_id: int, update_signal) -> bool:
        if not self.page or self.page.is_closed():
            update_signal.emit(f"Bot {browser_id + 1}: Page closed/invalid before {operation_name}.", browser_id)
            logging.warning(f"Browser {browser_id}: Page closed/invalid before {operation_name}.")
            return False
        return True


    def navigate_next_page(self, browser_id, update_signal) -> bool:
        """Finds and clicks a next page element. Returns True if navigation likely succeeded."""
        if not settings.get("IMPRESSION_ENABLED", False): return False # Feature disabled
        if not self._check_page("next page navigation", browser_id, update_signal): return False

        current_selector = self.next_page_selector # Use selector loaded at init
        if not current_selector:
            update_signal.emit(f"Bot {browser_id + 1}: Next page selector empty. Cannot navigate.", browser_id)
            return False

        try:
            update_signal.emit(f"Bot {browser_id + 1}: Searching for next page link (selector: {current_selector[:60]}...).", browser_id)
            possible_links = self.page.locator(current_selector)
            next_link = None

            # Prioritize links with common 'next' text or symbols, check visibility
            common_texts_patterns = [r'next\b', r'>\s*$', r'»'] # \b for word boundary, $ for end
            for pattern in common_texts_patterns:
                 try:
                      link_with_text = possible_links.filter(has_text=re.compile(pattern, re.IGNORECASE)).first
                      if link_with_text.is_visible(timeout=1500): # Slightly longer timeout
                          next_link = link_with_text
                          update_signal.emit(f"Bot {browser_id + 1}: Found candidate 'next' link (pattern: '{pattern}').", browser_id)
                          break
                 except PlaywrightError: continue # Element might have disappeared or timeout

            # If no specific match, take the first visible element matching the generic selector
            if not next_link:
                 update_signal.emit(f"Bot {browser_id + 1}: No specific text match, checking generic visible selector matches...", browser_id)
                 try:
                      count = possible_links.count()
                      # Iterate through first few matches if many exist
                      for i in range(min(count, 5)):
                          generic_link = possible_links.nth(i)
                          if generic_link.is_visible(timeout=1000):
                               next_link = generic_link
                               update_signal.emit(f"Bot {browser_id + 1}: Found generic visible next link (index {i}).", browser_id)
                               break
                 except PlaywrightError: pass # No visible generic match either

            if next_link:
                update_signal.emit(f"Bot {browser_id + 1}: Clicking next page element.", browser_id)
                initial_url = self.page.url
                try:
                    # Use expect_navigation for reliability
                    with self.page.expect_navigation(timeout=30000, wait_until="domcontentloaded"):
                         next_link.click(timeout=10000) # Timeout for the click action itself
                    # Check if URL actually changed after navigation finishes
                    current_url = self.page.url
                    if current_url != initial_url:
                         update_signal.emit(f"Bot {browser_id + 1}: Successfully navigated to next page: {current_url}", browser_id)
                         random_delay(1, 3) # Wait after successful navigation
                         return True
                    else:
                         update_signal.emit(f"Bot {browser_id + 1}: Clicked element, but URL did not change. Impression may have failed.", browser_id)
                         return False
                except PlaywrightError as click_nav_err:
                    err_str = str(click_nav_err)
                    if "Timeout" in err_str and "navigation" in err_str:
                        update_signal.emit(f"Bot {browser_id + 1}: Clicked next link, but navigation timed out.", browser_id)
                    elif "Target page, context or browser has been closed" in err_str:
                         update_signal.emit(f"Bot {browser_id + 1}: Page closed during next page click/navigation.", browser_id)
                    else:
                        update_signal.emit(f"Bot {browser_id + 1}: Error during next page click/navigation: {err_str}", browser_id)
                    error_logger.warning(f"Browser {browser_id}: Error clicking/navigating next page: {click_nav_err}")
                    return False
            else:
                update_signal.emit(f"Bot {browser_id + 1}: No visible next page element found.", browser_id)
                return False
        except PlaywrightError as e:
            if "Target page, context or browser has been closed" not in str(e):
                 update_signal.emit(f"Bot {browser_id + 1}: Playwright error searching for next page link: {e}", browser_id)
                 error_logger.warning(f"Browser {browser_id}: Playwright error during next page search: {e}")
            return False
        except Exception as e:
            update_signal.emit(f"Bot {browser_id + 1}: Unexpected error during next page navigation: {e}", browser_id)
            error_logger.exception(f"Browser {browser_id}: Unexpected error during next page navigation:")
            return False


# --- Ad Click Manager --- (Assumed unchanged)
class AdClickManager:
    """Handles ad detection and clicking."""
    def __init__(self, page: Optional[Page]):
        self.page = page
        self.ad_selector = settings.get("AD_SELECTOR", "[id*='ad']")
        self.ad_click_probability = settings.get("AD_CLICK_PROBABILITY", 0.1)

    def set_page(self, page: Page):
        self.page = page

    def _check_page(self, operation_name: str, browser_id: int, update_signal) -> bool:
        if not self.page or self.page.is_closed():
            update_signal.emit(f"Bot {browser_id + 1}: Page closed/invalid before {operation_name}.", browser_id)
            logging.warning(f"Browser {browser_id}: Page closed/invalid before {operation_name}.")
            return False
        return True

    def click_ad(self, browser_id, update_signal) -> bool:
        """Attempts to find and click an ad based on probability."""
        if not settings.get("AD_CLICK_ENABLED", False): return False
        if not self._check_page("ad click", browser_id, update_signal): return False

        current_selector = self.ad_selector # Use loaded selector
        current_prob = self.ad_click_probability # Use loaded probability
        if not current_selector:
            update_signal.emit(f"Bot {browser_id + 1}: Ad selector is empty. Cannot click ads.", browser_id)
            return False

        if random.random() >= current_prob:
            return False # Skipped based on probability

        try:
            update_signal.emit(f"Bot {browser_id + 1}: Searching for ads (selector: {current_selector[:50]}...).", browser_id)
            ad_elements = self.page.locator(current_selector)
            visible_ads = []

            # Efficiently find the first few visible ads
            count = ad_elements.count()
            if count == 0:
                 update_signal.emit(f"Bot {browser_id + 1}: No elements found matching ad selector.", browser_id)
                 return False

            update_signal.emit(f"Bot {browser_id + 1}: Found {count} potential ad elements. Checking visibility...", browser_id)
            # Check first few elements for visibility
            for i in range(min(count, 10)): # Check up to 10 potential ads
                 try:
                      ad = ad_elements.nth(i)
                      # Use bounding_box check for potential visibility and non-zero size
                      box = ad.bounding_box(timeout=500)
                      if box and box['width'] > 0 and box['height'] > 0:
                           # Further check if it's clickable (is_enabled is implicit in click)
                           if ad.is_visible(timeout=200): # Quick final visible check
                                visible_ads.append(ad)
                 except PlaywrightError:
                     continue # Ignore elements that disappear or cause errors

            if visible_ads:
                ad_element_to_click = random.choice(visible_ads)
                update_signal.emit(f"Bot {browser_id + 1}: Found {len(visible_ads)} visible ad(s). Attempting click.", browser_id)
                random_delay(0.8, 2.0) # Pause before click

                original_url = self.page.url
                original_context = self.page.context # Get context to check for new pages
                new_page = None

                try:
                    # Expect a popup (new tab) - common ad behavior
                    with original_context.expect_page(timeout=15000) as new_page_info:
                        ad_element_to_click.click(timeout=10000, modifiers=["Control", "Shift"]) # Try Ctrl+Shift click to force new tab? Or just click()

                    new_page = new_page_info.value
                    update_signal.emit(f"Bot {browser_id + 1}: Ad click opened a new tab: {new_page.url[:80]}...", browser_id)
                    random_delay(3, 7) # Simulate viewing new tab

                    # Close the new ad tab and switch back
                    new_page.close()
                    update_signal.emit(f"Bot {browser_id + 1}: Closed ad tab.", browser_id)
                    if not self.page.is_closed(): # Ensure original page still exists
                        self.page.bring_to_front()
                    return True

                except PlaywrightError as expect_err:
                    # If expect_page times out, click might have navigated in place or done nothing
                    if "Timeout" in str(expect_err) and "expect_page" in str(expect_err):
                        update_signal.emit(f"Bot {browser_id + 1}: Ad click did not open new tab. Checking navigation...", browser_id)
                        # Give navigation a moment
                        self.page.wait_for_timeout(2000)
                        if not self.page.is_closed() and self.page.url != original_url:
                             update_signal.emit(f"Bot {browser_id + 1}: Ad click navigated current page to {self.page.url[:80]}...", browser_id)
                             random_delay(1, 3)
                             try:
                                 self.page.go_back(timeout=15000, wait_until='domcontentloaded')
                                 update_signal.emit(f"Bot {browser_id + 1}: Navigated back.", browser_id)
                                 return True
                             except PlaywrightError as back_err:
                                 update_signal.emit(f"Bot {browser_id + 1}: Could not navigate back: {back_err}.", browser_id)
                                 return False # Failed to return, counts as failed click outcome
                        else:
                             update_signal.emit(f"Bot {browser_id + 1}: Ad click had no apparent effect (no new tab, no navigation).", browser_id)
                             return False
                    elif "Target page, context or browser has been closed" in str(expect_err):
                         update_signal.emit(f"Bot {browser_id + 1}: Page closed during ad click attempt.", browser_id)
                         return False
                    else: # Other error during click or expect_page
                        update_signal.emit(f"Bot {browser_id + 1}: Error during ad click interaction: {expect_err}", browser_id)
                        error_logger.warning(f"Browser {browser_id}: Playwright error during ad click/expect: {expect_err}")
                        return False

            else:
                update_signal.emit(f"Bot {browser_id + 1}: No visible ad elements found.", browser_id)
                return False

        except PlaywrightError as e:
            if "Target page, context or browser has been closed" not in str(e):
                 update_signal.emit(f"Bot {browser_id + 1}: Playwright error during ad search: {e}", browser_id)
                 error_logger.warning(f"Browser {browser_id}: Playwright error during ad search: {e}")
            return False
        except Exception as e:
            update_signal.emit(f"Bot {browser_id + 1}: Unexpected error during ad click attempt: {e}", browser_id)
            error_logger.exception(f"Browser {browser_id}: Unexpected error during ad click:")
            return False


# --- Form Filler --- (Assumed unchanged)
class FormFiller:
    """Fills out forms with realistic typos and corrections."""
    def __init__(self, page: Optional[Page]):
        self.page = page
        self.typo_probability = 0.08  # Reduced typo chance
        self.correction_probability = 0.85 # Increased correction chance

    def set_page(self, page: Page):
        self.page = page

    def _check_page(self, operation_name: str, browser_id: int, update_signal) -> bool:
        if not self.page or self.page.is_closed():
            update_signal.emit(f"Bot {browser_id + 1}: Page closed/invalid before {operation_name}.", browser_id)
            logging.warning(f"Browser {browser_id}: Page closed/invalid before {operation_name}.")
            return False
        return True

    def fill_form(self, browser_id, update_signal):
        """Locates and fills visible form fields on the page."""
        if not settings.get("FORM_FILL_ENABLED", False): return
        if not self._check_page("form filling", browser_id, update_signal): return

        update_signal.emit(f"Bot {browser_id + 1}: Searching for form fields...", browser_id)
        try:
            # Common input types + textarea, visible and enabled
            field_selector = "input[type='text'], input[type='email'], input[type='password'], input[type='search'], input[type='tel'], input[type='url'], textarea"
            form_fields = self.page.locator(f"{field_selector}:visible:enabled") # Use Playwright pseudo-classes
            count = form_fields.count()

            if count == 0:
                update_signal.emit(f"Bot {browser_id + 1}: No visible & enabled form fields found.", browser_id)
                return

            update_signal.emit(f"Bot {browser_id + 1}: Found {count} fields. Attempting to fill...", browser_id)
            filled_count = 0
            for i in range(count):
                if not self._check_page(f"filling field {i+1}", browser_id, update_signal): break
                field = form_fields.nth(i)
                try:
                    field_type = field.get_attribute("type") or field.evaluate("node => node.tagName.toLowerCase()")
                    field_name = field.get_attribute("name") or field.get_attribute("id") or field.get_attribute("aria-label") or f"field_{i}"
                    update_signal.emit(f"Bot {browser_id + 1}: Filling field '{field_name}' (type: {field_type}).", browser_id)

                    value_to_fill = self._generate_input_value(field_type, field_name)
                    field.scroll_into_view_if_needed(timeout=5000) # Ensure field is in view
                    field.click(timeout=5000) # Focus field
                    random_delay(0.1, 0.3)
                    # Use typo simulation
                    self._type_with_typos(field, value_to_fill, browser_id, update_signal)
                    random_delay(0.4, 1.0) # Pause after filling a field
                    filled_count += 1

                except PlaywrightError as e:
                    if "Target page, context or browser has been closed" not in str(e):
                        update_signal.emit(f"Bot {browser_id + 1}: Playwright error interacting with field {i+1} ('{field_name}'): {e}", browser_id)
                        error_logger.warning(f"Browser {browser_id}: Could not fill form field {i+1} ('{field_name}'): {e}")
                    else: break # Stop if page closed
                except Exception as e:
                     update_signal.emit(f"Bot {browser_id + 1}: Unexpected error with field {i+1} ('{field_name}'): {e}", browser_id)
                     error_logger.exception(f"Browser {browser_id}: Unexpected error filling form field {i+1}:")

            update_signal.emit(f"Bot {browser_id + 1}: Form filling attempt complete. Filled {filled_count}/{count} fields.", browser_id)
        except PlaywrightError as e:
             if "Target page, context or browser has been closed" not in str(e):
                 update_signal.emit(f"Bot {browser_id + 1}: Playwright error locating form fields: {e}", browser_id)
                 error_logger.error(f"Browser {browser_id}: Error locating form fields: {e}")
        except Exception as e:
            update_signal.emit(f"Bot {browser_id + 1}: Unexpected error during form filling setup: {e}", browser_id)
            error_logger.exception(f"Browser {browser_id}: Unexpected error during form filling setup:")

    def _generate_input_value(self, field_type: str, field_name: str) -> str:
        """Generates plausible input values based on field type or name hints."""
        field_type = field_type.lower()
        field_name = field_name.lower()
        # Simple random data generation
        names = ["Alex", "Maria", "Sam", "Jordan", "Taylor", "Casey"]
        surnames = ["Smith", "Jones", "Williams", "Brown", "Davis", "Miller"]
        words = ["info", "query", "feedback", "support", "hello", "world", "example", "data", "login", "user"]
        domains = ["example.com", "test.org", "mail.net", "domain.xyz", "email.co"]

        if "email" in field_type or "email" in field_name:
            user = random.choice(words) + str(random.randint(10,999))
            domain = random.choice(domains)
            return f"{user}@{domain}"
        elif "password" in field_type or "pass" in field_name:
            prefix = random.choice(["Pass", "Secret", "Login", "Word", "Key"])
            suffix = random.choice(["!", "#", "$", "%", "?", "&", "*"])
            return f"{prefix}{random.randint(1000,9999)}{random.choice(surnames)}{suffix}"
        elif "name" in field_name:
             if "first" in field_name: return random.choice(names)
             if "last" in field_name or "sur" in field_name: return random.choice(surnames)
             return f"{random.choice(names)} {random.choice(surnames)}"
        elif "phone" in field_name or "tel" in field_type:
             return f"{random.randint(200, 999)}-555-{random.randint(1000, 9999)}" # US format with 555
        elif "search" in field_type or "query" in field_name or "search" in field_name:
             return f"{random.choice(words)} {random.choice(words)} {random.choice(surnames).lower()}"
        elif "subject" in field_name:
             return f"{random.choice(['Inquiry', 'Question', 'Feedback', 'Support Request', 'Regarding'])} {random.choice(words)}"
        elif field_type == "textarea" or "message" in field_name or "comment" in field_name:
            num_sentences = random.randint(1, 3)
            text = []
            available_words = words + [s.lower() for s in surnames] + [n.lower() for n in names]
            for _ in range(num_sentences):
                 num_words = random.randint(6, 15)
                 sentence_words = random.sample(available_words, min(num_words, len(available_words)))
                 sentence = " ".join(sentence_words)
                 text.append(sentence.capitalize() + ".")
            return " ".join(text)
        else: # Default fallback for generic text inputs
            num_words = random.randint(2, 5)
            return " ".join(random.sample(words, min(num_words, len(words))))

    def _type_with_typos(self, field, value: str, browser_id, update_signal):
        """Types into a field with simulated typos, corrections, and delays."""
        if not self._check_page("typing simulation", browser_id, update_signal): return
        try:
            alphabet = "abcdefghijklmnopqrstuvwxyz" # Chars for typos
            for char_index, char in enumerate(value):
                if not self._check_page(f"typing char {char_index+1}", browser_id, update_signal): break
                make_typo = random.random() < self.typo_probability

                if make_typo and char.lower() in alphabet: # Only make typos on letters
                    typo_char = random.choice(alphabet.replace(char.lower(), ''))
                    if char.isupper(): typo_char = typo_char.upper()
                    field.type(typo_char, delay=random.uniform(40, 120))
                    if random.random() < self.correction_probability:
                        random_delay(0.15, 0.5)
                        field.press("Backspace", delay=random.uniform(30, 80))
                        random_delay(0.05, 0.2)
                        field.type(char, delay=random.uniform(50, 150))
                    # else: Intentionally leave the typo
                else:
                    # Type normally
                    field.type(char, delay=random.uniform(50, 150))

                # Small random pause sometimes between characters
                if random.random() < 0.08:
                    random_delay(0.1, 0.35)
        except PlaywrightError as e:
             if "Target page, context or browser has been closed" not in str(e):
                 update_signal.emit(f"Bot {browser_id + 1}: Playwright error during typing: {e}", browser_id)
                 error_logger.warning(f"Browser {browser_id}: Playwright error during typing: {e}")
                 # Fallback to direct fill might be less realistic but ensures field is filled
                 try:
                      update_signal.emit(f"Bot {browser_id + 1}: Typing failed, attempting direct fill.", browser_id)
                      field.fill(value, timeout=5000)
                 except Exception as fill_e:
                      if "Target page, context or browser has been closed" not in str(fill_e):
                           error_logger.warning(f"Browser {browser_id}: Direct fill fallback also failed: {fill_e}")
        except Exception as e:
             update_signal.emit(f"Bot {browser_id + 1}: Unexpected error during typing: {e}", browser_id)
             error_logger.exception(f"Browser {browser_id}: Unexpected error during typing simulation:")


# --- License ---
class LicenseManager:
    """Verifies the license key (placeholder)."""
    # Basic placeholder - replace with actual secure verification logic
    def __init__(self):
        self.expiration_date: Optional[datetime.date] = None
        self.is_valid = False
        self.last_checked_key = None

    def verify_license(self, license_key: str) -> Tuple[bool, str]:
        """Verifies the license key and sets the expiration date."""
        # --- !!! IMPORTANT: Replace this with secure validation !!! ---
        self.last_checked_key = license_key # Store key for re-check on load

        # Simple placeholder validation logic
        lk = license_key.strip() if license_key else ""
        if lk == "VALID_KEY_PLACEHOLDER": # Replace with your actual logic/key
            self.expiration_date = datetime.date.today() + datetime.timedelta(days=365)
            self.is_valid = True
            logging.info(f"License key accepted. Expires: {self.expiration_date}")
            return True, f"Activated (Expires: {self.expiration_date.strftime('%Y-%m-%d')})"
        elif lk == "HamzaAkmal": # Legacy key example
             self.expiration_date = datetime.date.today() + datetime.timedelta(days=30) # Shorter expiry
             self.is_valid = True
             logging.info(f"Legacy license key accepted. Expires: {self.expiration_date}")
             return True, f"Activated [Legacy] (Expires: {self.expiration_date.strftime('%Y-%m-%d')})"
        # Add more checks or call an external validation server here
        else:
            self.is_valid = False
            self.expiration_date = None
            msg = "Invalid license key." if lk else "License key cannot be empty."
            logging.warning(f"Invalid license key provided: '{lk[:5]}...'. Reason: {msg}")
            return False, msg

    def check_license_status(self) -> Tuple[bool, str]:
        """Checks if the current license state is valid and not expired."""
        if not self.is_valid or self.expiration_date is None:
            return False, "License not activated or invalid."

        today = datetime.date.today()
        if today > self.expiration_date:
            self.is_valid = False # Mark as invalid if expired
            return False, f"License expired on {self.expiration_date.strftime('%Y-%m-%d')}."
        else:
            days_left = (self.expiration_date - today).days
            return True, f"Activated (Expires: {self.expiration_date.strftime('%Y-%m-%d')}, {days_left} days left)"

# --- Utility Functions ---
def random_delay(min_seconds: Optional[float] = None, max_seconds: Optional[float] = None):
    """Pauses execution for a random time using settings if args are None."""
    min_s = min_seconds if min_seconds is not None else settings.get("MIN_DELAY", 1.0)
    max_s = max_seconds if max_seconds is not None else settings.get("MAX_DELAY", 3.0)
    try:
        min_s = float(min_s)
        max_s = float(max_s)
    except (ValueError, TypeError):
        min_s = 1.0
        max_s = 3.0

    min_s = max(0.1, min_s) # Minimum delay of 0.1s
    max_s = max(min_s, max_s) # Ensure min <= max

    delay = random.uniform(min_s, max_s)
    time.sleep(delay)

# --- GUI Components and Logic ---

class BotThread(QThread):
    """Thread for running the bot logic for one browser instance."""
    update_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str, int)
    progress_signal = pyqtSignal(int, int)

    def __init__(self, browser_id: int, urls: List[str], proxy_config: Optional[Dict],
                 headless: bool, use_chromium_blue: bool,
                 chromium_blue_path: str, chromium_blue_args: str,
                 gemini_api_key: Optional[str], user_agent_manager: FingerprintManager):
        super().__init__()
        self.browser_id = browser_id
        self.urls = urls
        self.proxy_config = proxy_config
        self.headless = headless
        self.use_chromium_blue = use_chromium_blue
        self.chromium_blue_path = chromium_blue_path
        self.chromium_blue_args = chromium_blue_args
        self.gemini_api_key = gemini_api_key
        self.user_agent_manager = user_agent_manager
        self._is_running = True # Flag to allow stopping

    def stop(self):
        """Signals the thread to stop gracefully."""
        self.update_signal.emit(f"Bot {self.browser_id + 1}: Stop request received.", self.browser_id)
        self._is_running = False

    def run(self):
        """Runs the bot logic for a single browser."""
        self.update_signal.emit(f"Bot {self.browser_id + 1}: Thread starting...", self.browser_id)
        self.progress_signal.emit(0, self.browser_id)
        bot = None

        try:
            with sync_playwright() as playwright:
                 if not self._is_running:
                     self.update_signal.emit(f"Bot {self.browser_id + 1}: Stopped before Playwright context.", self.browser_id)
                     return

                 bot = HumanTrafficBot(
                     playwright=playwright,
                     urls=self.urls,
                     proxy_config=self.proxy_config,
                     headless=self.headless,
                     use_chromium_blue=self.use_chromium_blue,
                     chromium_blue_path=self.chromium_blue_path,
                     chromium_blue_args=self.chromium_blue_args,
                     gemini_api_key=self.gemini_api_key,
                     user_agent_manager=self.user_agent_manager
                 )
                 # Pass the lambda check function
                 bot.run(self.update_signal, self.progress_signal, self.browser_id, lambda: self._is_running)

        except Exception as e:
            # Catch broad exceptions (like Playwright startup, bot init)
            error_msg = f"Critical Error in Bot {self.browser_id + 1}: {e}"
            error_logger.exception(f"Critical error in BotThread {self.browser_id + 1}:")
            self.error_signal.emit(error_msg, self.browser_id)
        finally:
             # Ensure browser resources are closed if bot object exists
             if bot and bot.browser_manager:
                 self.update_signal.emit(f"Bot {self.browser_id + 1}: Closing browser resources...", self.browser_id)
                 bot.browser_manager.close_browser() # Call the manager's close method
             else:
                 self.update_signal.emit(f"Bot {self.browser_id + 1}: Browser resources may not have been initialized.", self.browser_id)

             self.progress_signal.emit(100, self.browser_id) # Signal completion / error stop
             self.finished_signal.emit(self.browser_id) # Signal thread termination
             self.update_signal.emit(f"Bot {self.browser_id + 1}: Thread finished.", self.browser_id)


class HumanTrafficBot:
    """Main bot class orchestrating browser actions."""
    def __init__(self, playwright: Playwright, urls: List[str],
                 proxy_config: Optional[Dict] = None,
                 headless: bool = True,
                 use_chromium_blue: bool = False,
                 chromium_blue_path: str = "",
                 chromium_blue_args: str = "",
                 gemini_api_key: Optional[str] = None,
                 user_agent_manager: Optional[FingerprintManager] = None):

        self.playwright = playwright
        self.urls = urls
        self.gemini_api_key = gemini_api_key

        if user_agent_manager is None:
             error_logger.critical("UserAgentManager not provided to HumanTrafficBot!")
             raise ValueError("UserAgentManager is required for HumanTrafficBot")
        self.fingerprint_manager = user_agent_manager

        self.browser_manager = BrowserManager(
             playwright,
             proxy_config=proxy_config,
             headless=headless,
             use_chromium_blue=use_chromium_blue,
             chromium_blue_path=chromium_blue_path,
             chromium_blue_args=chromium_blue_args
        )

        # Initialize managers to None, instantiate after page exists
        self.scrolling_manager = None
        self.text_selection_manager = None
        self.form_filler = None
        self.next_page_navigator = None
        self.ad_click_manager = None

        # Load feature flags directly from global settings at instance creation
        self.form_fill_enabled = settings.get("FORM_FILL_ENABLED", False)
        self.impression_enabled = settings.get("IMPRESSION_ENABLED", False)
        self.ad_click_enabled = settings.get("AD_CLICK_ENABLED", False)
        self.mouse_movement_enabled = settings.get("MOUSE_MOVEMENT_ENABLED", True)


    def run(self, update_signal: pyqtSignal, progress_signal: pyqtSignal, browser_id: int, is_running_check: Callable):
        """Runs the main bot loop for a single browser instance."""
        total_urls = len(self.urls)
        if total_urls == 0:
             update_signal.emit(f"Bot {browser_id + 1}: No URLs provided. Exiting.", browser_id)
             return

        for i, url in enumerate(self.urls):
             if not is_running_check():
                 update_signal.emit(f"Bot {browser_id + 1}: Stop requested before processing URL {i+1}.", browser_id)
                 break

             progress = int(((i + 0.5) / total_urls) * 95)
             progress_signal.emit(progress, browser_id)
             update_signal.emit(f"Bot {browser_id + 1}: URL {i + 1}/{total_urls}: {url}", browser_id)

             # Reset page-dependent managers for each URL
             self.scrolling_manager = None
             self.text_selection_manager = None
             self.form_filler = None
             self.next_page_navigator = None
             self.ad_click_manager = None

             try:
                 # --- Setup Browser (Handles potential reuse/cleanup implicitly) ---
                 update_signal.emit(f"Bot {browser_id + 1}: Selecting User Agent...", browser_id)
                 user_agent = self.fingerprint_manager.get_user_agent(browser_id, update_signal)
                 viewport_size = self.fingerprint_manager.get_random_viewport_size()

                 # Close previous resources IF reusing the *same* BrowserManager instance
                 # Note: Currently, BrowserManager is created per thread, so this is less critical
                 if self.browser_manager.page and not self.browser_manager.page.is_closed():
                      try: self.browser_manager.page.close()
                      except Exception: pass
                 if self.browser_manager.context and not getattr(self.browser_manager.context, '_closed', False):
                      try: self.browser_manager.context.close()
                      except Exception: pass

                 update_signal.emit(f"Bot {browser_id + 1}: Starting browser context/page...", browser_id)
                 # Start browser (or just context/page if browser is reused, handled internally)
                 self.browser_manager.start_browser(user_agent=user_agent, viewport_size=viewport_size)

                 if not is_running_check(): break

                 # --- Instantiate/Set Page for Managers ---
                 if self.browser_manager.page and not self.browser_manager.page.is_closed():
                     page = self.browser_manager.page
                     # Instantiate managers that require page in __init__
                     self.scrolling_manager = ScrollingManager(page)
                     # Instantiate or set page for others
                     self.text_selection_manager = TextSelectionManager(page)
                     self.form_filler = FormFiller(page)
                     self.next_page_navigator = NextPageNavigator(page)
                     self.ad_click_manager = AdClickManager(page)
                     # If TextSelectionManager etc. still use set_page:
                     # self.text_selection_manager.set_page(page)
                     # self.form_filler.set_page(page)
                     # self.next_page_navigator.set_page(page)
                     # self.ad_click_manager.set_page(page)
                 else:
                      raise Exception("Critical: Failed to obtain a valid browser page object.")

                 # --- Navigate ---
                 update_signal.emit(f"Bot {browser_id + 1}: Navigating...", browser_id)
                 self.browser_manager.navigate_to(url)
                 if not is_running_check(): break

                 # --- Core Actions ---
                 update_signal.emit(f"Bot {browser_id + 1}: Performing actions on page...", browser_id)
                 random_delay(0.5, 1.5) # Initial delay

                 if self.mouse_movement_enabled and self.text_selection_manager:
                      self.text_selection_manager.interact_with_page(browser_id, update_signal)
                      if not is_running_check(): break
                      random_delay()

                 if self.scrolling_manager:
                     if self.gemini_api_key:
                          self.scrolling_manager.gemini_scroll(browser_id, update_signal, is_running_check) # Pass check
                     else:
                          self.scrolling_manager.random_scroll()
                     if not is_running_check(): break
                     random_delay()

                 if self.form_fill_enabled and self.form_filler:
                      self.form_filler.fill_form(browser_id, update_signal)
                      if not is_running_check(): break
                      random_delay()

                 ad_clicked = False
                 if self.ad_click_enabled and self.ad_click_manager:
                      ad_clicked = self.ad_click_manager.click_ad(browser_id, update_signal)
                      if not is_running_check(): break
                      random_delay(0.5, 1.5) if not ad_clicked else random_delay(0.2, 0.8)

                 impression_handled = False
                 if self.impression_enabled and self.next_page_navigator:
                      impression_handled = self.next_page_navigator.navigate_next_page(browser_id, update_signal)
                      if impression_handled and self.scrolling_manager:
                          update_signal.emit(f"Bot {browser_id + 1}: Impression: Navigated. Scrolling new page.", browser_id)
                          self.scrolling_manager.random_scroll(min_scrolls=1, max_scrolls=3)
                      if not is_running_check(): break
                      random_delay()

                 if random.random() < 0.1:
                      self.browser_manager.take_screenshot(f"browser_{browser_id+1}_url_{i+1}_final")

                 update_signal.emit(f"Bot {browser_id + 1}: Finished actions for URL {i+1}.", browser_id)
                 random_delay(1.0, 2.5)

             except Exception as e:
                 # Catch errors specific to this URL processing
                 error_msg = f"Error processing URL {url} in Bot {browser_id + 1}: {e}"
                 error_logger.exception(f"Error processing URL {url} in Bot {browser_id + 1}:")
                 update_signal.emit(error_msg, browser_id)
                 # Take screenshot on error if possible
                 if self.browser_manager:
                     self.browser_manager.take_screenshot(f"browser_{browser_id+1}_url_{i+1}_error")
                 update_signal.emit(f"Bot {browser_id + 1}: Continuing to next URL after error.", browser_id)
                 random_delay(2, 4)

             # No finally block needed here for resource cleanup, handled by thread's finally

        # --- End of URL loop ---
        if is_running_check():
             update_signal.emit(f"Bot {browser_id + 1}: Finished all URLs.", browser_id)
        else:
             update_signal.emit(f"Bot {browser_id + 1}: Stopped during URL processing.", browser_id)
        # Progress signal handled by BotThread finally block


# --- Main Window Class ---
class MainWindow(QWidget):
    """Main application window."""
    proxy_check_result_signal = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aladdin Traffic Bot")
        try:
            if os.path.exists(LOGO_PATH):
                 self.setWindowIcon(QIcon(LOGO_PATH))
            else:
                 logging.warning(f"Logo file not found at {LOGO_PATH}, using default icon.")
        except Exception as e:
            logging.warning(f"Could not load window icon '{LOGO_PATH}': {e}")

        self.setGeometry(100, 100, 1150, 800) # Adjusted size
        self.q_settings = QSettings("HamzaAkmal", "AladdinTrafficBot") # Use unique names
        self.license_manager = LicenseManager()
        self.bot_threads: Dict[int, BotThread] = {}
        self.browser_logs: Dict[int, QTextEdit] = {}
        self.browser_progress: Dict[int, QProgressBar] = {}
        self.proxy_validator = ProxyValidator()
        self.user_agent_manager: Optional[FingerprintManager] = None # Instantiated before starting runs
        self.total_runs_requested = 0
        self.runs_completed = 0
        self.active_threads_count = 0
        self.is_stopping = False # Flag to prevent relaunching groups during stop process
        self.current_proxy_config_for_run: Optional[Dict] = None # Store proxy config used for current run

        # Connect proxy validator signal to GUI update slot
        self.proxy_validator.check_complete_signal.connect(self.update_proxy_status_ui)

        self.setup_ui()
        self.load_state() # Load saved UI state (includes loading settings file)
        self.update_license_status_display() # Update based on loaded license key
        # Ensure proxy placeholder and fields reflect loaded settings
        self.update_proxy_placeholder(self.proxy_type_combo.currentText())
        self.toggle_proxy_fields(Qt.CheckState.Checked if self.proxy_enabled_check.isChecked() else Qt.CheckState.Unchecked)

        logging.info("Main window initialized.")

    # --- UI Setup Methods ---
    def setup_ui(self):
        """Creates the GUI layout and widgets."""
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        # Main Tab
        self.main_tab = QWidget()
        self.setup_main_tab()
        self.tabs.addTab(self.main_tab, "Main Control")
        # Settings Tab
        self.settings_tab = QWidget()
        self.setup_settings_tab()
        self.tabs.addTab(self.settings_tab, "Settings")
        # Proxy Tab
        self.proxy_tab = QWidget()
        self.setup_proxy_tab()
        self.tabs.addTab(self.proxy_tab, "Proxy Management")
        # License Tab
        self.license_tab = QWidget()
        self.setup_license_tab()
        self.tabs.addTab(self.license_tab, "License")
        main_layout.addWidget(self.tabs)

    def setup_main_tab(self):
        main_tab_layout = QVBoxLayout(self.main_tab)
        url_group = QGroupBox("Target URLs (one per line)")
        url_layout = QVBoxLayout(); self.url_text_edit = QTextEdit()
        self.url_text_edit.setPlaceholderText("https://example.com\nhttps://another-site.org")
        self.url_text_edit.setMinimumHeight(100); url_layout.addWidget(self.url_text_edit)
        url_group.setLayout(url_layout); main_tab_layout.addWidget(url_group, stretch=1)

        bottom_layout = QHBoxLayout()
        run_config_group = QGroupBox("Run Configuration"); run_config_layout = QFormLayout()
        self.total_runs_spinbox = QSpinBox(); self.total_runs_spinbox.setRange(1, 10000)
        self.total_runs_spinbox.setValue(settings.get("TOTAL_RUNS", 1))
        run_config_layout.addRow("Total Runs (Bots):", self.total_runs_spinbox)
        self.run_group_size_spinbox = QSpinBox(); self.run_group_size_spinbox.setRange(1, 50)
        self.run_group_size_spinbox.setValue(settings.get("RUN_GROUP_SIZE", 1))
        run_config_layout.addRow("Concurrent Bots:", self.run_group_size_spinbox)
        run_config_group.setLayout(run_config_layout); bottom_layout.addWidget(run_config_group)

        progress_controls_group = QGroupBox("Bot Control"); pc_layout = QVBoxLayout()
        progress_layout = QHBoxLayout(); progress_layout.addWidget(QLabel("Overall Progress:"))
        self.global_progress_bar = QProgressBar(); self.global_progress_bar.setTextVisible(True)
        self.global_progress_bar.setValue(0); progress_layout.addWidget(self.global_progress_bar)
        pc_layout.addLayout(progress_layout)
        self.global_status_label = QLabel("Status: Idle"); self.global_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pc_layout.addWidget(self.global_status_label)
        controls_layout = QHBoxLayout()
        icon_start = QIcon.fromTheme("media-playback-start", QIcon(f"{LOGO_PATH}")) if os.path.exists(LOGO_PATH) else QIcon.fromTheme("media-playback-start")
        self.start_button = QPushButton(icon_start, " Start Bots"); self.start_button.clicked.connect(self.start_bots_grouped)
        self.start_button.setStyleSheet("padding: 5px;"); controls_layout.addWidget(self.start_button)
        self.stop_button = QPushButton(QIcon.fromTheme("media-playback-stop"), " Stop All"); self.stop_button.clicked.connect(self.stop_all_bots)
        self.stop_button.setEnabled(False); self.stop_button.setStyleSheet("padding: 5px;"); controls_layout.addWidget(self.stop_button)
        pc_layout.addLayout(controls_layout); progress_controls_group.setLayout(pc_layout)
        bottom_layout.addWidget(progress_controls_group)
        main_tab_layout.addLayout(bottom_layout)

        logs_group = QGroupBox("Individual Bot Logs & Progress"); logs_layout = QVBoxLayout()
        self.browser_logs_tab_widget = QTabWidget(); self.browser_logs_tab_widget.setTabsClosable(True)
        self.browser_logs_tab_widget.tabCloseRequested.connect(self.close_log_tab)
        logs_layout.addWidget(self.browser_logs_tab_widget); logs_group.setLayout(logs_layout)
        main_tab_layout.addWidget(logs_group, stretch=3)

    def setup_settings_tab(self):
        settings_tab_layout = QVBoxLayout(self.settings_tab)
        form_layout = QFormLayout(); form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        general_group = QGroupBox("General"); general_layout = QFormLayout()
        self.headless_check = QCheckBox("Run Headless (No Visible Browser)"); general_layout.addRow(self.headless_check)
        self.min_delay_spin = QDoubleSpinBox(); self.min_delay_spin.setRange(0.1, 60.0); self.min_delay_spin.setSingleStep(0.1); self.min_delay_spin.setSuffix(" s")
        general_layout.addRow("Min Action Delay:", self.min_delay_spin)
        self.max_delay_spin = QDoubleSpinBox(); self.max_delay_spin.setRange(0.2, 120.0); self.max_delay_spin.setSingleStep(0.1); self.max_delay_spin.setSuffix(" s")
        general_layout.addRow("Max Action Delay:", self.max_delay_spin)
        general_group.setLayout(general_layout); form_layout.addRow(general_group)

        features_group = QGroupBox("Features"); features_layout = QFormLayout()
        self.mouse_movement_check = QCheckBox("Simulate Mouse Movement"); features_layout.addRow(self.mouse_movement_check)
        self.form_fill_check = QCheckBox("Attempt Form Filling"); features_layout.addRow(self.form_fill_check)
        impression_layout = QHBoxLayout(); self.impression_enabled_check = QCheckBox("Enable Impression")
        self.impression_enabled_check.setToolTip("Automatically tries to find and click 'next page' links."); impression_layout.addWidget(self.impression_enabled_check)
        self.next_page_selector_input = QLineEdit(); self.next_page_selector_input.setPlaceholderText(DEFAULT_SETTINGS["NEXT_PAGE_SELECTOR"])
        impression_layout.addWidget(self.next_page_selector_input); features_layout.addRow("Next Page:", impression_layout)
        ad_click_layout = QHBoxLayout(); self.ad_click_enabled_check = QCheckBox("Enable Ad Click"); ad_click_layout.addWidget(self.ad_click_enabled_check)
        self.ad_selector_input = QLineEdit(); self.ad_selector_input.setPlaceholderText(DEFAULT_SETTINGS["AD_SELECTOR"]); ad_click_layout.addWidget(self.ad_selector_input)
        prob_layout = QHBoxLayout(); prob_layout.addWidget(QLabel("Prob:")); self.ad_click_probability_spin = QDoubleSpinBox()
        self.ad_click_probability_spin.setRange(0.0, 1.0); self.ad_click_probability_spin.setSingleStep(0.05); self.ad_click_probability_spin.setDecimals(2)
        prob_layout.addWidget(self.ad_click_probability_spin); ad_click_layout.addLayout(prob_layout); features_layout.addRow("Ad Click:", ad_click_layout)
        features_group.setLayout(features_layout); form_layout.addRow(features_group)

        gemini_group = QGroupBox("Gemini AI Enhancements"); gemini_layout = QFormLayout()
        self.gemini_api_key_input = QLineEdit(); self.gemini_api_key_input.setPlaceholderText("Enter Google AI Gemini API Key (Optional)")
        self.gemini_api_key_input.setEchoMode(QLineEdit.EchoMode.Password); gemini_layout.addRow("Gemini API Key:", self.gemini_api_key_input)
        self.user_agent_generation_check = QCheckBox("Use Gemini to Generate User Agents")
        self.user_agent_generation_check.setToolTip("If enabled and API key provided, generates more UAs when needed."); gemini_layout.addRow(self.user_agent_generation_check)
        self.user_agent_generation_count_spin = QSpinBox(); self.user_agent_generation_count_spin.setRange(5, 50); self.user_agent_generation_count_spin.setValue(DEFAULT_SETTINGS["USER_AGENT_GENERATION_COUNT"])
        gemini_layout.addRow("UAs to Generate:", self.user_agent_generation_count_spin); gemini_group.setLayout(gemini_layout); form_layout.addRow(gemini_group)

        chromium_blue_group = QGroupBox("Custom Browser Executable (Advanced)"); cb_layout = QFormLayout()
        self.chromium_blue_check = QCheckBox("Use Custom Chromium/Chrome Path"); self.chromium_blue_check.stateChanged.connect(self.toggle_chromium_blue_fields)
        cb_layout.addRow(self.chromium_blue_check); cb_path_layout = QHBoxLayout()
        self.chromium_blue_path_input = QLineEdit(); self.chromium_blue_path_input.setPlaceholderText("Path to chromium/chrome executable"); self.chromium_blue_path_input.setEnabled(False)
        cb_path_layout.addWidget(self.chromium_blue_path_input); cb_browse_button = QPushButton("Browse..."); cb_browse_button.clicked.connect(self.browse_chromium_path)
        cb_browse_button.setEnabled(False); self.chromium_blue_browse_button = cb_browse_button; cb_path_layout.addWidget(cb_browse_button)
        cb_layout.addRow("Executable Path:", cb_path_layout); self.chromium_blue_args_input = QLineEdit()
        self.chromium_blue_args_input.setPlaceholderText("--no-sandbox --disable-gpu (Space-separated args)"); self.chromium_blue_args_input.setEnabled(False)
        cb_layout.addRow("Launch Arguments:", self.chromium_blue_args_input); chromium_blue_group.setLayout(cb_layout); form_layout.addRow(chromium_blue_group)

        settings_tab_layout.addLayout(form_layout); settings_tab_layout.addStretch()
        settings_buttons_layout = QHBoxLayout(); settings_buttons_layout.addStretch()
        icon_save = QIcon.fromTheme("document-save", QIcon(f"{LOGO_PATH}")) if os.path.exists(LOGO_PATH) else QIcon.fromTheme("document-save")
        self.save_settings_button = QPushButton(icon_save, " Save Settings")
        self.save_settings_button.clicked.connect(self.save_current_settings_to_file)
        self.save_settings_button.setToolTip("Saves the current settings from all tabs to config/settings.py")
        self.save_settings_button.setMinimumWidth(150); self.save_settings_button.setStyleSheet("padding: 5px;"); settings_buttons_layout.addWidget(self.save_settings_button)
        settings_tab_layout.addLayout(settings_buttons_layout)

    def setup_proxy_tab(self):
        proxy_tab_layout = QVBoxLayout(self.proxy_tab)
        proxy_config_group = QGroupBox("Proxy Configuration"); proxy_config_layout = QFormLayout()
        self.proxy_enabled_check = QCheckBox("Enable Proxy"); self.proxy_enabled_check.stateChanged.connect(self.toggle_proxy_fields)
        proxy_config_layout.addRow(self.proxy_enabled_check); self.proxy_type_combo = QComboBox()
        self.proxy_type_combo.addItems(["socks5", "http", "https", "socks4"]); self.proxy_type_combo.setEnabled(False)
        self.proxy_type_combo.currentTextChanged.connect(self.update_proxy_placeholder); proxy_config_layout.addRow("Proxy Type:", self.proxy_type_combo)
        self.proxy_input = QLineEdit(); self.proxy_input.setEnabled(False); proxy_config_layout.addRow("Proxy String:", self.proxy_input)
        test_layout = QHBoxLayout()
        icon_test = QIcon.fromTheme("network-test", QIcon(f"{LOGO_PATH}")) if os.path.exists(LOGO_PATH) else QIcon.fromTheme("network-test")
        self.proxy_check_button = QPushButton(icon_test, " Test Connection"); self.proxy_check_button.clicked.connect(self.test_proxy_connection)
        self.proxy_check_button.setEnabled(False); self.proxy_check_button.setToolTip("Checks if the bot can connect using the provided proxy string.")
        test_layout.addWidget(self.proxy_check_button); self.proxy_status_label = QLabel("Status: Untested")
        self.proxy_status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.proxy_status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); test_layout.addWidget(self.proxy_status_label)
        proxy_config_layout.addRow(test_layout); proxy_config_group.setLayout(proxy_config_layout); proxy_tab_layout.addWidget(proxy_config_group)
        proxy_tab_layout.addStretch()

    def setup_license_tab(self):
        license_tab_layout = QVBoxLayout(self.license_tab)
        license_group = QGroupBox("License Activation"); license_layout = QFormLayout()
        self.license_key_input = QLineEdit(); self.license_key_input.setPlaceholderText("Enter your license key")
        license_layout.addRow("License Key:", self.license_key_input)
        activation_layout = QHBoxLayout()
        icon_activate = QIcon.fromTheme("input-checked", QIcon(f"{LOGO_PATH}")) if os.path.exists(LOGO_PATH) else QIcon.fromTheme("input-checked")
        self.activate_button = QPushButton(icon_activate, " Activate / Check"); self.activate_button.clicked.connect(self.activate_license)
        self.activate_button.setToolTip("Verifies the entered license key."); activation_layout.addWidget(self.activate_button)
        self.license_status_display_label = QLabel("Status: Unknown"); self.license_status_display_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.license_status_display_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); activation_layout.addWidget(self.license_status_display_label)
        license_layout.addRow(activation_layout); license_group.setLayout(license_layout); license_tab_layout.addWidget(license_group)
        license_tab_layout.addStretch()

    # --- UI Helper & Slot Functions ---
    def update_proxy_placeholder(self, proxy_type: str):
        proxy_type = proxy_type.lower()
        placeholder, tooltip = "", ""
        if proxy_type == "socks5":
            placeholder = "hostname:port OR user:pass@hostname:port"
            tooltip = "Format: proxy.example.com:1080\nOR\nuser:password@proxy.example.com:1080"
            # Consider also mentioning user:pass:host:port format if supported by parser
        elif proxy_type == "socks4":
             placeholder = "hostname:port OR userid@hostname:port"
             tooltip = "Format: proxy.example.com:1080\nOR (for SOCKS4a): your_user_id@proxy.example.com:1080"
        elif proxy_type in ["http", "https"]:
            placeholder = "hostname:port OR user:pass@hostname:port"
            tooltip = f"Format: proxy.example.com:8080\nOR (with auth): YourUsername:YourPassword@proxy.example.com:8080"
        self.proxy_input.setPlaceholderText(placeholder)
        self.proxy_input.setToolTip(tooltip)

    def toggle_proxy_fields(self, state):
        checked = (state == Qt.CheckState.Checked.value or state == Qt.CheckState.Checked) # Handle both int and enum
        self.proxy_type_combo.setEnabled(checked)
        self.proxy_input.setEnabled(checked)
        self.proxy_check_button.setEnabled(checked)
        if not checked:
             self.proxy_status_label.setText("Status: Disabled")
             self.proxy_status_label.setStyleSheet("") # Reset color

    def toggle_chromium_blue_fields(self, state):
        checked = (state == Qt.CheckState.Checked.value or state == Qt.CheckState.Checked)
        self.chromium_blue_path_input.setEnabled(checked)
        self.chromium_blue_args_input.setEnabled(checked)
        self.chromium_blue_browse_button.setEnabled(checked)

    def browse_chromium_path(self):
        start_dir = self.q_settings.value("last_browse_dir", os.path.expanduser("~"))
        filters = "Executables (*.exe);;Applications (*.app);;All Files (*)" # Combined filters
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Chromium/Chrome Executable", start_dir, filters)
        if filepath:
            self.chromium_blue_path_input.setText(filepath)
            self.q_settings.setValue("last_browse_dir", os.path.dirname(filepath))

    def update_license_status_display(self):
        is_valid, message = self.license_manager.check_license_status()
        self.license_status_display_label.setText(f"Status: {message}")
        if is_valid: self.license_status_display_label.setStyleSheet("color: #2ecc71; font-weight: bold;") # Green
        elif "expired" in message.lower(): self.license_status_display_label.setStyleSheet("color: #f39c12; font-weight: bold;") # Orange
        else: self.license_status_display_label.setStyleSheet("color: #e74c3c; font-weight: bold;") # Red

    def activate_license(self):
        license_key = self.license_key_input.text().strip()
        if not license_key:
             QMessageBox.warning(self, "Activation Error", "Please enter a license key to activate or check.")
             return
        is_valid, message = self.license_manager.verify_license(license_key)
        if is_valid:
             QMessageBox.information(self, "License Status", f"License check successful.\nStatus: {message}")
             self.q_settings.setValue("license_key_input", license_key) # Save entered key to easily retrieve later if needed
             self.save_state() # Save QSettings including new expiry
             # Also save to settings file immediately?
             current_settings = self.get_current_settings_from_ui()
             current_settings["LICENSE_KEY"] = license_key # Ensure it's updated
             save_settings_to_file(current_settings)
             global settings; settings["LICENSE_KEY"] = license_key # Update global dict too
        else:
             QMessageBox.critical(self, "License Status", f"License check failed.\nStatus: {message}")
        self.update_license_status_display() # Update UI

    def test_proxy_connection(self):
        if not self.proxy_enabled_check.isChecked():
             self.update_proxy_status_ui(False, "Status: Disabled")
             return
        proxy_type = self.proxy_type_combo.currentText()
        proxy_string = self.proxy_input.text().strip()
        if not self.proxy_validator.validate_proxy_format(proxy_string, proxy_type):
            self.update_proxy_status_ui(False, "Status: Invalid Format")
            QMessageBox.warning(self, "Proxy Error", f"Invalid proxy format for {proxy_type.upper()}.\nExpected format:\n{self.proxy_input.toolTip()}")
            return
        proxy_info = self.proxy_validator.parse_proxy_string(proxy_string, proxy_type)
        if not proxy_info:
             self.update_proxy_status_ui(False, "Status: Parsing Error")
             QMessageBox.critical(self, "Proxy Error", "Could not parse the proxy string. Check format.")
             return

        self.proxy_status_label.setText("Status: Testing...")
        self.proxy_status_label.setStyleSheet("color: #e67e22;") # Orange
        self.proxy_check_button.setEnabled(False); QApplication.processEvents()
        self.proxy_validator.run_check_in_thread(proxy_info)

    def update_proxy_status_ui(self, is_connected: bool, message: str):
        logging.info(f"Proxy check result: Connected={is_connected}, Message={message}")
        self.proxy_status_label.setText(message)
        if "Successful" in message: self.proxy_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;") # Green
        elif "Failed" in message or "Error" in message or "Invalid" in message: self.proxy_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;") # Red
        elif "Disabled" in message: self.proxy_status_label.setStyleSheet("") # Default
        else: self.proxy_status_label.setStyleSheet("color: #e67e22;") # Orange (Testing)
        if self.proxy_enabled_check.isChecked(): self.proxy_check_button.setEnabled(True)

    # --- Bot Control ---
    def update_global_progress(self):
        if self.total_runs_requested > 0:
             progress = int((self.runs_completed / self.total_runs_requested) * 100) if self.total_runs_requested > 0 else 0
             self.global_progress_bar.setValue(progress)
             status_text = f"Status: Running ({self.active_threads_count} active)"
             status_text += f" - {self.runs_completed}/{self.total_runs_requested} completed"
             self.global_status_label.setText(status_text)
        else:
             self.global_progress_bar.setValue(0)
             self.global_status_label.setText(f"Status: Idle")

    def start_bots_grouped(self):
        if self.active_threads_count > 0:
            QMessageBox.warning(self, "Bots Running", "Bots are already running. Stop them before starting again.")
            return

        is_valid, license_message = self.license_manager.check_license_status()
        if not is_valid:
            QMessageBox.critical(self, "License Error", f"Cannot start bots.\n{license_message}\nPlease activate a valid license on the License tab.")
            self.tabs.setCurrentWidget(self.license_tab); return

        urls_text = self.url_text_edit.toPlainText()
        urls = [url.strip() for url in urls_text.splitlines() if url.strip() and url.startswith(('http://', 'https://'))]
        if not urls:
            QMessageBox.warning(self, "Input Error", "Please enter at least one valid URL (starting with http:// or https://)."); return

        self.total_runs_requested = self.total_runs_spinbox.value()
        run_group_size = self.run_group_size_spinbox.value()
        self.runs_completed = 0
        self.active_threads_count = 0
        self.is_stopping = False

        # Reload global settings from UI *before* starting any threads
        current_ui_settings = self.get_current_settings_from_ui()
        global settings; settings.update(current_ui_settings)
        logging.info("Applied current UI settings to global config for this run.")

        # Validate and Prepare Proxy Config for Playwright
        self.current_proxy_config_for_run = None # Reset stored config
        proxy_type = settings.get("PROXY_TYPE", "socks5")
        proxy_string = settings.get("PROXY_STRING", "")

        if settings.get("PROXY_ENABLED", False):
             if not proxy_string:
                 QMessageBox.critical(self, "Proxy Error", "Proxy enabled, but Proxy String is empty."); self.tabs.setCurrentWidget(self.proxy_tab); return
             if not self.proxy_validator.validate_proxy_format(proxy_string, proxy_type):
                 QMessageBox.critical(self, "Proxy Error", f"Invalid proxy format for {proxy_type.upper()}. Check Proxy tab.\nExpected: {self.proxy_input.toolTip()}"); self.tabs.setCurrentWidget(self.proxy_tab); return
             proxy_info = self.proxy_validator.parse_proxy_string(proxy_string, proxy_type)
             if not proxy_info:
                 QMessageBox.critical(self, "Proxy Error", "Could not parse proxy string."); self.tabs.setCurrentWidget(self.proxy_tab); return

             # Ask to test if not tested successfully
             if not self.proxy_status_label.text().startswith("Status: Connection Successful"):
                reply = QMessageBox.question(self, "Proxy Not Tested", "Proxy connection hasn't been successfully tested recently. Start anyway?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No: self.tabs.setCurrentWidget(self.proxy_tab); return

             # Prepare Playwright proxy dict
             pw_proxy_dict = {}
             pw_server = proxy_info.get('playwright_server')
             if not pw_server:
                  QMessageBox.critical(self, "Proxy Error", "Internal error: Could not determine Playwright server format."); return
             pw_proxy_dict['server'] = pw_server
             if 'username' in proxy_info: pw_proxy_dict['username'] = proxy_info['username']
             if 'password' in proxy_info: pw_proxy_dict['password'] = proxy_info['password']
             self.current_proxy_config_for_run = pw_proxy_dict # Store for relaunching groups
             logging.info(f"Proxy Enabled: Using {proxy_type.upper()} via {proxy_info.get('hostname')}:{proxy_info.get('port')}")
        else:
             logging.info("Proxy Disabled.")

        # Initialize User Agent Manager for this batch run
        global user_agents, generated_user_agents
        user_agents = load_json_data(settings.get("USER_AGENTS_FILE"), "user_agents")
        generated_user_agents = load_json_data(settings.get("GENERATED_USER_AGENTS_FILE"), "generated_user_agents")
        self.user_agent_manager = FingerprintManager(
            user_agents, generated_user_agents, settings.get("GEMINI_API_KEY")
        )

        # UI State Update & Start Loop
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.browser_logs_tab_widget.clear(); self.browser_logs.clear()
        self.browser_progress.clear(); self.bot_threads.clear()
        self.global_status_label.setText("Status: Starting..."); self.update_global_progress(); QApplication.processEvents()

        # Start the first group
        self.launch_next_group(urls, run_group_size)

    def launch_next_group(self, urls, group_size):
        """Launches the next group of bot threads."""
        if self.is_stopping or self.runs_completed >= self.total_runs_requested:
            if self.active_threads_count == 0: self.finish_all_runs() # Ensure final state check
            return

        start_index = self.runs_completed + self.active_threads_count
        num_to_launch = min(group_size, self.total_runs_requested - start_index)

        if num_to_launch <= 0:
             if self.active_threads_count == 0: self.finish_all_runs()
             return

        self.global_status_label.setText(f"Status: Launching group ({start_index + 1} - {start_index + num_to_launch} / {self.total_runs_requested})...")
        logging.info(f"Launching next group of {num_to_launch} bot(s). Total active will be {self.active_threads_count + num_to_launch}.")

        if self.user_agent_manager: self.user_agent_manager.reset_used_agents()

        for i in range(num_to_launch):
             browser_id = start_index + i
             log_widget = QWidget(); log_layout = QVBoxLayout(log_widget); log_layout.setContentsMargins(2,2,2,2)
             progress_bar = QProgressBar(); progress_bar.setRange(0, 100); progress_bar.setValue(0); progress_bar.setTextVisible(True)
             progress_bar.setFormat(f"Bot {browser_id + 1}: %p%"); log_layout.addWidget(progress_bar)
             log_text_edit = QTextEdit(); log_text_edit.setReadOnly(True); log_text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
             log_layout.addWidget(log_text_edit)
             self.browser_logs[browser_id] = log_text_edit
             self.browser_progress[browser_id] = progress_bar
             tab_index = self.browser_logs_tab_widget.addTab(log_widget, f"Bot {browser_id + 1}")
             self.browser_logs_tab_widget.setCurrentIndex(tab_index)

             # Use current global settings + the prepared proxy config for this run
             bot_thread = BotThread(
                 browser_id=browser_id, urls=list(urls), proxy_config=self.current_proxy_config_for_run,
                 headless=settings.get("HEADLESS", True), use_chromium_blue=settings.get("CHROMIUM_BLUE_ENABLED", False),
                 chromium_blue_path=settings.get("CHROMIUM_BLUE_PATH", ""), chromium_blue_args=settings.get("CHROMIUM_BLUE_ARGS", ""),
                 gemini_api_key=settings.get("GEMINI_API_KEY"), user_agent_manager=self.user_agent_manager
             )
             bot_thread.update_signal.connect(self.update_log)
             bot_thread.progress_signal.connect(self.update_progress)
             bot_thread.finished_signal.connect(self.on_bot_finished)
             bot_thread.error_signal.connect(self.on_bot_error)

             self.bot_threads[browser_id] = bot_thread
             self.active_threads_count += 1
             bot_thread.start()
             logging.info(f"Bot thread {browser_id + 1} started (Active: {self.active_threads_count}).")
        self.update_global_progress()

    def stop_all_bots(self):
        if self.active_threads_count == 0 and not self.is_stopping:
            self.reset_ui_after_stop(); return
        if self.is_stopping: return

        self.is_stopping = True
        self.global_status_label.setText(f"Status: Stopping ({self.active_threads_count} active)...")
        logging.info(f"Stop requested for {self.active_threads_count} active bot threads.")
        self.stop_button.setEnabled(False); QApplication.processEvents()

        active_thread_ids = list(self.bot_threads.keys())
        if not active_thread_ids:
             self.reset_ui_after_stop(); return

        for browser_id in active_thread_ids:
             bot_thread = self.bot_threads.get(browser_id)
             if bot_thread and bot_thread.isRunning():
                 logging.info(f"Sending stop signal to Bot {browser_id + 1}")
                 bot_thread.stop()
             else:
                 # Clean up immediately if thread already finished/removed
                 if browser_id in self.bot_threads: del self.bot_threads[browser_id]
                 self.active_threads_count = max(0, self.active_threads_count - 1)

        self.update_global_progress()
        # Optional: Start a timer to check if threads actually stop
        # QTimer.singleShot(30000, self.check_force_stop)

    def reset_ui_after_stop(self):
         self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
         self.global_progress_bar.setValue(0); self.global_status_label.setText("Status: Idle")
         self.active_threads_count = 0; self.runs_completed = 0; self.total_runs_requested = 0
         self.is_stopping = False; self.bot_threads.clear()
         self.current_proxy_config_for_run = None # Clear stored proxy
         logging.info("All bot operations stopped/completed. UI reset to Idle.")

    def update_log(self, message: str, browser_id: int):
        if browser_id in self.browser_logs:
            log_text_edit = self.browser_logs[browser_id]
            timestamp = QTime.currentTime().toString("hh:mm:ss.zzz")
            log_text_edit.append(f"[{timestamp}] {message}")
            # Scroll to bottom
            scrollbar = log_text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum()) # Force scroll down
        else:
            logging.warning(f"(Log Tab Missing for Bot {browser_id+1}) {message}")

    def update_progress(self, value: int, browser_id: int):
        if browser_id in self.browser_progress:
            progress_bar = self.browser_progress[browser_id]
            progress_bar.setValue(value)
            current_format = progress_bar.format()
            # Update format based on value and potential error state (indicated by style)
            if value == 100:
                 style = progress_bar.styleSheet()
                 if "e74c3c" in style: # Check if red (error color)
                     progress_bar.setFormat(f"Bot {browser_id + 1}: Error")
                 else:
                     progress_bar.setFormat(f"Bot {browser_id + 1}: Done")
            elif not current_format.startswith(f"Bot {browser_id + 1}: %p%"): # Reset format if needed
                 progress_bar.setFormat(f"Bot {browser_id + 1}: %p%")

    def on_bot_finished(self, browser_id: int):
        log_msg_prefix = f"Bot {browser_id + 1}: "
        was_stopped = False
        if browser_id in self.bot_threads:
             if hasattr(self.bot_threads[browser_id], '_is_running') and not self.bot_threads[browser_id]._is_running:
                 was_stopped = True
             del self.bot_threads[browser_id] # Remove thread reference

        self.active_threads_count = max(0, self.active_threads_count - 1) # Decrement safely
        if not was_stopped:
            self.runs_completed += 1 # Only increment if finished naturally
            log_msg = log_msg_prefix + "Task Completed."
        else:
            log_msg = log_msg_prefix + "Stopped."

        logging.info(f"Bot thread {browser_id + 1} finished/stopped. Active: {self.active_threads_count}, Completed: {self.runs_completed}/{self.total_runs_requested}")
        self.update_log(log_msg, browser_id)

        if browser_id in self.browser_progress:
             self.browser_progress[browser_id].setValue(100)
             if "e74c3c" not in self.browser_progress[browser_id].styleSheet(): # Don't override error style
                 self.browser_progress[browser_id].setStyleSheet("QProgressBar::chunk { background-color: #2ecc71; }") # Green chunk
                 self.browser_progress[browser_id].setFormat(f"Bot {browser_id + 1}: Done")

        self.update_global_progress()

        # Launch next group if needed and not stopping
        if not self.is_stopping and self.runs_completed + self.active_threads_count < self.total_runs_requested:
             urls_text = self.url_text_edit.toPlainText()
             urls = [url.strip() for url in urls_text.splitlines() if url.strip() and url.startswith(('http://', 'https://'))]
             run_group_size = self.run_group_size_spinbox.value()
             # Relaunch if active count is less than group size
             if self.active_threads_count < run_group_size:
                 self.launch_next_group(urls, run_group_size)
             # else: group size limit met, wait for another thread to finish

        # Check if all runs are completed or accounted for
        elif self.active_threads_count == 0:
             self.finish_all_runs()

    def on_bot_error(self, error_message: str, browser_id: int):
        self.update_log(f"Bot {browser_id + 1}: ERROR - {error_message}", browser_id)
        if browser_id in self.browser_progress:
             self.browser_progress[browser_id].setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }") # Red chunk
             self.browser_progress[browser_id].setValue(100)
             self.browser_progress[browser_id].setFormat(f"Bot {browser_id + 1}: Error")

        if browser_id in self.bot_threads:
             del self.bot_threads[browser_id]
        self.active_threads_count = max(0, self.active_threads_count - 1)
        logging.warning(f"Bot thread {browser_id + 1} reported error. Active: {self.active_threads_count}")

        # Treat error as completed attempt for counting purposes
        self.runs_completed += 1
        self.update_global_progress()

        # Check if this error means all runs are accounted for or launch next
        if not self.is_stopping and self.runs_completed + self.active_threads_count < self.total_runs_requested:
            urls_text = self.url_text_edit.toPlainText()
            urls = [url.strip() for url in urls_text.splitlines() if url.strip() and url.startswith(('http://', 'https://'))]
            run_group_size = self.run_group_size_spinbox.value()
            if self.active_threads_count < run_group_size:
                self.launch_next_group(urls, run_group_size)
        elif self.active_threads_count == 0:
            self.finish_all_runs()

    def finish_all_runs(self):
         """Called when all requested bot runs are completed or stopped."""
         # Only show message and reset if not already idle
         if not self.start_button.isEnabled():
             if self.runs_completed >= self.total_runs_requested:
                 QMessageBox.information(self, "Run Complete", f"All {self.total_runs_requested} bot runs have finished.")
             else: # Must have been stopped early
                 QMessageBox.warning(self, "Run Stopped", f"Bot run stopped. {self.runs_completed}/{self.total_runs_requested} runs attempted.")
             self.reset_ui_after_stop()

    def close_log_tab(self, index):
        widget = self.browser_logs_tab_widget.widget(index)
        if not widget: return

        browser_id_to_remove = -1
        # Find ID by iterating through known progress bars/logs attached to the widget's layout
        for bid, progress_bar in self.browser_progress.items():
             if progress_bar.parentWidget() == widget:
                 browser_id_to_remove = bid
                 break

        if browser_id_to_remove != -1:
             logging.info(f"Closing log tab for Bot {browser_id_to_remove + 1}. Requesting stop if running.")
             # Request stop for the corresponding thread
             bot_thread = self.bot_threads.get(browser_id_to_remove)
             if bot_thread and bot_thread.isRunning():
                  bot_thread.stop()
             # Clean up UI elements immediately
             if browser_id_to_remove in self.browser_logs: del self.browser_logs[browser_id_to_remove]
             if browser_id_to_remove in self.browser_progress: del self.browser_progress[browser_id_to_remove]
             # Note: Thread cleanup (removing from self.bot_threads) happens in on_bot_finished/on_bot_error
        else:
             logging.warning(f"Could not find browser ID for tab index {index} to close.")

        self.browser_logs_tab_widget.removeTab(index)
        widget.deleteLater() # Schedule deletion

    # --- Settings Load/Save ---
    def get_current_settings_from_ui(self) -> Dict:
         """Reads current values from UI widgets and returns a settings dictionary."""
         s = {}
         # General
         s["HEADLESS"] = self.headless_check.isChecked()
         s["MIN_DELAY"] = self.min_delay_spin.value()
         s["MAX_DELAY"] = self.max_delay_spin.value()
         # Features
         s["MOUSE_MOVEMENT_ENABLED"] = self.mouse_movement_check.isChecked()
         s["FORM_FILL_ENABLED"] = self.form_fill_check.isChecked()
         s["IMPRESSION_ENABLED"] = self.impression_enabled_check.isChecked()
         s["NEXT_PAGE_SELECTOR"] = self.next_page_selector_input.text().strip() or DEFAULT_SETTINGS["NEXT_PAGE_SELECTOR"]
         s["AD_CLICK_ENABLED"] = self.ad_click_enabled_check.isChecked()
         s["AD_SELECTOR"] = self.ad_selector_input.text().strip() or DEFAULT_SETTINGS["AD_SELECTOR"]
         s["AD_CLICK_PROBABILITY"] = self.ad_click_probability_spin.value()
         # Gemini
         s["GEMINI_API_KEY"] = self.gemini_api_key_input.text().strip()
         s["USER_AGENT_GENERATION_ENABLED"] = self.user_agent_generation_check.isChecked()
         s["USER_AGENT_GENERATION_COUNT"] = self.user_agent_generation_count_spin.value()
         # Chromium Blue
         s["CHROMIUM_BLUE_ENABLED"] = self.chromium_blue_check.isChecked()
         s["CHROMIUM_BLUE_PATH"] = self.chromium_blue_path_input.text().strip()
         s["CHROMIUM_BLUE_ARGS"] = self.chromium_blue_args_input.text().strip()
         # Proxy
         s["PROXY_ENABLED"] = self.proxy_enabled_check.isChecked()
         s["PROXY_TYPE"] = self.proxy_type_combo.currentText()
         s["PROXY_STRING"] = self.proxy_input.text().strip()
         # Run Config
         s["TOTAL_RUNS"] = self.total_runs_spinbox.value()
         s["RUN_GROUP_SIZE"] = self.run_group_size_spinbox.value()
         # License Key
         s["LICENSE_KEY"] = self.license_key_input.text().strip()

         # Add non-UI settings from defaults to ensure they are saved
         for key in DEFAULT_SETTINGS:
             if key not in s:
                 s[key] = settings.get(key, DEFAULT_SETTINGS[key]) # Use current global or default

         return s

    def apply_settings_to_ui(self, s: Dict):
         """Applies loaded settings dictionary values to the UI widgets."""
         # General
         self.headless_check.setChecked(s.get("HEADLESS", DEFAULT_SETTINGS["HEADLESS"]))
         self.min_delay_spin.setValue(s.get("MIN_DELAY", DEFAULT_SETTINGS["MIN_DELAY"]))
         self.max_delay_spin.setValue(s.get("MAX_DELAY", DEFAULT_SETTINGS["MAX_DELAY"]))
         # Features
         self.mouse_movement_check.setChecked(s.get("MOUSE_MOVEMENT_ENABLED", DEFAULT_SETTINGS["MOUSE_MOVEMENT_ENABLED"]))
         self.form_fill_check.setChecked(s.get("FORM_FILL_ENABLED", DEFAULT_SETTINGS["FORM_FILL_ENABLED"]))
         self.impression_enabled_check.setChecked(s.get("IMPRESSION_ENABLED", DEFAULT_SETTINGS["IMPRESSION_ENABLED"]))
         self.next_page_selector_input.setText(s.get("NEXT_PAGE_SELECTOR", DEFAULT_SETTINGS["NEXT_PAGE_SELECTOR"]))
         self.ad_click_enabled_check.setChecked(s.get("AD_CLICK_ENABLED", DEFAULT_SETTINGS["AD_CLICK_ENABLED"]))
         self.ad_selector_input.setText(s.get("AD_SELECTOR", DEFAULT_SETTINGS["AD_SELECTOR"]))
         self.ad_click_probability_spin.setValue(s.get("AD_CLICK_PROBABILITY", DEFAULT_SETTINGS["AD_CLICK_PROBABILITY"]))
         # Gemini
         self.gemini_api_key_input.setText(s.get("GEMINI_API_KEY", DEFAULT_SETTINGS["GEMINI_API_KEY"]))
         self.user_agent_generation_check.setChecked(s.get("USER_AGENT_GENERATION_ENABLED", DEFAULT_SETTINGS["USER_AGENT_GENERATION_ENABLED"]))
         self.user_agent_generation_count_spin.setValue(s.get("USER_AGENT_GENERATION_COUNT", DEFAULT_SETTINGS["USER_AGENT_GENERATION_COUNT"]))
         # Chromium Blue
         self.chromium_blue_check.setChecked(s.get("CHROMIUM_BLUE_ENABLED", DEFAULT_SETTINGS["CHROMIUM_BLUE_ENABLED"]))
         self.chromium_blue_path_input.setText(s.get("CHROMIUM_BLUE_PATH", DEFAULT_SETTINGS["CHROMIUM_BLUE_PATH"]))
         self.chromium_blue_args_input.setText(s.get("CHROMIUM_BLUE_ARGS", DEFAULT_SETTINGS["CHROMIUM_BLUE_ARGS"]))
         self.toggle_chromium_blue_fields(Qt.CheckState.Checked if self.chromium_blue_check.isChecked() else Qt.CheckState.Unchecked)
         # Proxy
         self.proxy_enabled_check.setChecked(s.get("PROXY_ENABLED", DEFAULT_SETTINGS["PROXY_ENABLED"]))
         self.proxy_type_combo.setCurrentText(s.get("PROXY_TYPE", DEFAULT_SETTINGS["PROXY_TYPE"]))
         self.proxy_input.setText(s.get("PROXY_STRING", DEFAULT_SETTINGS["PROXY_STRING"]))
         self.toggle_proxy_fields(Qt.CheckState.Checked if self.proxy_enabled_check.isChecked() else Qt.CheckState.Unchecked)
         self.update_proxy_placeholder(self.proxy_type_combo.currentText())
         # Run Config
         self.total_runs_spinbox.setValue(s.get("TOTAL_RUNS", DEFAULT_SETTINGS["TOTAL_RUNS"]))
         self.run_group_size_spinbox.setValue(s.get("RUN_GROUP_SIZE", DEFAULT_SETTINGS["RUN_GROUP_SIZE"]))
         # License Key
         self.license_key_input.setText(s.get("LICENSE_KEY", DEFAULT_SETTINGS["LICENSE_KEY"]))
         # Clear proxy status on load
         self.proxy_status_label.setText("Status: Untested")
         self.proxy_status_label.setStyleSheet("")

    def save_current_settings_to_file(self):
        """Saves current settings from UI to config/settings.py"""
        logging.info("Saving settings from UI to config/settings.py...")
        current_settings = self.get_current_settings_from_ui()
        if save_settings_to_file(current_settings):
            QMessageBox.information(self, "Settings Saved", f"Settings saved successfully to config/settings.py")
            global settings; settings = load_settings() # Reload global settings after save
        # Error message box handled within save_settings_to_file

    def load_settings_from_file(self):
        """Loads settings from config/settings.py and updates the UI."""
        logging.info("Loading settings from config/settings.py and applying to UI...")
        global settings; settings = load_settings() # Update global settings dict
        self.apply_settings_to_ui(settings)
        # Don't show message box on initial load, only on manual action
        # Re-verify license based on loaded key
        self.license_manager.verify_license(settings.get("LICENSE_KEY", ""))
        self.update_license_status_display()

    # --- State Persistence (QSettings for UI elements not in config/settings.py) ---
    def save_state(self):
        """Saves volatile UI state (window geom, URLs, last license tried)."""
        logging.debug("Saving UI state using QSettings.")
        self.q_settings.setValue("geometry", self.saveGeometry())
        self.q_settings.setValue("urls", self.url_text_edit.toPlainText())
        # License state (only save if currently considered valid)
        if self.license_manager.is_valid and self.license_manager.expiration_date and self.license_manager.last_checked_key:
             self.q_settings.setValue("license_expiry_date", self.license_manager.expiration_date.isoformat())
             self.q_settings.setValue("license_last_key", self.license_manager.last_checked_key)
        else:
             self.q_settings.remove("license_expiry_date")
             self.q_settings.remove("license_last_key")
        self.q_settings.sync()

    def load_state(self):
        """Loads UI state (geom, URLs) and verifies license state."""
        logging.debug("Loading UI/License state.")
        # Load settings from file first to populate UI
        self.load_settings_from_file() # Calls apply_settings_to_ui

        # Restore geometry from QSettings
        geometry = self.q_settings.value("geometry")
        if geometry and isinstance(geometry, (bytes, bytearray)): # Check type
            try: self.restoreGeometry(geometry)
            except Exception as e: logging.warning(f"Could not restore window geometry: {e}")

        # Restore URLs from QSettings
        self.url_text_edit.setText(self.q_settings.value("urls", ""))

        # Load and verify license state from QSettings (overriding file state if valid)
        saved_expiry_str = self.q_settings.value("license_expiry_date")
        saved_key = self.q_settings.value("license_last_key")

        if saved_expiry_str and saved_key and saved_key == settings.get("LICENSE_KEY"): # Check if key matches file
             try:
                 expiry_date = datetime.date.fromisoformat(saved_expiry_str)
                 if datetime.date.today() <= expiry_date:
                      # Restore saved state if key still matches file and not expired
                      self.license_manager.verify_license(saved_key) # Re-verify to set internal state
                      self.license_manager.expiration_date = expiry_date # Ensure correct expiry
                      logging.info(f"Restored valid license state from QSettings (Key: ...{saved_key[-5:]}, Expires: {expiry_date})")
                 else:
                      logging.warning("Saved license state from QSettings has expired. Re-activation needed.")
                      self.license_manager.is_valid = False; self.license_manager.expiration_date = None
             except (ValueError, TypeError) as e:
                  logging.warning(f"Error parsing saved license expiry date: {e}. Resetting license state.")
                  self.license_manager.is_valid = False; self.license_manager.expiration_date = None
        else:
            # If no valid saved state, the state from load_settings_from_file() remains.
            logging.info("No valid license state found/restored from QSettings.")

        # Ensure UI reflects the final loaded/verified state
        self.update_license_status_display()

    def closeEvent(self, event):
        """Handles the window closing event."""
        logging.info("Close event triggered.")
        if self.active_threads_count > 0:
             reply = QMessageBox.question(self, "Bots Running",
                                          f"{self.active_threads_count} bot(s) are still running. Stopping them might take a moment.\nDo you want to exit anyway?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.No:
                  event.ignore(); return
             else: # User chose Yes
                  logging.info("Attempting to stop all bots before closing...")
                  self.stop_all_bots() # Request threads to stop
                  # Give threads a very short time - A better approach involves waiting properly
                  QApplication.processEvents() # Process stop signals quickly
                  # time.sleep(0.5) # Avoid sleep in GUI thread if possible

        self.save_state() # Save UI state regardless
        logging.info("Exiting application.")
        # Ensure log handlers are closed
        logging.shutdown()
        for handler in logging.root.handlers + error_logger.handlers:
            handler.close()
        event.accept()


# --- Main Execution ---
def check_playwright_install():
     """Checks if Playwright browsers are installed."""
     logging.info("Checking Playwright browser installation...")
     pw_instance = None
     try:
         pw_instance = sync_playwright().start()
         browser = pw_instance.chromium.launch(headless=True, timeout=15000) # Short timeout
         browser.close()
         pw_instance.stop()
         logging.info("Playwright browser check successful.")
         return True
     except PlaywrightError as e:
         error_msg = str(e)
         if "Executable doesn't exist" in error_msg or "Timed out" in error_msg:
             logging.error(f"Playwright browser check failed: {error_msg}")
             return False
         else:
             logging.warning(f"Playwright check encountered unexpected PlaywrightError (assuming installed): {e}")
             return True # Assume installed for other PW errors
     except Exception as e:
        logging.error(f"Unexpected error during Playwright check: {e}", exc_info=True)
        return False
     finally:
         if pw_instance:
             try: pw_instance.stop() # Ensure stop is called even on early exit
             except Exception: pass


def main():
    """Entry point for the GUI application."""
    logging.info("Application starting...")
    _ensure_structure()

    app = QApplication(sys.argv)
    # app.setStyle('Fusion') # Optional style

    if not check_playwright_install():
         msg = ("Playwright browsers not found, installation corrupted, or check timed out.\n\n"
                "Please ensure you have internet access and run:\n"
                "'playwright install'\n"
                "in your terminal/command prompt, then restart the application.")
         logging.critical(msg)
         QMessageBox.critical(None, "Playwright Installation Error", msg)
         sys.exit(1)

    try:
        window = MainWindow()
        window.show()
        logging.info("Main window shown.")
        exit_code = app.exec()
        logging.info(f"Application exiting with code {exit_code}.")
        sys.exit(exit_code)
    except Exception as e:
         logging.critical(f"Fatal error during application startup or execution: {e}", exc_info=True)
         QMessageBox.critical(None, "Fatal Error", f"Application failed to start or crashed:\n{e}")
         sys.exit(1)


if __name__ == "__main__":
    threading.current_thread().name = "MainGUI"
    main()