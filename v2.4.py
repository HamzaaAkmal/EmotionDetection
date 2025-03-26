import os
import sys
import random
import time
import logging
import json
import subprocess
import threading
import datetime
import re
from typing import Optional, List, Dict, Tuple, Union

# --- Dependencies ---
try:
    from playwright.sync_api import (
        sync_playwright, Browser, Page, BrowserContext, Playwright, Error as PlaywrightError, Locator, TimeoutError
    )
    import google.generativeai as genai
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QLineEdit, QPushButton, QTextEdit, QCheckBox, QSpinBox,
        QDoubleSpinBox, QFileDialog, QMessageBox, QTabWidget, QGroupBox, QFormLayout,
        QTableWidget, QTableWidgetItem, QComboBox, QListWidget, QListWidgetItem,
        QStyledItemDelegate, QStyle, QScrollArea,
    )
    from PyQt6.QtCore import QThread, pyqtSignal, QSettings, QDate, QTime, Qt, QSize, QTimer
    from PyQt6.QtGui import QIcon, QPixmap, QColor
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Please ensure PyQt6, playwright, and google-generativeai are installed.")
    print("You might need to run: pip install PyQt6 playwright google-generativeai")
    print("Also run: playwright install")
    sys.exit(1)

# --- Project Structure ---
def _ensure_structure():
    for dir_path in ("core", "config", "logs", "data", "tests", "resources"):
        os.makedirs(dir_path, exist_ok=True)
    for file_path in ("config/settings.py", "config/proxies.txt", "config/fingerprint_profiles.json",
                      "logs/bot_activity.log", "logs/errors.log",
                      "data/user_agents.json", "data/important_words.json", "data/generated_user_agents.json"):
        if not os.path.exists(file_path):
            # Ensure parent directory exists before creating file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                if file_path.endswith(".json"):
                    # Create default fingerprint file if missing
                    if file_path == "config/fingerprint_profiles.json":
                         default_fp_config = {
                            "profiles": [
                                {
                                "name": "Default Realistic Chrome Win10", "description": "Simulates Chrome on Windows 10",
                                "navigator": {"user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "vendor": "Google Inc.", "platform": "Win32", "hardwareConcurrency": [8, 12, 16], "deviceMemory": [4, 8], "languages": ["en-US", "en"],
                                            "plugins": [{"name": "Chrome PDF Plugin", "filename": "internal-pdf-viewer", "description": "Portable Document Format", "mimeTypes": [{"type": "application/pdf", "suffixes": "pdf", "description": "Portable Document Format"}, {"type": "application/x-google-chrome-pdf", "suffixes": "pdf", "description": "Portable Document Format"}]}, {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai", "description": ""}, {"name": "Native Client", "filename": "internal-nacl-plugin", "description": "", "mimeTypes": [{"type": "application/x-nacl", "suffixes": "", "description": "Native Client Executable"}, {"type": "application/x-pnacl", "suffixes": "", "description": "Portable Native Client Executable"}]}],
                                            "mimeTypes": [{"type": "application/pdf", "suffixes": "pdf", "description": "Portable Document Format"}, {"type": "application/x-google-chrome-pdf", "suffixes": "pdf", "description": "Portable Document Format"}, {"type": "application/x-nacl", "suffixes": "", "description": "Native Client Executable"}, {"type": "application/x-pnacl", "suffixes": "", "description": "Portable Native Client Executable"}]
                                            },
                                "screen": {"colorDepth": 24, "pixelDepth": 24},
                                "canvas": {"noise_level": 0.05}, # Added noise level
                                "webgl": {"vendor": "Google Inc. (NVIDIA)", "renderer": "ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)"},
                                "timezone": "America/New_York",
                                # "webrtc_public_ip_only": True # Note: Actual setting is via launch arg and master toggle, not profile
                                },
                                {
                                "name": "Realistic Firefox Win10", "description": "Simulates Firefox on Windows 10",
                                "navigator": {"user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0", "vendor": "", "platform": "Win32", "hardwareConcurrency": [8, 12], "deviceMemory": [8], "languages": ["en-GB", "en"], "plugins": [], "mimeTypes": []},
                                "screen": {"colorDepth": 24, "pixelDepth": 24}, "canvas": {"noise_level": 0.03}, "webgl": {"vendor": "Mozilla", "renderer": "WebGL Vendor"}, "timezone": "Europe/London",
                                }
                             ],
                             "selected_profile_name": "Default Realistic Chrome Win10" # Default profile selection stored here
                         }
                         json.dump(default_fp_config, f, indent=2)
                    elif file_path == "data/user_agents.json":
                         default_uas = {"user_agents": [
                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                              "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
                         ]}
                         json.dump(default_uas, f, indent=4)
                    elif file_path == "data/important_words.json":
                         default_words = {"important_words": ["Contact", "About", "Price", "Download", "Login", "Register", "Support", "Features", "Blog", "News"]}
                         json.dump(default_words, f, indent=4)
                    elif file_path == "data/generated_user_agents.json":
                         # Start with empty list for generated UAs
                         json.dump({"generated_user_agents": []}, f, indent=4)
                    else:
                         f.write("{}") # Empty JSON for others
                elif file_path == "config/settings.py":
                    # Write default settings dictionary to settings.py
                     f.write("# Aladdin Traffic Bot Settings\n")
                     f.write(f"# Default file created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                     # Use the DEFAULT_SETTINGS dict defined below
                     try:
                         for key, value in DEFAULT_SETTINGS.items():
                              f.write(f"{key} = {repr(value)}\n")
                     except NameError:
                         # If DEFAULT_SETTINGS is not defined yet, write a placeholder
                         f.write("# DEFAULT_SETTINGS not defined at time of file creation.\n")
                else:
                    f.write("") # Empty file for others like proxies.txt


# --- Configuration ---
# Moved DEFAULT_SETTINGS definition before _ensure_structure call
DEFAULT_SETTINGS = {
    # --- Core & Control ---
    "TOTAL_RUNS": 1,
    "CONCURRENT_BROWSERS": 1, # Recommendation: Keep at 1 for simplicity with GUI/Proxies unless using many valid proxies
    "RUN_GROUP_SIZE": 1, # How many browsers run concurrently per group
    "LICENSE_KEY": "", # Store last entered key
    "MIN_DELAY": 2.0,
    "MAX_DELAY": 5.0,
    # --- Proxy ---
    "PROXY_ENABLED": False,
    "PROXY_FILE": "config/proxies.txt",
    # --- Browser & Launch ---
    "HEADLESS": True,
    "CHROMIUM_BLUE_ENABLED": False,
    "CHROMIUM_BLUE_PATH": "",
    "CHROMIUM_BLUE_ARGS": "",
    # --- Fingerprinting & Evasion ---
    "FINGERPRINT_PROFILE_NAME": "Default Realistic Chrome Win10", # Name from json, "Random", or "Generate & Use New UA"
    "DISABLE_AUTOMATION_FLAGS": True,
    "PREVENT_WEBRTC_IP_LEAK": True, # <<< Master switch for WebRTC leak prevention
    "USER_AGENT_SOURCE": "Combined", # Future use? "Static", "Generated", "Combined"
    "FINGERPRINT_FILE": "config/fingerprint_profiles.json",
    "VIEWPORT_MIN_WIDTH": 1100,
    "VIEWPORT_MAX_WIDTH": 1920,
    "VIEWPORT_MIN_HEIGHT": 800,
    "VIEWPORT_MAX_HEIGHT": 1080,
    # --- User Agent Generation ---
    "GEMINI_API_KEY": "",
    "USER_AGENT_GENERATION_ENABLED": False, # Enable *automatic* generation when needed
    "USER_AGENT_GENERATION_COUNT": 10, # How many to generate *automatically*
    "USER_AGENTS_FILE": "data/user_agents.json",
    "GENERATED_USER_AGENTS_FILE": "data/generated_user_agents.json",
    # --- Behavior ---
    "MOUSE_MOVEMENT_ENABLED": True,
    "SCROLL_DURATION_MIN": 500, # For random_scroll animation
    "SCROLL_DURATION_MAX": 1500, # For random_scroll animation
    "ENABLE_BEHAVIORAL_STATES": True, # Scanning vs Reading states
    "SKIP_ACTION_PROBABILITY": 0.05, # Chance to randomly skip an action
    "IMPORTANT_WORDS_FILE": "data/important_words.json",
    # --- Interaction Features ---
    "FORM_FILL_ENABLED": False,
    "IMPRESSION_ENABLED": False, # Enable next page navigation
    "NEXT_PAGE_SELECTORS": [".next-page", ".next", "a[rel='next']", ".pagination .active + li a", "li.pagination-next a"],
    "NEXT_PAGE_TEXT_FALLBACK": ["Next", "Next Page", ">", ">>"],
    "AD_CLICK_ENABLED": False,
    "AD_SELECTORS": [".ad-link", ".advertisement", "[id*='ad']", "[class*='ad']", "ins.adsbygoogle", "iframe[id*='google_ads']"],
    "AD_CLICK_PROBABILITY": 0.1, # Chance to click an ad IF found
}

# Now call _ensure_structure after DEFAULT_SETTINGS is defined
_ensure_structure()

# --- Logo ---
if not os.path.exists("resources/chromium_blue.png"):
    try:
        from PIL import Image
        img = Image.new('RGB', (64, 64), color=(70, 130, 180)) # Steel Blue color
        os.makedirs("resources", exist_ok=True) # Ensure dir exists
        img.save('resources/chromium_blue.png')
    except Exception as e:
        print(f"Note: Error creating dummy logo: {e}. You can place 'chromium_blue.png' in 'resources'.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, # Initial level, can be changed later
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bot_activity.log"),
        logging.StreamHandler(sys.stdout), # Also print logs to console
    ],
)
error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler("logs/errors.log")
error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s") # More detail
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)


def load_settings() -> Dict:
    """Loads settings from config/settings.py, falling back to defaults."""
    settings_from_file = DEFAULT_SETTINGS.copy()
    filepath = "config/settings.py"
    try:
        if not os.path.exists(filepath):
             logging.warning(f"{filepath} not found. Using default settings.")
             return settings_from_file # Return defaults if file doesn't exist

        with open(filepath, "r") as f:
            settings_module = {}
            exec(f.read(), settings_module)
            # Update only keys that exist in defaults or are all uppercase
            for k, v in settings_module.items():
                 # Update only keys that exist in defaults
                 if k in settings_from_file:
                      settings_from_file[k] = v
                 elif k.isupper(): # Also load any other uppercase settings defined in the file
                      settings_from_file[k] = v
            logging.info(f"Loaded settings from {filepath}")
    except Exception as e:
        error_logger.warning(f"Error loading {filepath}: {e}. Using default/previous settings.")
        # Keep existing defaults if loading fails
    return settings_from_file

settings = load_settings()


# --- Load Data Files ---

def load_user_agents() -> List[str]:
    ua_file = settings.get("USER_AGENTS_FILE", "data/user_agents.json")
    try:
        with open(ua_file, "r") as f:
            data = json.load(f)
            agents = data.get("user_agents", [])
            if not isinstance(agents, list):
                error_logger.error(f"User agents data in {ua_file} is not a list. Returning [].")
                return []
            return [str(ua) for ua in agents if ua] # Ensure strings and filter empty
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        error_logger.error(f"Error loading user agents from {ua_file}: {e}. Returning [].")
        return []

def load_generated_user_agents() -> List[str]:
    filepath = settings.get("GENERATED_USER_AGENTS_FILE", "data/generated_user_agents.json")
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
             with open(filepath, "r") as f:
                 data = json.load(f)
                 agents = data.get("generated_user_agents", [])
                 if not isinstance(agents, list):
                    error_logger.error(f"Generated user agents data in {filepath} is not a list. Returning [].")
                    return []
                 return [str(ua) for ua in agents if ua] # Ensure strings and filter empty
        else:
            return [] # Return empty if file doesn't exist or is empty
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        error_logger.error(f"Error loading generated agents from {filepath}: {e}. Returning [].")
        return []

def save_generated_user_agents(user_agents_list: List[str]):
    filepath = settings.get("GENERATED_USER_AGENTS_FILE", "data/generated_user_agents.json")
    try:
        # Ensure list contains only unique strings
        unique_agents = sorted(list(set(filter(None, [str(ua) for ua in user_agents_list]))))
        with open(filepath, "w") as f:
            json.dump({"generated_user_agents": unique_agents}, f, indent=4)
        logging.info(f"Saved {len(unique_agents)} unique generated user agents to {filepath}")
    except Exception as e:
        error_logger.error(f"Error saving generated user agents to {filepath}: {e}")

def load_important_words() -> List[str]:
    words_file = settings.get("IMPORTANT_WORDS_FILE", "data/important_words.json")
    try:
        with open(words_file, "r") as f:
             data = json.load(f)
             words = data.get("important_words", [])
             if not isinstance(words, list):
                 error_logger.error(f"Important words data in {words_file} is not a list. Returning [].")
                 return []
             return [str(w) for w in words if w] # Ensure strings and filter empty
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        error_logger.error(f"Error loading important words from {words_file}: {e}. Returning [].")
        return []

def load_proxies() -> List[Tuple[str, str]]:
    """Loads proxies: [(address, type), ...]. Type is extracted from the line."""
    proxies: List[Tuple[str, str]] = []
    proxy_file = settings.get("PROXY_FILE", "config/proxies.txt")
    try:
        with open(proxy_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue # Skip empty lines and comments

                parts = line.split()
                if not parts: continue

                address = parts[0]
                proxy_type = "http" # Default type

                # Basic HOST:PORT validation (allows hostnames and user:pass@host:port)
                # Improved regex to handle user:pass@hostname:port
                if not re.match(r"^(?:[^:@/]+(?::[^:@/]+)?@)?[\w.-]+:\d{1,5}$", address):
                    error_logger.warning(f"Invalid proxy address format on line {line_num} in {proxy_file}: '{address}'. Skipping.")
                    continue

                # Check for optional type specification (allow more flexible spacing/delimiters)
                potential_type = ""
                if len(parts) > 1:
                    potential_type = parts[-1].lower() # Check last part for type
                    if potential_type in ("http", "https", "socks4", "socks5"):
                        proxy_type = potential_type
                    # No else needed, if not a type, ignore it

                # Special case: if address starts with scheme, use that scheme
                # BUT playwright proxy arg expects `http` for https proxy too!
                scheme_override = False
                if address.startswith("http://"):
                    proxy_type = "http"
                    address = address[len("http://"):]
                    scheme_override = True
                elif address.startswith("https://"):
                     address = address[len("https://"):]
                     proxy_type = "http" # Override back to http as required by Playwright proxy arg
                     scheme_override = True
                     logging.debug(f"Treating https:// proxy '{address}' as type 'http' for Playwright.")
                elif address.startswith("socks4://"):
                    proxy_type = "socks4"
                    address = address[len("socks4://"):]
                    scheme_override = True
                elif address.startswith("socks5://"):
                    proxy_type = "socks5"
                    address = address[len("socks5://"):]
                    scheme_override = True

                # Re-validate address format after potential scheme removal
                if scheme_override:
                    if not re.match(r"^(?:[^:@/]+(?::[^:@/]+)?@)?[\w.-]+:\d{1,5}$", address):
                         error_logger.warning(f"Invalid proxy address format after scheme processing on line {line_num}: '{address}'. Skipping.")
                         continue

                # Ensure the final determined type is valid
                if proxy_type not in ("http", "https", "socks4", "socks5"):
                    error_logger.warning(f"Invalid final proxy type '{proxy_type}' determined for line {line_num}: '{parts[0]}'. Defaulting to 'http'.")
                    proxy_type = "http" # Fallback to default if logic resulted in invalid type

                # Crucially, map 'https' type back to 'http' for Playwright server argument later
                # Keep the 'https' type if determined here for potential other uses, but BrowserManager will handle the mapping.
                # Let's store the *intended* type ('https' if specified) and let BrowserManager choose the scheme.
                # NO - let's simplify and store the type Playwright needs ('http' for https).
                effective_playwright_type = 'http' if proxy_type == 'https' else proxy_type

                proxies.append((address, effective_playwright_type)) # Store address and Playwright-compatible type

    except FileNotFoundError:
        error_logger.warning(f"Proxy file not found: {proxy_file}. No proxies loaded.")
    except Exception as e:
        error_logger.error(f"Error loading proxies from {proxy_file}: {e}")
    return proxies


# Load initial data
user_agents = load_user_agents()
generated_user_agents = load_generated_user_agents()
important_words = load_important_words()

# --- Core Classes ---

class FingerprintManager:
    """Manages Fingerprints (User Agents, Profiles), including generation."""

    def __init__(self, initial_user_agents: List[str], generated_user_agents: List[str], gemini_api_key: str):
        self.user_agents = initial_user_agents
        self.generated_user_agents = generated_user_agents
        self.gemini_api_key = gemini_api_key
        self.used_user_agents_in_run = set()
        self.fingerprint_profiles = self._load_fingerprint_profiles()

    def _load_fingerprint_profiles(self) -> List[Dict]:
        """Loads fingerprint profiles from JSON file."""
        filepath = settings.get("FINGERPRINT_FILE", "config/fingerprint_profiles.json")
        try:
            with open(filepath, "r") as f: data = json.load(f); profiles = data.get("profiles", [])
            if not isinstance(profiles, list): error_logger.error(f"{filepath} 'profiles' not a list."); return []
            valid_profiles = [p for p in profiles if isinstance(p, dict) and 'name' in p]
            if len(valid_profiles) != len(profiles): error_logger.warning(f"Some invalid profiles in {filepath}")
            return valid_profiles
        except Exception as e: error_logger.error(f"Error loading profiles from {filepath}: {e}"); return []

    def get_profile_names(self) -> List[str]:
        """Returns available profile names including special options."""
        names = ["Random", "Generate & Use New UA"] # Keep special options
        names.extend([p.get("name", f"Unnamed Profile {i+1}") for i, p in enumerate(self.fingerprint_profiles)])
        return names

    def get_random_viewport_size(self) -> Dict:
        w_min, w_max = settings.get("VIEWPORT_MIN_WIDTH", 1100), settings.get("VIEWPORT_MAX_WIDTH", 1920)
        h_min, h_max = settings.get("VIEWPORT_MIN_HEIGHT", 800), settings.get("VIEWPORT_MAX_HEIGHT", 1080)
        return {"width": random.randint(w_min, w_max), "height": random.randint(h_min, h_max)}

    def select_fingerprint_profile(self, profile_name_setting: str) -> Optional[Dict]:
        if not self.fingerprint_profiles: return None
        if profile_name_setting == "Random": return random.choice(self.fingerprint_profiles)
        for profile in self.fingerprint_profiles:
            if profile.get("name") == profile_name_setting: return profile
        if profile_name_setting not in ["Random", "Generate & Use New UA"] and self.fingerprint_profiles:
            error_logger.warning(f"Profile '{profile_name_setting}' not found. Using random.")
            return random.choice(self.fingerprint_profiles)
        elif self.fingerprint_profiles: # Fallback for "Generate & Use" if no others exist
             return self.fingerprint_profiles[0]
        else: error_logger.warning(f"Profile '{profile_name_setting}' requested but none loaded."); return None

    def get_fingerprint(self, browser_id, update_signal) -> Tuple[str, Dict, Optional[Dict]]:
        chosen_agent, profile = None, None
        profile_name_setting = settings.get("FINGERPRINT_PROFILE_NAME", "Random")

        if profile_name_setting == "Generate & Use New UA":
            update_signal.emit(f"Bot {browser_id+1}: Generating UA on-demand...", browser_id)
            generated_ua = self._generate_single_user_agent(browser_id, update_signal)
            if generated_ua:
                chosen_agent = generated_ua
                # Select a base profile to apply other settings from, or use random if default fails
                base_profile_name = settings.get("FINGERPRINT_PROFILE_NAME", "Default Realistic Chrome Win10") # Use default as base, not "Random"
                if base_profile_name in ["Random", "Generate & Use New UA"]: # If setting itself is dynamic, use a hardcoded default
                    base_profile_name = "Default Realistic Chrome Win10"
                profile = self.select_fingerprint_profile(base_profile_name) or self.select_fingerprint_profile("Random") # Fallback
                update_signal.emit(f"Bot {browser_id+1}: Using generated UA. Base profile: {profile.get('name', 'N/A') if profile else 'N/A'}", browser_id)
            else:
                update_signal.emit(f"Bot {browser_id+1}: On-demand UA gen failed. Falling back.", browser_id)
                profile_name_setting = "Random" # Fallback to random profile selection

        if not chosen_agent: # Standard selection or fallback from failed generation
            profile = self.select_fingerprint_profile(profile_name_setting)
            profile_ua = profile.get('navigator', {}).get('user_agent') if profile else None

            # Get a UA based on combined list or profile
            chosen_agent = self._get_user_agent_internal(browser_id, update_signal)

            # If a specific profile (not Random/Gen) was chosen AND it has a UA, prioritize THAT UA
            # unless chosen_agent already got one via explicit generation.
            if profile_ua and profile_name_setting not in ["Random", "Generate & Use New UA"]:
                 chosen_agent = profile_ua
                 update_signal.emit(f"Bot {browser_id+1}: Using UA from profile '{profile_name_setting}'.", browser_id)
                 self.used_user_agents_in_run.add(chosen_agent) # Ensure profile UA is marked as used

        viewport_size = self.get_random_viewport_size()
        if profile: update_signal.emit(f"Bot {browser_id+1}: Using FP Profile: {profile.get('name', 'Unnamed')}", browser_id)
        elif profile_name_setting != "Generate & Use New UA": update_signal.emit(f"Bot {browser_id+1}: Warning: No profile found/selected.", browser_id)

        # Final fallback if absolutely no UA was determined
        if not chosen_agent:
            chosen_agent = DEFAULT_SETTINGS['navigator']['user_agent'] # Use the default UA from the default profile
            update_signal.emit(f"Bot {browser_id+1}: CRITICAL: No UA determined. Using default: {chosen_agent[:60]}...", browser_id)

        return chosen_agent, viewport_size, profile

    def _get_user_agent_internal(self, browser_id, update_signal) -> str:
        all_uas = list(set(filter(None, self.user_agents + self.generated_user_agents)))
        available = [ua for ua in all_uas if ua not in self.used_user_agents_in_run]
        # Use the UA from the first default profile as the ultimate fallback
        default_ua = DEFAULT_SETTINGS.get("navigator", {}).get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")

        if not available:
            if settings.get("USER_AGENT_GENERATION_ENABLED", False):
                update_signal.emit(f"Bot {browser_id+1}: UAs exhausted. Auto-generating...", browser_id)
                newly_generated = self._generate_user_agents_gemini_auto(settings.get("USER_AGENT_GENERATION_COUNT", 10), browser_id, update_signal)
                if newly_generated:
                    self.generated_user_agents = load_generated_user_agents() # Reload updated list
                    all_uas = list(set(filter(None, self.user_agents + self.generated_user_agents)))
                    available = [ua for ua in all_uas if ua not in self.used_user_agents_in_run]
                    if not available: update_signal.emit(f"Bot {browser_id+1}: Warn: Gen OK but no unique UAs. Reusing.", browser_id)
                else: update_signal.emit(f"Bot {browser_id+1}: Warn: Auto-gen failed. Reusing.", browser_id)
            else: update_signal.emit(f"Bot {browser_id+1}: Warn: UAs exhausted, auto-gen disabled. Reusing.", browser_id)

            # After potential generation, if still no available UAs, reuse from all_uas or use default
            if all_uas:
                 chosen = random.choice(all_uas); update_signal.emit(f"Bot {browser_id+1}: Reusing UA: {chosen[:60]}...", browser_id); return chosen
            else: update_signal.emit(f"Bot {browser_id+1}: CRITICAL: No UAs available. Using default.", browser_id); return default_ua

        # If available UAs exist, choose one and mark as used
        chosen = random.choice(available); self.used_user_agents_in_run.add(chosen); return chosen

    def _call_gemini_api(self, prompt: str, api_key: str, browser_id: Optional[int] = None, update_signal: Optional[pyqtSignal] = None) -> Optional[str]:
        if not api_key: msg = "Gemini API key missing."; error_logger.error(msg); return None
        try:
            if update_signal and browser_id is not None: update_signal.emit(f"Bot {browser_id+1}: Calling Gemini...", browser_id)
            elif update_signal and browser_id is None: update_signal.emit("Calling Gemini for generation...", -1) # General message for manual gen
            logging.debug(f"Gemini Prompt: {prompt[:150]}...")
            genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-2.0-flash'); config = genai.types.GenerationConfig(temperature=0.8)
            response = model.generate_content(prompt, generation_config=config, request_options={'timeout': 45}) # Add timeout
            if not response.candidates or not response.text:
                 try: reason = response.prompt_feedback.block_reason; message = response.prompt_feedback.block_reason_message; msg = f"Gemini response blocked ({reason}: {message})."
                 except Exception: msg = "Gemini response empty/blocked."
                 error_logger.error(msg); return None
            logging.debug(f"Gemini Response: {response.text[:100]}..."); return response.text
        except Exception as e: error_msg = f"Gemini API Error: {e}"; error_logger.error(error_msg); return None

    def _parse_generated_uas(self, generated_text: str) -> List[str]:
        if not generated_text: return []
        # More robust parsing: handle potential surrounding text/markers
        uas = []
        lines = generated_text.strip().splitlines()
        for line in lines:
            line = line.strip()
            # Remove potential list markers or quotes
            if line.startswith(("* ", "- ", "`", '"')) and len(line) > 2: line = line[2:].strip()
            if line.endswith(("`", '"')): line = line[:-1].strip()
            # Basic validation
            if line.startswith('Mozilla/') and 50 < len(line) < 350: # Increased max length slightly
                 uas.append(line)
        return uas

    def _generate_user_agents_gemini_auto(self, count: int, browser_id, update_signal) -> List[str]:
        prompt = f"""Generate {count} diverse, realistic user agent strings (latest Chrome/FF/Safari/Edge on Win/Mac/Android/iOS). One per line. Output ONLY the strings, no other text or formatting."""
        api_key = settings.get("GEMINI_API_KEY")
        generated_text = self._call_gemini_api(prompt, api_key, browser_id, update_signal)
        if not generated_text: return []
        new_uas = self._parse_generated_uas(generated_text)
        if not new_uas: update_signal.emit(f"Bot {browser_id+1}: Gemini auto-gen parsing failed.", browser_id); return []
        if len(new_uas) < count: update_signal.emit(f"Bot {browser_id+1}: Auto-gen got {len(new_uas)}/{count} UAs.", browser_id)
        static = set(self.user_agents); generated = set(load_generated_user_agents())
        unique_new = [ua for ua in new_uas if ua not in static and ua not in generated]
        if unique_new:
            updated_list = sorted(list(generated.union(unique_new))); save_generated_user_agents(updated_list)
            update_signal.emit(f"Bot {browser_id+1}: Auto-generated and saved {len(unique_new)} new UAs.", browser_id); self.generated_user_agents = updated_list; return unique_new
        else: update_signal.emit(f"Bot {browser_id+1}: Auto-generated UAs were duplicates.", browser_id); return []

    def _generate_single_user_agent(self, browser_id, update_signal) -> Optional[str]:
         prompt = """Generate exactly one diverse, realistic user agent string (latest browser/OS). Output ONLY the string, nothing else."""
         api_key = settings.get("GEMINI_API_KEY")
         generated_text = self._call_gemini_api(prompt, api_key, browser_id, update_signal)
         if not generated_text: return None
         parsed_uas = self._parse_generated_uas(generated_text)
         if not parsed_uas: update_signal.emit(f"Bot {browser_id+1}: Single UA parsing failed.", browser_id); return None
         new_ua = parsed_uas[0]; update_signal.emit(f"Bot {browser_id+1}: Generated single UA: {new_ua[:60]}...", browser_id)
         current_gen = load_generated_user_agents()
         if new_ua not in current_gen and new_ua not in self.user_agents:
              updated = sorted(current_gen + [new_ua]); save_generated_user_agents(updated); self.generated_user_agents = updated
              update_signal.emit(f"Bot {browser_id+1}: Saved new unique UA.", browser_id)
         self.used_user_agents_in_run.add(new_ua); return new_ua

    def generate_user_agents_manual(self, count: int, api_key: str) -> Tuple[bool, str]:
        """Manually triggers generation of UAs and saves them."""
        if not api_key: return False, "Gemini API key missing."
        prompt = f"""Generate {count} diverse, realistic user agent strings (latest browsers/OS). One per line. Output ONLY the strings, no other text or formatting."""
        logging.info(f"Manual Gen: Requesting {count} UAs...")
        generated_text = self._call_gemini_api(prompt, api_key) # No browser_id/signal needed for manual
        if not generated_text: return False, "Failed to get response from Gemini."
        new_uas = self._parse_generated_uas(generated_text); num_gen = len(new_uas)
        if not new_uas: logging.warning(f"Manual Gen: Parsing failed. Snippet: {generated_text[:200]}..."); return False, "Response received, but no valid UAs parsed."
        logging.info(f"Manual Gen: Parsed {num_gen} potential UAs.")
        if num_gen < count: logging.warning(f"Manual Gen: Got {num_gen}/{count} UAs.")
        static = set(self.user_agents); current_gen = set(load_generated_user_agents())
        unique_new = [ua for ua in new_uas if ua not in static and ua not in current_gen]
        num_unique = len(unique_new)
        if num_unique > 0:
            updated = sorted(list(current_gen.union(unique_new))); save_generated_user_agents(updated); self.generated_user_agents = updated
            msg = f"Generated and saved {num_unique} new unique UAs." + (f" ({num_gen - num_unique} duplicates)" if num_gen > num_unique else "")
            logging.info(f"Manual Gen: {msg}"); return True, msg
        else: msg = f"Generated {num_gen} UAs, but all were duplicates of existing ones."; logging.warning(f"Manual Gen: {msg}"); return False, msg

    # --- Manual Fingerprint PROFILE Generation ---
    def generate_fingerprint_profile_manual(self, api_key: str) -> Tuple[bool, str]:
        """Manually generates a *full fingerprint profile* via Gemini and saves it."""
        if not api_key: return False, "Gemini API key missing."

        # Detailed prompt asking for a JSON profile object
        prompt = f"""
        Generate a single, complete, and realistic browser fingerprint profile as a JSON object.
        The goal is to simulate a common, modern browser setup (e.g., Chrome/Firefox/Edge/Safari on Windows/macOS/Linux/Android).
        The JSON object MUST have the following top-level keys: "name", "description", "navigator", "screen", "canvas", "webgl", "timezone".

        - "name": A short, descriptive name (e.g., "Chrome 123 Win11 HighEnd"). MUST be unique.
        - "description": A brief description of the simulated setup.
        - "navigator": An object with keys:
            - "user_agent": A realistic UA string (e.g., latest stable Chrome/Firefox/Edge/Safari on Win/Mac/Mobile).
            - "vendor": e.g., "Google Inc." for Chrome, "" for Firefox. MUST match browser in UA.
            - "platform": e.g., "Win32", "MacIntel", "Linux armv8l", "iPhone". MUST match OS in UA.
            - "hardwareConcurrency": An array of one or more plausible core counts, e.g., [8, 12, 16]. Pick realistic values.
            - "deviceMemory": An array of one or more plausible memory values, e.g., [8, 16]. Pick realistic values.
            - "languages": An array of language codes, e.g., ["en-US", "en"].
            - "plugins": An array of plugin objects (name, filename, description, mimeTypes array) OR an empty array []. Keep it simple and realistic (e.g., Chrome PDF plugin for Chrome, empty for Firefox). mimeTypes within should have type, suffixes, description.
            - "mimeTypes": An array of mimeType objects (type, suffixes, description) OR an empty array []. Should correspond to plugins.
        - "screen": An object with "colorDepth" and "pixelDepth" (usually 24).
        - "canvas": An object with "noise_level" (a small float between 0.01 and 0.08).
        - "webgl": An object with "vendor" and "renderer". MUST be consistent with the browser/OS/GPU hints (e.g., Google Inc./ANGLE for Chrome, Mozilla/Generic for Firefox, Apple/Apple GPU for Safari). Provide realistic examples like "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 Direct3D11 vs_5_0 ps_5_0, D3D11)".
        - "timezone": A valid IANA timezone ID, e.g., "America/New_York", "Europe/London", "Asia/Tokyo".

        Constraints:
        - Generate only ONE JSON object.
        - Ensure consistency between UA, vendor, platform, plugins, and WebGL info.
        - Use realistic values for versions, hardware, etc. Avoid placeholders like 'Example'.
        - Output ONLY the JSON object, starting with {{ and ending with }}. Do NOT include explanations, ```json markers, backticks, or any other text.

        Example Structure (Do not copy values directly, generate new realistic ones):
        {{
          "name": "Generated Firefox 122 Win10",
          "description": "Simulated Firefox 122 on Windows 10 Mid-Range",
          "navigator": {{
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/122.0",
            "vendor": "",
            "platform": "Win32",
            "hardwareConcurrency": [8],
            "deviceMemory": [8],
            "languages": ["en-GB", "en"],
            "plugins": [],
            "mimeTypes": []
          }},
          "screen": {{ "colorDepth": 24, "pixelDepth": 24 }},
          "canvas": {{ "noise_level": 0.03 }},
          "webgl": {{ "vendor": "Mozilla", "renderer": "ANGLE (AMD, Radeon RX 6800 XT Direct3D11 vs_5_0 ps_5_0, D3D11)" }},
          "timezone": "Europe/Paris"
        }}
        """
        logging.info("Manual FP Profile Gen: Requesting profile from Gemini...")
        generated_text = self._call_gemini_api(prompt, api_key) # No browser_id/signal needed
        if not generated_text: return False, "Failed to get response from Gemini."

        # Parse and Validate
        new_profile, validation_error = self._parse_and_validate_profile_json(generated_text)
        if not new_profile:
            logging.warning(f"Manual FP Profile Gen: Parsing/validation failed. Error: {validation_error}. Snippet: {generated_text[:300]}...")
            return False, f"Response received, but failed validation: {validation_error}\n\n(Make sure API key is valid and Gemini can fulfill the request)."

        # Save the validated profile
        filepath = settings.get("FINGERPRINT_FILE", "config/fingerprint_profiles.json")
        try:
            # Load existing profiles safely
            existing_profiles = []
            selected_profile_name = None
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                try:
                    with open(filepath, "r") as f:
                        config_data = json.load(f)
                        existing_profiles = config_data.get("profiles", [])
                        selected_profile_name = config_data.get("selected_profile_name") # Preserve selection
                        if not isinstance(existing_profiles, list): existing_profiles = []
                except json.JSONDecodeError:
                     logging.error(f"Error decoding existing profiles file {filepath}. Will overwrite with new profile.")
                     existing_profiles = [] # Start fresh if file is corrupt
            existing_names = {p.get("name") for p in existing_profiles if p.get("name")}

            # Ensure unique name
            original_name = new_profile["name"]
            counter = 1
            while new_profile["name"] in existing_names:
                new_profile["name"] = f"{original_name}_{counter}"
                counter += 1
            if new_profile["name"] != original_name:
                logging.warning(f"Manual FP Profile Gen: Renamed generated profile to '{new_profile['name']}' to avoid duplication.")

            # Append and save
            existing_profiles.append(new_profile)
            config_data_to_save = {
                "profiles": existing_profiles,
                "selected_profile_name": selected_profile_name or new_profile["name"] # Update selection if needed
            }
            with open(filepath, "w") as f:
                json.dump(config_data_to_save, f, indent=2)

            self.fingerprint_profiles = existing_profiles # Update internal list
            msg = f"Successfully generated and saved new profile: '{new_profile['name']}' to {filepath}"
            logging.info(f"Manual FP Profile Gen: {msg}")
            return True, msg

        except Exception as e:
            error_msg = f"Failed to save generated profile to {filepath}: {e}"
            error_logger.exception("Manual FP Profile Gen: Save Error")
            return False, error_msg

    def _parse_and_validate_profile_json(self, json_text: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Attempts to parse JSON text into a profile dict and performs basic validation."""
        try:
            # Clean potential markdown/extra text - find first '{' and last '}'
            start_index = json_text.find('{')
            end_index = json_text.rfind('}')
            if start_index == -1 or end_index == -1 or start_index >= end_index:
                return None, "Could not find valid JSON object boundaries {} in the response."
            json_cleaned = json_text[start_index : end_index + 1]

            profile = json.loads(json_cleaned)
            if not isinstance(profile, dict): return None, "Generated data is not a JSON object."

            # Basic structure validation
            required_keys = ["name", "description", "navigator", "screen", "canvas", "webgl", "timezone"]
            for key in required_keys:
                if key not in profile: return None, f"Missing required top-level key: '{key}'"

            nav = profile["navigator"]
            if not isinstance(nav, dict): return None, "'navigator' key is not an object."
            nav_keys = ["user_agent", "vendor", "platform", "hardwareConcurrency", "deviceMemory", "languages", "plugins", "mimeTypes"]
            for key in nav_keys:
                if key not in nav: return None, f"Missing required key in 'navigator': '{key}'"

            # Type checks for crucial fields
            if not isinstance(profile.get("name"), str) or not profile["name"].strip(): return None, "Profile 'name' is missing or empty."
            if not isinstance(nav.get("user_agent"), str) or not nav["user_agent"].strip(): return None, "Profile 'navigator.user_agent' is missing or empty."
            if not isinstance(nav.get("hardwareConcurrency"), list): return None, "Profile 'navigator.hardwareConcurrency' must be a list."
            if not isinstance(nav.get("deviceMemory"), list): return None, "Profile 'navigator.deviceMemory' must be a list."
            if not isinstance(nav.get("languages"), list): return None, "Profile 'navigator.languages' must be a list."
            if not isinstance(profile.get("webgl"), dict): return None, "'webgl' key is not an object."
            if not isinstance(profile.get("canvas"), dict): return None, "'canvas' key is not an object."
            if not isinstance(profile.get("screen"), dict): return None, "'screen' key is not an object."
            if not isinstance(profile.get("timezone"), str) or not profile["timezone"]: return None, "'timezone' must be a non-empty string."

            # Could add more checks for consistency (UA vs platform/vendor) but keep basic for now
            return profile, None # Validation passed

        except json.JSONDecodeError as e:
            return None, f"Invalid JSON format: {e}. Response snippet:\n{json_text[:300]}"
        except Exception as e:
            return None, f"Unexpected validation error: {e}"


class BrowserManager:
    """Manages Chromium/Chromium Blue browser instances with fingerprinting."""

    def __init__(self, playwright: Playwright, proxy_server: Optional[str] = None,
                 proxy_type: Optional[str] = None, headless: bool = True,
                 use_chromium_blue: bool = False, chromium_blue_path: str = "",
                 chromium_blue_args: str = ""):
        self.playwright = playwright
        self.proxy_server = proxy_server
        # proxy_type should already be mapped ('http' for https) from load_proxies
        self.proxy_type = proxy_type.lower() if proxy_type else None
        self.headless = headless
        self.use_chromium_blue = use_chromium_blue
        self.chromium_blue_path = chromium_blue_path
        self.chromium_blue_args = chromium_blue_args.split() if chromium_blue_args else []
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def _generate_fingerprint_script(self, profile: Optional[Dict], viewport_size: Dict) -> str:
        """Generates JS script to inject for fingerprint spoofing. Includes WebRTC JS fix."""
        if not profile: return ""

        parts = []
        nav = profile.get("navigator", {})
        scr = profile.get("screen", {})
        can = profile.get("canvas", {})
        wgl = profile.get("webgl", {})
        tz = profile.get("timezone")
        noise_level = can.get('noise_level', 0)
        # Check the global setting from the config dict passed to the bot instance
        apply_webrtc_js_fix = settings.get("PREVENT_WEBRTC_IP_LEAK", True)

        # --- Screen Spoofing ---
        parts.append("/* --- Screen Spoofing --- */")
        parts.append(f"try {{ Object.defineProperty(screen, 'colorDepth', {{ value: {scr.get('colorDepth', 24)}, configurable: true, writable: false }}); }} catch(e){{ console.error('FP Screen colorDepth:', e) }}")
        parts.append(f"try {{ Object.defineProperty(screen, 'pixelDepth', {{ value: {scr.get('pixelDepth', 24)}, configurable: true, writable: false }}); }} catch(e){{ console.error('FP Screen pixelDepth:', e) }}")

        # --- Navigator Spoofing ---
        parts.append("\n/* --- Navigator Spoofing --- */")
        if 'vendor' in nav: parts.append(f"try {{ Object.defineProperty(navigator, 'vendor', {{ value: {json.dumps(nav['vendor'])}, configurable: true, writable: false }}); }} catch(e){{ console.error('FP Nav vendor:', e) }}")
        if 'platform' in nav: parts.append(f"try {{ Object.defineProperty(navigator, 'platform', {{ value: {json.dumps(nav['platform'])}, configurable: true, writable: false }}); }} catch(e){{ console.error('FP Nav platform:', e) }}")
        if 'languages' in nav and isinstance(nav['languages'], list) and nav['languages']:
             parts.append(f"try {{ Object.defineProperty(navigator, 'languages', {{ value: Object.freeze({json.dumps(nav['languages'])}), configurable: true }}); }} catch(e){{ console.error('FP Nav languages:', e) }}")
        elif 'languages' not in nav: # If profile missing languages, inject a default
             parts.append(f"try {{ Object.defineProperty(navigator, 'languages', {{ value: Object.freeze(['en-US', 'en']), configurable: true }}); }} catch(e){{ console.error('FP Nav default languages:', e) }}")

        if 'hardwareConcurrency' in nav and isinstance(nav['hardwareConcurrency'], list) and nav['hardwareConcurrency']:
             hc_value = random.choice(nav['hardwareConcurrency'])
             parts.append(f"try {{ Object.defineProperty(navigator, 'hardwareConcurrency', {{ value: {hc_value}, configurable: true, writable: false }}); }} catch(e){{ console.error('FP Nav hardwareConcurrency:', e) }}")
        elif 'hardwareConcurrency' not in nav: # Inject default if missing
             parts.append(f"try {{ Object.defineProperty(navigator, 'hardwareConcurrency', {{ value: {random.choice([4, 8, 12, 16])}, configurable: true, writable: false }}); }} catch(e){{ console.error('FP Nav default hardwareConcurrency:', e) }}")

        if 'deviceMemory' in nav and isinstance(nav['deviceMemory'], list) and nav['deviceMemory']:
             dm_value = random.choice(nav['deviceMemory'])
             parts.append(f"try {{ Object.defineProperty(navigator, 'deviceMemory', {{ value: {dm_value}, configurable: true, writable: false }}); }} catch(e){{ console.error('FP Nav deviceMemory:', e) }}")
        elif 'deviceMemory' not in nav: # Inject default if missing
             parts.append(f"try {{ Object.defineProperty(navigator, 'deviceMemory', {{ value: {random.choice([4, 8])}, configurable: true, writable: false }}); }} catch(e){{ console.error('FP Nav default deviceMemory:', e) }}")

        # --- Plugins and MimeTypes Simulation ---
        parts.append("\n/* --- Plugins/MimeTypes Spoofing --- */")
        if 'plugins' in nav and 'mimeTypes' in nav:
            profile_plugins = nav.get('plugins', [])
            profile_mimeTypes = nav.get('mimeTypes', [])
            if isinstance(profile_plugins, list) and isinstance(profile_mimeTypes, list):
                try:
                    # Ensure no invalid JSON characters in descriptions etc.
                    safe_plugins = json.dumps(profile_plugins, ensure_ascii=False)
                    safe_mimeTypes = json.dumps(profile_mimeTypes, ensure_ascii=False)
                    parts.append(f"""
                        try {{
                            const pluginsData = JSON.parse('{safe_plugins.replace("'", "\\'")}'); // Re-parse in JS
                            const mimeTypesData = JSON.parse('{safe_mimeTypes.replace("'", "\\'")}'); // Re-parse in JS

                            // Check if prototype exists before trying to use them
                            if (typeof MimeType === 'undefined' || typeof Plugin === 'undefined' || typeof MimeTypeArray === 'undefined' || typeof PluginArray === 'undefined') {{
                                console.warn('FP: MimeType/Plugin prototypes not defined, cannot spoof.');
                            }} else {{
                                const createMimeType = (data) => {{
                                    const mime = {{}}; Object.assign(mime, data, {{ enabledPlugin: null }});
                                    // Handle potential lack of prototype chain in some environments
                                    try {{ Object.setPrototypeOf(mime, MimeType.prototype); }} catch (e) {{ console.warn("FP: Failed setPrototypeOf MimeType", e); }}
                                    return Object.freeze(mime);
                                }};
                                const createPlugin = (data, mimeTypeMap) => {{
                                    const plugin = {{}}; Object.assign(plugin, data);
                                    const pluginMimeTypes = (data.mimeTypes || []).map(mtData => mtData && mtData.type ? mimeTypeMap[mtData.type] : null).filter(Boolean);
                                    plugin.length = pluginMimeTypes.length;
                                    pluginMimeTypes.forEach((mime, i) => {{ plugin[i] = mime; if (mime) mime.enabledPlugin = plugin; }});
                                    plugin.namedItem = (name) => pluginMimeTypes.find(mt => mt && mt.type === name) || null;
                                    plugin.item = (index) => pluginMimeTypes[index] || null;
                                    try {{ Object.setPrototypeOf(plugin, Plugin.prototype); }} catch (e) {{ console.warn("FP: Failed setPrototypeOf Plugin", e); }}
                                    return Object.freeze(plugin);
                                }};

                                const mimeTypeMap = {{}}; mimeTypesData.forEach(d => {{ if(d && d.type) mimeTypeMap[d.type] = createMimeType(d); }});
                                const pluginMap = {{}}; pluginsData.forEach(d => {{ if(d && d.name) pluginMap[d.name] = createPlugin(d, mimeTypeMap); }});

                                const mimeTypesArray = Object.values(mimeTypeMap);
                                try {{ Object.setPrototypeOf(mimeTypesArray, MimeTypeArray.prototype); }} catch (e) {{ console.warn("FP: Failed setPrototypeOf MimeTypeArray", e); }}
                                Object.assign(mimeTypesArray, {{
                                    namedItem: (name) => mimeTypeMap[name] || null,
                                    item: (index) => mimeTypesArray[index] || null,
                                    length: mimeTypesArray.length // Ensure length property
                                }});
                                Object.defineProperty(navigator, 'mimeTypes', {{ value: Object.freeze(mimeTypesArray), configurable: true, enumerable: true }}); // Use value directly

                                const pluginsArray = Object.values(pluginMap);
                                try {{ Object.setPrototypeOf(pluginsArray, PluginArray.prototype); }} catch (e) {{ console.warn("FP: Failed setPrototypeOf PluginArray", e); }}
                                Object.assign(pluginsArray, {{
                                    namedItem: (name) => pluginMap[name] || null,
                                    item: (index) => pluginsArray[index] || null,
                                    refresh: () => {{}},
                                    length: pluginsArray.length // Ensure length property
                                }});
                                Object.defineProperty(navigator, 'plugins', {{ value: Object.freeze(pluginsArray), configurable: true, enumerable: true }}); // Use value directly
                            }}
                        }} catch (e) {{ console.error('Fingerprint Error: Failed to spoof plugins/mimeTypes:', e); }}
                    """)
                except Exception as json_err:
                     error_logger.error(f"FP Error: Failed to prepare JSON for plugins/mimetypes: {json_err}")


        # --- Canvas Spoofing (Noise) ---
        parts.append("\n/* --- Canvas Spoofing --- */")
        if noise_level > 0:
            parts.append(f"""
                try {{
                    const canvasNoiseLevel = {noise_level};
                    const originalGetContext = HTMLCanvasElement.prototype.getContext;
                    const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
                    const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;

                    // Use a more subtle noise function
                    const addNoise = (imageData) => {{
                        if (!imageData || !imageData.data) return;
                        const pixels = imageData.data;
                        const len = pixels.length;
                        const noiseFactor = canvasNoiseLevel * 255 * 0.1; // Smaller noise factor
                        for (let i = 0; i < len; i += 4) {{
                            // Only apply noise to non-transparent pixels slightly
                            if (pixels[i+3] > 128) {{ // Check alpha
                                const noiseR = (Math.random() - 0.5) * noiseFactor;
                                const noiseG = (Math.random() - 0.5) * noiseFactor;
                                const noiseB = (Math.random() - 0.5) * noiseFactor;
                                pixels[i]   = Math.max(0, Math.min(255, pixels[i]   + noiseR));
                                pixels[i+1] = Math.max(0, Math.min(255, pixels[i+1] + noiseG));
                                pixels[i+2] = Math.max(0, Math.min(255, pixels[i+2] + noiseB));
                            }}
                        }}
                    }};

                    // Hook getContext to potentially modify contexts later
                    HTMLCanvasElement.prototype.getContext = function(type, attributes) {{
                        let context = null;
                        try {{
                            context = originalGetContext.call(this, type, attributes);
                        }} catch (e) {{ console.error('FP: Original getContext failed:', e); throw e; }}
                        if (!context) return null;

                        try {{ this.__fp_contextType = type; }} catch(e) {{}} // Store context type

                        // Hook 2D context's getImageData
                        if (type === '2d' && context.getImageData && !context.__fp_hookedGetImageData) {{
                            const originalCtxGetImageData = context.getImageData;
                            context.getImageData = function(...args) {{
                                let imageData = null;
                                try {{ imageData = originalCtxGetImageData.apply(context, args); }}
                                catch (e) {{ console.error('FP: Original getImageData failed:', e); throw e; }}
                                try {{ addNoise(imageData); }}
                                catch (e) {{ console.error('FP: Adding canvas noise failed:', e); }}
                                return imageData;
                            }};
                            context.__fp_hookedGetImageData = true; // Mark as hooked
                        }}

                        // Hook WebGL context's getParameter
                        if (type && type.includes('webgl') && context.getParameter && !context.__fp_hookedGetParameter) {{
                             const webglVendor = {json.dumps(wgl.get('vendor', 'WebGL Vendor'))};
                             const webglRenderer = {json.dumps(wgl.get('renderer', 'WebGL Renderer'))};
                             const originalGetParameter = context.getParameter;
                             context.getParameter = function(parameter) {{
                                const RENDERER = context.RENDERER || 7937; // Use context constants if available
                                const VENDOR = context.VENDOR || 7936;
                                const UNMASKED_RENDERER_WEBGL = context.UNMASKED_RENDERER_WEBGL || 37446;
                                const UNMASKED_VENDOR_WEBGL = context.UNMASKED_VENDOR_WEBGL || 37445;

                                if (parameter === RENDERER || parameter === UNMASKED_RENDERER_WEBGL) return webglRenderer;
                                if (parameter === VENDOR || parameter === UNMASKED_VENDOR_WEBGL) return webglVendor;
                                try {{ return originalGetParameter.call(context, parameter); }}
                                catch (e) {{ console.error('FP Error: Calling original WebGL getParameter failed.', e); throw e; }}
                             }};
                             context.__fp_hookedGetParameter = true; // Mark as hooked
                         }}
                         return context;
                    }};

                    // Hook toDataURL to trigger noise injection for 2D contexts
                    HTMLCanvasElement.prototype.toDataURL = function(type, quality) {{
                        // If it's a 2D context and noise hasn't been triggered yet via getImageData
                        // We need to draw something tiny to ensure the buffer exists for getImageData call
                        if (this.__fp_contextType === '2d') {{
                            try {{
                                const ctx = originalGetContext.call(this, '2d'); // Get raw context
                                if (ctx && this.width > 0 && this.height > 0 && !ctx.__fp_hookedGetImageData) {{
                                     // GetImageData wasn't called, try to trigger noise by reading a pixel
                                     ctx.getImageData(0, 0, 1, 1);
                                }}
                            }} catch (e) {{ console.error("FP Error: Noise trigger before toDataURL:", e); }}
                        }}
                        let dataURL = "";
                        try {{ dataURL = originalToDataURL.call(this, type, quality); }}
                        catch (e) {{ console.error('FP: Original toDataURL failed:', e); throw e; }}
                        return dataURL;
                    }};

                }} catch(e) {{ console.error("Fingerprint Error: Failed setting up canvas hooks:", e); }}
            """)

        # --- Timezone Spoofing ---
        parts.append("\n/* --- Timezone Spoofing --- */")
        # // Date.prototype.toLocaleTimeString = function(...args) {{ /* ... spoof with targetTimezone ... */ }};



        # --- WebDriver flag removal ---
        parts.append("\n/* --- WebDriver Flag --- */")
        if settings.get("DISABLE_AUTOMATION_FLAGS", True):
             parts.append("""
                try {
                    // Remove navigator.webdriver
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined, configurable: true });

                    // Remove window.cdc_ and related properties often used for detection
                    const propsToRemove = ['cdc_', '$cdc_', '_selenium', '_driver', 'callSelenium', 'callPhantom'];
                    for (const prop of propsToRemove) {
                        if (window[prop]) delete window[prop];
                        if (window.document[prop]) delete window.document[prop]; // Check document too
                    }
                    // Remove from frames if possible (limited access)
                    try {
                        for (const frame of window.frames) {
                           if (!frame) continue;
                           try {
                              for (const prop of propsToRemove) {
                                  if (frame[prop]) delete frame[prop];
                                  if (frame.document && frame.document[prop]) delete frame.document[prop];
                              }
                           } catch(e) {} // Ignore cross-origin errors
                        }
                    } catch(e) {}

                    // Override Permissions API query for notifications (often abused)
                    if (navigator.permissions && navigator.permissions.query) {
                        const originalQuery = navigator.permissions.query;
                        navigator.permissions.query = function(permissionDesc) {
                            if (permissionDesc && permissionDesc.name === 'notifications') {
                                return Promise.resolve({ state: 'prompt', name: 'notifications' }); // Simulate default prompt state
                            }
                            return originalQuery.call(this, permissionDesc);
                        };
                         // Preserve original toString if possible
                         try { navigator.permissions.query.toString = () => originalQuery.toString(); } catch(e) {}
                    }

                } catch(e){ console.error('FP WebDriver flag removal:', e) }
             """)

        # --- WebRTC Local IP Leak Prevention (JS Based - Backup/Additional Layer) ---
        parts.append("\n/* --- WebRTC IP Leak Mitigation (JS) --- */")
        if apply_webrtc_js_fix:
            logging.info("Adding WebRTC JS mitigation script part.")
            parts.append("""
                try {
                    const originalRTCPeerConnection = window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection;
                    if (originalRTCPeerConnection) {
                        // Regex for local/private IP addresses (more comprehensive)
                        const LOCAL_IP_REGEX = /^(10\\.(?:\\d{1,3}\\.){2}\\d{1,3})|(172\\.(?:1[6-9]|2\\d|3[0-1])\\.\\d{1,3}\\.\\d{1,3})|(192\\.168\\.\\d{1,3}\\.\\d{1,3})|(169\\.254\\.\\d{1,3}\\.\\d{1,3})|(127\\.(?:\\d{1,3}\\.){2}\\d{1,3})|(::1)|(f[cd][0-9a-f]{2}:(?:[0-9a-f]{0,4}:){0,6}:?[0-9a-f]{0,4})|(fe80:(?::[0-9a-f]{0,4}){0,4}%[0-9a-zA-Z]{1,})|($^)/i;
                        // Regex for mDNS hostnames (.local)
                        const MDNS_REGEX = /^[\\w\\-]+\\.local$/i;

                        const NewRTCPeerConnection = function(...args) {
                            let pc;
                            try { pc = new originalRTCPeerConnection(...args); }
                            catch(e) { console.error('FP: Failed to create original RTCPeerConnection:', e); throw e; }

                            try {
                                // --- Hook addIceCandidate ---
                                const originalAddIceCandidate = pc.addIceCandidate;
                                pc.addIceCandidate = function(candidate, ...rest) {
                                    try {
                                        if (candidate && candidate.candidate) {
                                            const sdpLines = candidate.candidate.split('\\n');
                                            let candidateLine = sdpLines.find(line => line.startsWith('a=candidate:'));
                                            if (candidateLine) {
                                                const parts = candidateLine.split(' ');
                                                // Check IP address (usually parts[4])
                                                if (parts.length >= 5 && LOCAL_IP_REGEX.test(parts[4])) {
                                                    console.log('FP: Blocking local IP ICE candidate via addIceCandidate:', parts[4]);
                                                    return Promise.resolve(); // Silently drop
                                                }
                                                // Check mDNS hostname (often parts[4] as well)
                                                if (parts.length >= 5 && MDNS_REGEX.test(parts[4])) {
                                                    console.log('FP: Blocking mDNS hostname ICE candidate via addIceCandidate:', parts[4]);
                                                    return Promise.resolve(); // Silently drop
                                                }
                                                // Check related IP (parts[6] if exists) - less common but possible
                                                 if (parts.length >= 7 && LOCAL_IP_REGEX.test(parts[6])) {
                                                     console.log('FP: Blocking related local IP ICE candidate via addIceCandidate:', parts[6]);
                                                     return Promise.resolve(); // Silently drop
                                                 }
                                                 if (parts.length >= 7 && MDNS_REGEX.test(parts[6])) {
                                                     console.log('FP: Blocking related mDNS hostname ICE candidate via addIceCandidate:', parts[6]);
                                                     return Promise.resolve(); // Silently drop
                                                 }
                                            }
                                        }
                                    } catch (e) { console.error('FP Error filtering ICE candidate in addIceCandidate:', e); }
                                    // Call original method if candidate is not local or check fails
                                    return originalAddIceCandidate.call(this, candidate, ...rest);
                                };

                                // --- Hook onicecandidate ---
                                const originalOnIceCandidate = pc.onicecandidate;
                                pc.onicecandidate = function(event) {
                                    try {
                                        if (event && event.candidate && event.candidate.candidate) {
                                            const sdpLines = event.candidate.candidate.split('\\n');
                                            let candidateLine = sdpLines.find(line => line.startsWith('a=candidate:'));
                                            if (candidateLine) {
                                                 const parts = candidateLine.split(' ');
                                                 // Check IP address (parts[4])
                                                 if (parts.length >= 5 && (LOCAL_IP_REGEX.test(parts[4]) || MDNS_REGEX.test(parts[4]))) {
                                                     console.log('FP: Detected local IP/mDNS via onicecandidate:', parts[4], '- Replacing candidate with null.');
                                                     // Modify the event object before dispatching to original handler (if any)
                                                     // Create a new event or modify if possible (can be tricky)
                                                     // Simplest is to nullify the candidate if possible, BUT this might break functionality.
                                                     // Preferring addIceCandidate filtering. Logging detection here.
                                                     // event.candidate = null; // Risky - might break things
                                                 }
                                                 // Check related IP (parts[6])
                                                 if (parts.length >= 7 && (LOCAL_IP_REGEX.test(parts[6]) || MDNS_REGEX.test(parts[6]))) {
                                                     console.log('FP: Detected related local IP/mDNS via onicecandidate:', parts[6], '- Replacing candidate with null.');
                                                     // event.candidate = null; // Risky
                                                 }
                                            }
                                        }
                                    } catch (e) { console.error('FP Error checking ICE candidate in onicecandidate:', e); }

                                    // Call original handler if it exists, possibly with modified event
                                    if (originalOnIceCandidate) {
                                        // If we modified event.candidate, pass the modified event
                                        return originalOnIceCandidate.call(this, event);
                                    }
                                };

                                // --- Hook createOffer --- (Less common leak vector, but possible via SDP)
                                const originalCreateOffer = pc.createOffer;
                                pc.createOffer = function(...args) {
                                    return originalCreateOffer.call(this, ...args).then(offer => {
                                        try {
                                            // Basic check: remove lines with local IPs from SDP if found
                                            offer.sdp = offer.sdp.split('\\n').filter(line => {
                                                const parts = line.split(' ');
                                                // Check 'c=' lines (connection data)
                                                if (line.startsWith('c=IN IP4') && parts.length >= 3 && LOCAL_IP_REGEX.test(parts[2])) {
                                                     console.log('FP: Removing local IP from createOffer SDP (c= line):', parts[2]); return false;
                                                }
                                                // Check 'a=candidate' lines (already handled by onicecandidate/addIceCandidate, but maybe belt-and-suspenders?)
                                                if (line.startsWith('a=candidate:') && parts.length >= 5 && (LOCAL_IP_REGEX.test(parts[4]) || MDNS_REGEX.test(parts[4]))) {
                                                     console.log('FP: Removing local IP/mDNS from createOffer SDP (a=candidate):', parts[4]); return false;
                                                }
                                                return true;
                                            }).join('\\n');
                                        } catch(e) { console.error("FP: Error filtering SDP in createOffer", e); }
                                        return offer;
                                    });
                                };

                                // --- Hook createAnswer --- (Similar filtering as createOffer)
                                const originalCreateAnswer = pc.createAnswer;
                                pc.createAnswer = function(...args) {
                                     return originalCreateAnswer.call(this, ...args).then(answer => {
                                         try {
                                             answer.sdp = answer.sdp.split('\\n').filter(line => {
                                                 const parts = line.split(' ');
                                                 if (line.startsWith('c=IN IP4') && parts.length >= 3 && LOCAL_IP_REGEX.test(parts[2])) {
                                                      console.log('FP: Removing local IP from createAnswer SDP (c= line):', parts[2]); return false;
                                                 }
                                                 if (line.startsWith('a=candidate:') && parts.length >= 5 && (LOCAL_IP_REGEX.test(parts[4]) || MDNS_REGEX.test(parts[4]))) {
                                                      console.log('FP: Removing local IP/mDNS from createAnswer SDP (a=candidate):', parts[4]); return false;
                                                 }
                                                 return true;
                                             }).join('\\n');
                                         } catch(e) { console.error("FP: Error filtering SDP in createAnswer", e); }
                                         return answer;
                                     });
                                };


                            } catch(e) { console.error('FP: Failed to hook RTCPeerConnection methods:', e); }
                            return pc;
                        };

                        // Maintain prototype chain and constructor identity
                        NewRTCPeerConnection.prototype = originalRTCPeerConnection.prototype;
                        NewRTCPeerConnection.prototype.constructor = NewRTCPeerConnection;
                        Object.defineProperty(NewRTCPeerConnection, 'name', { value: originalRTCPeerConnection.name, configurable: true });

                        // Replace global constructors
                        window.RTCPeerConnection = NewRTCPeerConnection;
                        window.webkitRTCPeerConnection = NewRTCPeerConnection;
                        window.mozRTCPeerConnection = NewRTCPeerConnection;
                        console.log("FP: WebRTC JS mitigation applied.");

                    } else {
                         console.log("FP: RTCPeerConnection not found, skipping WebRTC JS mitigation.");
                    }
                } catch (e) {
                    console.error('Fingerprint Error: Failed applying WebRTC JS mitigation:', e);
                }
            """)
        else:
            logging.info("WebRTC JS mitigation script part skipped (disabled in settings).")


        return "\n".join(parts)

    def start_browser(self, user_agent: str, viewport_size: Dict, fingerprint_profile: Optional[Dict]):
        """Starts a new browser instance applying fingerprint and args."""
        launch_options = {
            "headless": self.headless,
            "args": ['--no-sandbox', '--disable-setuid-sandbox'] # Common args
        }

        # --- Apply Proxy ---
        if self.proxy_server and self.proxy_type in ("http", "socks4", "socks5"):
            # Determine scheme: socks need socks://, http/https use http://
            # self.proxy_type should already be 'http' if original was 'https'
            scheme = self.proxy_type if self.proxy_type.startswith("socks") else "http"
            launch_options["proxy"] = {"server": f"{scheme}://{self.proxy_server}"}
            logging.info(f"Using Proxy: {scheme}://{self.proxy_server} (Internal Type: {self.proxy_type})")
            if self.proxy_type != "socks5" and settings.get("PREVENT_WEBRTC_IP_LEAK", True):
                 logging.warning("WebRTC leak prevention enabled, but proxy is not SOCKS5. Public IP exposure might still occur depending on proxy behavior.")
        else:
            logging.info("Not using proxy.")
            if settings.get("PREVENT_WEBRTC_IP_LEAK", True):
                 logging.info("WebRTC leak prevention enabled, but no proxy configured. Real public IP will be used for WebRTC (local IPs should still be hidden by flag).")

        # --- Apply Anti-Automation Args & Settings ---
        extra_args = []
        ignore_default = []
        if settings.get("DISABLE_AUTOMATION_FLAGS", True):
            extra_args.extend([
                '--disable-blink-features=AutomationControlled',
                '--disable-features=UserAgentClientHint', # Disable UA Client Hints for simplicity
                '--disable-component-extensions-with-background-pages', # Disable some extensions
                '--deny-permission-prompts', # Deny permission prompts automatically
                # '--use-fake-ui-for-media-stream', # Can sometimes help
                # '--use-fake-device-for-media-stream' # Can sometimes help
            ])
            ignore_default.extend(["--enable-automation", "--enable-logging", "--enable-blink-features=IdleDetection"]) # Add IdleDetection
            logging.info("Applying anti-automation flags.")

        # --- Apply WebRTC Launch Args (Most Important Part) ---
        if settings.get("PREVENT_WEBRTC_IP_LEAK", True):
            # This flag forces Chrome/Chromium to only bind WebRTC to the default public network interface.
            # It prevents enumeration and usage of local/VPN interfaces for ICE candidates.
            # Works best with SOCKS5 proxy to also mask public IP, but crucial for local IP hiding regardless.
            extra_args.append('--force-webrtc-ip-handling-policy=default_public_interface_only')
            # Optional: Disable mDNS hostname generation for candidates (another potential leak vector)
            extra_args.append('--disable-features=WebRtcHideLocalIpsWithMdns')
            logging.info("Applying WebRTC IP handling policy flag (--force-webrtc-ip-handling-policy) and disabling mDNS host candidates.")
        else:
             logging.warning("WebRTC IP leak prevention flag is DISABLED in settings.")


        # Combine args and ignore_default_args
        launch_options["args"].extend(extra_args)
        if ignore_default:
            # De-duplicate ignore_default
            launch_options["ignore_default_args"] = list(set(ignore_default))
            # Remove ignored args from the main args list if they were added manually
            launch_options["args"] = [arg for arg in launch_options["args"] if arg not in launch_options["ignore_default_args"]]

        # Ensure --no-sandbox remains if not explicitly ignored
        if '--no-sandbox' not in launch_options["args"] and '--no-sandbox' not in launch_options.get("ignore_default_args", []):
             launch_options["args"].append('--no-sandbox')


        # --- Apply Chromium Blue Specifics ---
        if self.use_chromium_blue:
            if not self.chromium_blue_path or not os.path.exists(self.chromium_blue_path):
                 error_logger.error(f"Chromium Blue enabled but path is invalid or missing: {self.chromium_blue_path}")
                 raise FileNotFoundError(f"Chromium Blue executable not found at: {self.chromium_blue_path}")
            launch_options["executable_path"] = self.chromium_blue_path
            launch_options["args"].extend(self.chromium_blue_args) # Add custom args for CB
            logging.info(f"Using Custom Chromium: {self.chromium_blue_path}")

        try:
            browser_type = self.playwright.chromium
            logging.info(f"Launching browser (Headless: {self.headless}) with options:")
            logging.info(f"  executable_path: {launch_options.get('executable_path', 'Default Playwright Chromium')}")
            logging.info(f"  args: {launch_options.get('args')}")
            if 'ignore_default_args' in launch_options: logging.info(f"  ignore_default_args: {launch_options['ignore_default_args']}")
            if 'proxy' in launch_options: logging.info(f"  proxy: {launch_options['proxy']}")

            self.browser = browser_type.launch(**launch_options)

            context_options = {
                "user_agent": user_agent,
                "viewport": viewport_size,
                "java_script_enabled": True,
                "bypass_csp": False, # Keep CSP enabled for realism
                "accept_downloads": False,
                # Set device scale factor based on typical ranges?
                "device_scale_factor": random.choice([1, 1.25, 1.5, 2]) if 'Mobile' not in user_agent else random.choice([2, 2.5, 3]),
                "is_mobile": 'Mobile' in user_agent or 'Android' in user_agent or 'iPhone' in user_agent,
                "has_touch": 'Mobile' in user_agent or 'Android' in user_agent or 'iPhone' in user_agent,
            }

            # Apply locale and timezone from profile directly to context options
            profile_locale = "en-US" # Default
            profile_timezone = "America/New_York" # Default
            if fingerprint_profile:
                profile_nav = fingerprint_profile.get("navigator", {})
                if profile_nav.get("languages"):
                    lang = profile_nav["languages"][0]
                    profile_locale = lang.replace("_", "-") # Ensure correct format (en-US)
                if fingerprint_profile.get("timezone"):
                    profile_timezone = fingerprint_profile.get("timezone")

            context_options["locale"] = profile_locale
            context_options["timezone_id"] = profile_timezone
            # Geolocation spoofing (optional, can add later if needed)
            # context_options["geolocation"] = {"longitude": ..., "latitude": ..., "accuracy": ...}
            # Permissions
            # context_options["permissions"] = ["geolocation"] # e.g., grant geolocation by default if spoofing

            logging.info(f"Creating context with options: {context_options}")
            self.context = self.browser.new_context(**context_options)
            self.context.set_default_navigation_timeout(90000) # Increase default nav timeout for context
            self.context.set_default_timeout(60000) # Increase default action timeout

            # --- Inject Fingerprint Spoofing Script ---
            spoof_script = self._generate_fingerprint_script(fingerprint_profile, viewport_size)
            if spoof_script:
                try:
                    self.context.add_init_script(script=spoof_script)
                    logging.info("Fingerprint spoofing script injected (incl. WebRTC JS attempt).")
                except Exception as script_error:
                     error_logger.error(f"Failed to inject fingerprint script: {script_error}")

            self.page = self.context.new_page()

            # Add Extra HTTP Headers (Refined)
            try:
                ch_ua_brand, ch_ua_version, platform_hint = "Chromium", "121", '"Unknown"' # Defaults
                is_mobile = context_options["is_mobile"]

                # Extract browser/version
                ua_lower = user_agent.lower()
                if "edg/" in ua_lower: ch_ua_brand = "Microsoft Edge"
                elif "chrome/" in ua_lower and "chromium/" not in ua_lower: ch_ua_brand = "Google Chrome"
                elif "firefox/" in ua_lower: ch_ua_brand = "Firefox"
                elif "safari/" in ua_lower and "chrome/" not in ua_lower: ch_ua_brand = "Safari"

                version_match = re.search(r"(?:Chrome|Firefox|Edg|Version|Safari)/([\d\.]+)", user_agent)
                if version_match:
                     try: ch_ua_version = str(int(version_match.group(1).split('.')[0]))
                     except: pass

                # Determine platform hint
                if "windows" in ua_lower: platform_hint = '"Windows"'
                elif "macintosh" in ua_lower or "mac os x" in ua_lower: platform_hint = '"macOS"'
                elif "android" in ua_lower: platform_hint = '"Android"'
                elif "linux" in ua_lower: platform_hint = '"Linux"'
                elif "iphone" in ua_lower or "ipad" in ua_lower: platform_hint = '"iOS"'


                # Construct Sec-CH-UA
                # Note: "Not/A)Brand" is deprecated, use "Not_A Brand" or similar
                brands_list = [f'"{ch_ua_brand}";v="{ch_ua_version}"']
                if ch_ua_brand != "Chromium": brands_list.append(f'"Chromium";v="{ch_ua_version}"') # Often includes Chromium
                brands_list.append(f'"Not_A Brand";v="8"') # Common placeholder
                full_ch_ua = ", ".join(brands_list)

                headers = {
                    'Accept-Language': context_options.get('locale', 'en-US') + ',en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br', # Support brotli
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-CH-UA': full_ch_ua,
                    'Sec-CH-UA-Mobile': '?1' if is_mobile else '?0',
                    'Sec-CH-UA-Platform': platform_hint,
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-User': '?1',
                    'Sec-Fetch-Dest': 'document',
                    'DNT': '1', # Do Not Track header
                }
                self.page.set_extra_http_headers(headers)
                logging.info("Set extra HTTP headers including Sec-CH-UA.")
            except Exception as header_error:
                error_logger.error(f"Failed to set extra HTTP headers: {header_error}")

            logging.info(f"Browser and context started successfully. Profile: {fingerprint_profile.get('name', 'None') if fingerprint_profile else 'None'}")

        except PlaywrightError as e:
            error_logger.error(f"Playwright Error: Failed to start browser or context: {e}")
            if "Executable doesn't exist" in str(e) and self.use_chromium_blue:
                 error_logger.error(f"Check Custom Chromium path: {self.chromium_blue_path}")
            elif "download" in str(e).lower() and "host" in str(e).lower():
                 error_logger.error("Potential issue connecting to Playwright browser download host. Check network/firewall.")
            elif "Target page, context or browser has been closed" in str(e):
                 logging.warning(f"Browser launch interrupted or failed: {e}") # Less severe log for closure during launch
            else:
                 error_logger.exception("Unhandled PlaywrightError during browser start:") # Log full trace for others
            raise
        except FileNotFoundError as e:
             error_logger.error(str(e))
             raise
        except Exception as e:
            error_logger.exception(f"Unexpected error during browser start:")
            raise

    def navigate_to(self, url: str):
        """Navigates to the given URL, waits for DOM content, adds short pause."""
        if not self.page or self.page.is_closed():
            raise Exception("Browser page is not available for navigation.")
        try:
            logging.info(f"Navigating to: {url}")
            # Use context's default timeout, wait for 'domcontentloaded'
            response = self.page.goto(url, wait_until="domcontentloaded")

            if response:
                 logging.info(f"Navigation initiated: Status {response.status} for {response.url}")
                 # Short random pause after DOM loaded, before further interactions
                 delay_ms = random.randint(500, 1500)
                 logging.debug(f"Pausing for {delay_ms}ms after domcontentloaded.")
                 try: self.page.wait_for_timeout(delay_ms)
                 except PlaywrightError as wait_err: logging.warning(f"Timeout during post-navigation pause: {wait_err}") # Non-critical
            else:
                 logging.warning(f"Navigation to {url} did not return a standard response object (might be unusual, e.g., about:blank).")

        except TimeoutError as e:
             error_logger.error(f"Navigation TimeoutError for {url}: {e}")
             self.take_screenshot(f"nav_timeout_error_{int(time.time())}.png")
             raise
        except PlaywrightError as e:
            # Filter common closure errors which might happen if stop is clicked during nav
            if "Target page, context or browser has been closed" in str(e) or \
               "Navigation failed because page was closed" in str(e) or \
               "frame was detached" in str(e):
                 logging.warning(f"Navigation cancelled or failed due to closure for {url}: {e}")
                 # Don't raise if likely due to stop? Or maybe still raise to signal failure? Raising is safer.
                 raise # Raise to indicate the operation didn't complete normally
            else:
                error_logger.error(f"Navigation PlaywrightError for {url}: {e}")
                self.take_screenshot(f"nav_playwright_error_{int(time.time())}.png")
                raise
        except Exception as e:
            error_logger.exception(f"Unexpected navigation error for {url}:")
            self.take_screenshot(f"nav_unexpected_error_{int(time.time())}.png")
            raise

    def close_browser(self):
        """Closes the browser instance gracefully."""
        logging.info("Attempting to close browser resources...")
        closed_something = False
        # Close page first, then context, then browser
        if self.page and not self.page.is_closed():
             try: self.page.close(); closed_something = True; logging.debug("Page closed.")
             except Exception as e: error_logger.warning(f"Error closing page: {e}")
             finally: self.page = None
        if self.context:
             try: self.context.close(); closed_something = True; logging.debug("Context closed.")
             except Exception as e: error_logger.warning(f"Error closing context: {e}")
             finally: self.context = None
        if self.browser:
            try: self.browser.close(); closed_something = True; logging.debug("Browser closed.")
            except Exception as e: error_logger.warning(f"Error closing browser: {e}")
            finally: self.browser = None

        if closed_something: logging.info("Browser resources closed.")
        else: logging.debug("No browser resources needed closing or were already closed/None.")


    def take_screenshot(self, filename: str = "screenshot.png"):
        """Takes a screenshot of the current page."""
        if self.page and not self.page.is_closed():
            try:
                screenshot_path = os.path.join("logs", filename)
                os.makedirs("logs", exist_ok=True) # Ensure dir exists
                self.page.screenshot(path=screenshot_path, full_page=False)
                logging.info(f"Screenshot saved to {screenshot_path}")
            except PlaywrightError as e: error_logger.error(f"Error taking screenshot: {e}")
            except Exception as e: error_logger.exception(f"Unexpected error taking screenshot:")
        else: logging.debug("Skipping screenshot, page is None or closed.")


# --- ScrollingManager, TextSelectionManager, NextPageNavigator, AdClickManager, FormFiller ---
# (No major changes needed here based on the request, kept previous improvements)
class ScrollingManager:
    """Handles human-like scrolling behavior, including Gemini-driven patterns."""

    def __init__(self, page: Page):
        self.page = page

    def smooth_scroll_to(self, scroll_to: int, duration: int):
        """Smoothly scrolls using JS requestAnimationFrame."""
        if not self.page or self.page.is_closed():
            logging.warning("Smooth scroll attempted but page is closed.")
            return
        try:
            current_scroll = self.page.evaluate("window.pageYOffset")
            if abs(scroll_to - current_scroll) < 10: return

            duration = max(100, min(10000, duration)) # Clamp duration

            js_code = f"""
                (() => {{
                    const startY = window.pageYOffset;
                    const endY = Math.max(0, Math.min(document.body.scrollHeight - window.innerHeight, {scroll_to}));
                    const distance = endY - startY;
                    if (Math.abs(distance) < 10) return Promise.resolve(); // Already close
                    const duration = {duration};
                    let startTime = null;
                    const easeInOutCubic = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

                    return new Promise(resolve => {{
                        const animation = (currentTime) => {{
                            if (startTime === null) startTime = currentTime;
                            const timeElapsed = currentTime - startTime;
                            const progress = Math.min(1, timeElapsed / duration);
                            const easedProgress = easeInOutCubic(progress);
                            window.scrollTo(0, startY + distance * easedProgress);
                            if (timeElapsed < duration) {{
                                requestAnimationFrame(animation);
                            }} else {{
                                resolve(); // Resolve promise when done
                            }}
                        }};
                        requestAnimationFrame(animation);
                    }});
                }})();
            """
            self.page.evaluate(js_code)
            # Wait slightly longer than duration to ensure JS completes
            self.page.wait_for_timeout(duration + random.uniform(80, 200))
        except PlaywrightError as e: error_logger.warning(f"Smooth scroll using JS failed: {e}")
        except Exception as e: error_logger.exception(f"Unexpected error during smooth scroll:")

    def random_scroll(self, min_scrolls: int = 3, max_scrolls: int = 6, duration_min:int = 500, duration_max:int = 1500):
        """Performs a series of random smooth scrolls."""
        if not self.page or self.page.is_closed(): return
        try:
            num_scrolls = random.randint(min_scrolls, max_scrolls)
            viewport_height = self.page.viewport_size["height"] if self.page.viewport_size else 800
            total_scroll_time_ms = 0
            max_total_duration_ms = 45 * 1000 # Limit total time spent scrolling randomly

            logging.debug(f"Performing {num_scrolls} random scrolls...")
            for i in range(num_scrolls):
                if total_scroll_time_ms >= max_total_duration_ms:
                    logging.debug("Reached max random scroll duration."); break
                if self.page.is_closed(): break

                scroll_fraction = random.uniform(0.2, 0.9)
                scroll_amount = int(viewport_height * scroll_fraction)
                direction = random.choice([1, 1, -1]) # Bias down

                try:
                    current_y = self.page.evaluate("window.pageYOffset")
                    max_y = self.page.evaluate("document.body.scrollHeight - window.innerHeight")
                    target_y = max(0, min(max_y, current_y + scroll_amount * direction))

                    if abs(target_y - current_y) < 50: continue # Skip tiny scrolls

                    scroll_duration = random.randint(settings.get("SCROLL_DURATION_MIN", duration_min), settings.get("SCROLL_DURATION_MAX", duration_max))
                    self.smooth_scroll_to(target_y, scroll_duration)
                    pause_after = random.uniform(0.4, 1.2)
                    self.page.wait_for_timeout(pause_after * 1000)
                    total_scroll_time_ms += scroll_duration + (pause_after * 1000)
                except PlaywrightError as scroll_eval_err:
                    error_logger.warning(f"Error during random scroll evaluation/execution: {scroll_eval_err}")
                    break # Stop scroll sequence on error

            logging.debug("Finished random scroll sequence.")
        except PlaywrightError as e: error_logger.warning(f"Random scroll failed: {e}")
        except Exception as e: error_logger.exception(f"Unexpected error during random scroll:")

    def gemini_scroll(self, browser_id, update_signal):
        """Generates human-like scroll instructions using Gemini API and executes smoothly."""
        if not self.page or self.page.is_closed(): return
        api_key = settings.get("GEMINI_API_KEY")
        if not api_key:
            logging.debug(f"Browser {browser_id+1}: Gemini API key missing, using random scroll.")
            self.random_scroll()
            return

        try:
            page_height = self.page.evaluate("document.body.scrollHeight")
            viewport_height = self.page.viewport_size["height"] if self.page.viewport_size else 800
            current_y = self.page.evaluate("window.pageYOffset")

            prompt = f"""
            You are simulating human browsing. Generate 6-15 scroll/pause instructions for a webpage interaction lasting roughly 30-60 seconds.
            Page Height: {page_height}px, Viewport Height: {viewport_height}px, Current Scroll Position: {current_y}px.
            Instructions:
            - 'scroll_down, <pixels>, <duration_sec>' (e.g., scroll_down, 550, 1.5)
            - 'scroll_up, <pixels>, <duration_sec>' (e.g., scroll_up, 300, 0.8)
            - 'pause, <duration_sec>' (e.g., pause, 2.5)
            Constraints:
            - Pixels: Reasonable values (e.g., 100 to {int(viewport_height * 1.5)}). Don't scroll beyond page boundaries.
            - Scroll Duration: 0.5 to 4.0 seconds.
            - Pause Duration: 0.5 to 5.0 seconds.
            - Pattern: Mostly scroll down, occasional small scroll ups are okay. Start with pause or scroll_down. End sequence with a pause.
            Output ONLY the instructions, one per line. No extra text or explanations.
            """
            update_signal.emit(f"Browser {browser_id+1}: Requesting scroll pattern from Gemini...", browser_id)
            fm = FingerprintManager([], [], api_key) # Temp instance to use _call_gemini_api
            scroll_instructions_text = fm._call_gemini_api(prompt, api_key, browser_id, update_signal)

            if not scroll_instructions_text:
                 update_signal.emit(f"Browser {browser_id+1}: Gemini scroll response blocked or empty. Using random scroll.", browser_id)
                 error_logger.warning(f"Gemini scroll generation failed: Response empty/blocked.")
                 self.random_scroll()
                 return

            self.execute_smooth_scroll_instructions(scroll_instructions_text, browser_id, update_signal)

        except Exception as e:
            error_logger.error(f"Browser {browser_id+1}: Error generating/executing Gemini scroll: {e}")
            update_signal.emit(f"Browser {browser_id+1}: Error with Gemini scroll: {e}. Using random scroll.", browser_id)
            self.random_scroll() # Fallback

    def execute_smooth_scroll_instructions(self, instructions_text, browser_id, update_signal):
        """Executes scroll instructions from Gemini using SMOOTH scrolling."""
        if not self.page or self.page.is_closed(): return
        if not instructions_text:
            logging.debug(f"Browser {browser_id+1}: No Gemini instructions received, using random scroll.")
            self.random_scroll(); return

        try:
            instructions = []
            for line in instructions_text.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                try:
                    if len(parts) == 3 and parts[0] in ['scroll_down', 'scroll_up']:
                         action, pixels_str, duration_str = parts
                         # Add validation/clamping
                         pixels = max(10, min(5000, int(pixels_str)))
                         duration = max(0.1, min(10.0, float(duration_str)))
                         instructions.append({'action': action, 'pixels': pixels, 'duration': duration})
                    elif len(parts) == 2 and parts[0] == 'pause':
                         action, duration_str = parts
                         duration = max(0.1, min(15.0, float(duration_str))) # Allow longer pauses
                         instructions.append({'action': action, 'duration': duration})
                    elif line.strip():
                        update_signal.emit(f"Browser {browser_id+1}: Unknown instruction format: '{line}'. Skipping.", browser_id)
                except (ValueError, IndexError):
                     update_signal.emit(f"Browser {browser_id+1}: Invalid instruction format: '{line}'. Skipping.", browser_id)

            if not instructions:
                 update_signal.emit(f"Browser {browser_id+1}: No valid instructions parsed. Using random scroll.", browser_id)
                 self.random_scroll(); return

            update_signal.emit(f"Browser {browser_id+1}: Executing {len(instructions)} Gemini scroll instructions...", browser_id)

            for instr in instructions:
                 if not self.page or self.page.is_closed():
                     update_signal.emit(f"Browser {browser_id+1}: Page closed during Gemini scroll.", browser_id)
                     break

                 action = instr['action']
                 duration_sec = instr['duration']
                 duration_ms = int(duration_sec * 1000)

                 if action == 'pause':
                     self.page.wait_for_timeout(duration_ms)
                 elif action in ['scroll_down', 'scroll_up']:
                     pixels = instr['pixels']
                     direction = 1 if action == 'scroll_down' else -1
                     try:
                         current_scroll_y = self.page.evaluate("window.pageYOffset")
                         max_scroll = self.page.evaluate("document.body.scrollHeight - window.innerHeight")
                         target_y = max(0, min(max_scroll, current_scroll_y + pixels * direction))
                         distance = abs(target_y - current_scroll_y)
                         if distance > 10:
                             # Adjust duration slightly based on distance, but keep close to Gemini's request
                             adjusted_duration_ms = max(200, min(8000, int(duration_ms + distance * 0.3)))
                             self.smooth_scroll_to(target_y, adjusted_duration_ms)
                         else:
                             self.page.wait_for_timeout(random.randint(50, 150)) # Short pause if no scroll needed
                     except (PlaywrightError, ValueError, Exception) as e:
                         update_signal.emit(f"Browser {browser_id+1}: Error executing scroll instruction: {instr}. Error: {e}", browser_id)
                         self.page.wait_for_timeout(500) # Pause on error

        except Exception as e:
            update_signal.emit(f"Browser {browser_id+1}: Major error executing Gemini scroll instructions: {e}", browser_id)
            error_logger.exception(f"Browser {browser_id+1}: Gemini scroll execution error:")
            self.random_scroll() # Fallback

class TextSelectionManager:
    """Handles realistic mouse movement, potentially towards content."""

    def __init__(self, page: Page):
        self.page = page

    def human_mouse_move(self, x: float, y: float, steps: Optional[int] = None, min_delay_ms: float = 8, max_delay_ms: float = 25):
        """Moves mouse more realistically with steps and slight pauses between steps."""
        if not settings.get("MOUSE_MOVEMENT_ENABLED", True) or not self.page or self.page.is_closed(): return
        try:
            vp = self.page.viewport_size or {"width": 1280, "height": 720}
            # Ensure target coords are within viewport bounds
            target_x = max(0, min(vp["width"] - 1, x + random.uniform(-1.5, 1.5)))
            target_y = max(0, min(vp["height"] - 1, y + random.uniform(-1.5, 1.5)))

            num_steps = steps or random.randint(15, 35)
            if num_steps <= 0 : num_steps = 1

            # Playwright's move with steps handles interpolation smoothly
            self.page.mouse.move(target_x, target_y, steps=num_steps)
            # Add a very short pause after movement completes
            pause_after_ms = random.uniform(30, 120)
            self.page.wait_for_timeout(pause_after_ms)

        except PlaywrightError as e:
             if "Target page, context or browser has been closed" not in str(e):
                 error_logger.warning(f"Human mouse move failed: {e}")
        except Exception as e: error_logger.exception(f"Unexpected error during human mouse move:")

    def select_important_text(self, browser_id, update_signal):
        """Tries to find and move mouse towards important words, links, or buttons."""
        if not settings.get("MOUSE_MOVEMENT_ENABLED", True) or not self.page or self.page.is_closed(): return
        if random.random() < settings.get("SKIP_ACTION_PROBABILITY", 0.05): return

        words_to_find = load_important_words()
        target_element = None
        found_type = None
        target_x, target_y = None, None
        search_timeout = 250 # Faster timeout for finding elements

        # Prioritize interactable elements (links, buttons) containing important words
        # if words_to_find:
        #     random.shuffle(words_to_find)
        #     # Regex: Case-insensitive, word boundary
        #     interactable_selectors = [f'a:text-matches("\\\\b{re.escape(word)}\\\\b", "i"):visible', f'button:text-matches("\\\\b{re.escape(word)}\\\\b", "i"):visible' for word in words_to_find[:7]]
        #     # Combine into one query for efficiency? Maybe not, check one by one
        #     for selector in interactable_selectors:
        #         if self.page.is_closed(): return
        #         try:
        #             # Find all visible elements matching, pick one randomly if any
        #             elements = self.page.locator(selector).all(timeout=search_timeout)
        #             enabled_elements = [el for el in elements if el.is_enabled(timeout=50)] # Quick enabled check
        #             if enabled_elements:
        #                 target_element = random.choice(enabled_elements)
        #                 found_type = 'keyword_interact'
        #                 break
        #         except (PlaywrightError, TimeoutError): continue # Expected if not found quickly
        #         except Exception as e: error_logger.error(f"Error locating important element '{selector}': {e}"); continue

        # If no interactable found, try general text matches (less likely to be hovered)
        if not target_element and words_to_find and random.random() < 0.3: # Lower chance for general text hover
            general_selectors = [f'*:text-matches("\\\\b{re.escape(word)}\\\\b", "i"):visible' for word in words_to_find[:3]]
            for selector in general_selectors:
                if self.page.is_closed(): return
                try:
                    elements = self.page.locator(selector).all(timeout=search_timeout)
                    # Avoid hovering huge containers or tiny elements
                    valid_elements = []
                    for el in elements:
                        try:
                            bb = el.bounding_box(timeout=50)
                            if bb and 5 < bb['height'] < 300 and 5 < bb['width'] < 600:
                                valid_elements.append(el)
                        except: continue
                    if valid_elements:
                        target_element = random.choice(valid_elements); found_type = 'keyword_text'; break
                except (PlaywrightError, TimeoutError): continue
                except Exception as e: error_logger.warning(f"Error locating general text '{selector}': {e}"); continue

        # Fallback: Hover common navigation/structural elements if nothing else found
        if not target_element and random.random() < 0.5: # Chance to fallback
             fallback_selectors = ['header a:visible', 'nav ul li a:visible', '#logo:visible', '.navbar-brand:visible', 'footer a:visible', 'h1:visible', 'h2:visible', 'main button:visible']
             random.shuffle(fallback_selectors)
             for fb_selector in fallback_selectors:
                  if self.page.is_closed(): return
                  try:
                       elements = self.page.locator(fb_selector).all(timeout=search_timeout)
                       enabled_elements = [el for el in elements if el.is_enabled(timeout=50)]
                       if enabled_elements:
                            target_element = random.choice(enabled_elements); found_type = 'fallback'; break
                  except (PlaywrightError, TimeoutError): continue
                  except Exception as e: error_logger.warning(f"Error finding fallback '{fb_selector}': {e}"); continue

        # Move mouse if a target was found
        if target_element:
            try:
                target_box = target_element.bounding_box(timeout=500)
                if target_box and target_box['width'] > 0 and target_box['height'] > 0:
                    # Move towards center of element
                    target_x = target_box['x'] + target_box['width'] * random.uniform(0.4, 0.6)
                    target_y = target_box['y'] + target_box['height'] * random.uniform(0.4, 0.6)
                    # Add slight random offset to avoid hitting exact center every time
                    target_x += random.uniform(-min(10, target_box['width']*0.1), min(10, target_box['width']*0.1))
                    target_y += random.uniform(-min(8, target_box['height']*0.1), min(8, target_box['height']*0.1))

                    logging.debug(f"Browser {browser_id+1}: Moving mouse towards element ({found_type}).")
                    self.human_mouse_move(target_x, target_y, steps=random.randint(12, 28))
                    self.page.wait_for_timeout(random.uniform(300, 900)) # Pause while hovering
                    # Optional: Tiny jiggle after hover
                    # if random.random() < 0.5:
                    #     self.human_mouse_move(target_x + random.uniform(-4, 4), target_y + random.uniform(-3, 3), steps=random.randint(3, 7), min_delay_ms=30, max_delay_ms=80)
                    #     self.page.wait_for_timeout(random.uniform(100, 400))

            except (PlaywrightError, TimeoutError) as e:
                if "Target page, context or browser has been closed" not in str(e):
                    error_logger.warning(f"Browser {browser_id+1}: Error getting bbox or moving mouse to target: {e}")
            except Exception as e: error_logger.exception(f"Browser {browser_id+1}: Unexpected error during targeted mouse movement:")
        # else: No target found, mouse doesn't move explicitly this cycle

class NextPageNavigator:
    """Handles navigation to the next page using robust selectors/text."""

    def __init__(self, page: Page):
        self.page = page
        self.next_page_selectors = settings.get("NEXT_PAGE_SELECTORS", [])
        self.next_page_text_fallback = settings.get("NEXT_PAGE_TEXT_FALLBACK", [])

    def find_next_page_element(self) -> Optional[Locator]:
        if not self.page or self.page.is_closed(): return None
        search_timeout = 350 # Quick check
        try:
            # Try selectors first - find all visible & enabled, choose one
            visible_enabled_elements = []
            for selector in self.next_page_selectors:
                if self.page.is_closed(): return None
                try:
                    elements = self.page.locator(selector).locator('visible=true').all(timeout=search_timeout)
                    for element in elements:
                        try:
                             if element.is_enabled(timeout=50): visible_enabled_elements.append(element)
                        except (PlaywrightError, TimeoutError): continue
                except (PlaywrightError, TimeoutError): continue
            if visible_enabled_elements:
                 logging.info(f"Found {len(visible_enabled_elements)} next page candidates via selectors.")
                 # Prioritize elements lower down the page? Choose randomly for now.
                 return random.choice(visible_enabled_elements)

            # Try text fallback (links/buttons) if selectors fail
            visible_enabled_elements_text = []
            for text in self.next_page_text_fallback:
                if self.page.is_closed(): return None
                try:
                    # Regex: Case-insensitive, exact match or word boundary
                    pattern = f"(^\\s*{re.escape(text)}\\s*$|\\b{re.escape(text)}\\b)"
                    # Combine link and button locator, check visibility/enabled
                    elements = self.page.locator(f'a:text-matches("{pattern}", "i"):visible, button:text-matches("{pattern}", "i"):visible').all(timeout=search_timeout)
                    for element in elements:
                         try:
                            if element.is_enabled(timeout=50): visible_enabled_elements_text.append(element)
                         except (PlaywrightError, TimeoutError): continue
                except (PlaywrightError, TimeoutError): continue
            if visible_enabled_elements_text:
                 logging.info(f"Found {len(visible_enabled_elements_text)} next page candidates via text fallback.")
                 return random.choice(visible_enabled_elements_text)

        except Exception as e:
            if "Target page, context or browser has been closed" not in str(e):
                error_logger.exception(f"Error finding next page element:")
        return None

    def navigate_next_page(self, browser_id, update_signal) -> bool:
        if not settings.get("IMPRESSION_ENABLED", False) or not self.page or self.page.is_closed(): return False
        if random.random() < settings.get("SKIP_ACTION_PROBABILITY", 0.05): return False

        try:
            next_page_element = self.find_next_page_element()
            if next_page_element:
                element_text = "N/A"
                try: element_text = (next_page_element.text_content(timeout=200) or "").strip()[:30]
                except: pass
                update_signal.emit(f"Browser {browser_id+1}: Found next page link/button (Text: '{element_text}...').", browser_id)

                # Hover before click
                try:
                    box = next_page_element.bounding_box(timeout=500)
                    if box:
                         tm = TextSelectionManager(self.page)
                         tm.human_mouse_move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2, steps=random.randint(8, 15))
                         self.page.wait_for_timeout(random.uniform(200, 600))
                except (PlaywrightError, TimeoutError, Exception) as hover_err: logging.warning(f"Hover next page failed: {hover_err}")
                if self.page.is_closed(): return False # Check if closed after hover attempt

                update_signal.emit(f"Browser {browser_id+1}: Clicking next page...", browser_id)
                try:
                    # Use expect_navigation for robustness
                    with self.page.expect_navigation(timeout=45000, wait_until='domcontentloaded') as nav_info:
                         next_page_element.click(timeout=15000, delay=random.uniform(50, 120))
                    # Check response status if available
                    response = nav_info.value
                    if response and response.ok:
                        update_signal.emit(f"Browser {browser_id+1}: Navigated to next page (Status {response.status}). New URL: {self.page.url[:80]}...", browser_id)
                        self.page.wait_for_timeout(random.uniform(700, 1800)) # Pause after load
                        return True
                    elif response:
                         update_signal.emit(f"Browser {browser_id+1}: Next page navigation resulted in status {response.status}. Treating as failure.", browser_id)
                         return False
                    else: # Should not happen if expect_navigation succeeded, but safety check
                         update_signal.emit(f"Browser {browser_id+1}: Next page navigation finished but no response object received.", browser_id)
                         self.page.wait_for_timeout(random.uniform(700, 1800)) # Pause anyway
                         return True # Assume success if no error?

                except (TimeoutError, PlaywrightError) as click_nav_err:
                     # Handle potential closure during click/nav
                     if "Target page, context or browser has been closed" in str(click_nav_err):
                         update_signal.emit(f"Browser {browser_id+1}: Page closed during next page click/navigation.", browser_id)
                     else:
                         update_signal.emit(f"Browser {browser_id+1}: Timeout/Error during/after next page click: {click_nav_err}", browser_id)
                         error_logger.error(f"Browser {browser_id+1}: Next page click/nav error: {click_nav_err}")
                         try: self.page.screenshot(path=f"logs/next_page_fail_{browser_id+1}_{int(time.time())}.png")
                         except: pass
                     return False # Indicate failure
            else:
                update_signal.emit(f"Browser {browser_id+1}: Next page element not found.", browser_id)
                return False
        except Exception as e:
            if "Target page, context or browser has been closed" not in str(e):
                update_signal.emit(f"Browser {browser_id+1}: Unexpected error during next page nav: {e}", browser_id)
                error_logger.exception(f"Browser {browser_id+1}: Unexpected next page navigation error:")
            return False

class AdClickManager:
    """Handles ad detection and clicking with robustness."""

    def __init__(self, page: Page):
        self.page = page
        self.ad_selectors = settings.get("AD_SELECTORS", [])
        self.ad_click_probability = settings.get("AD_CLICK_PROBABILITY", 0.1)

    def find_ad_element(self) -> Optional[Locator]:
        if not self.page or self.page.is_closed(): return None
        potential_ads_locators = []
        search_timeout = 300 # Quick check for ads
        try:
            # Check Selectors (Visible elements matching selectors)
            for selector in self.ad_selectors:
                if self.page.is_closed(): return None
                try:
                    elements = self.page.locator(selector).locator('visible=true').all(timeout=search_timeout)
                    for element in elements:
                        try: # Check basic size and add if reasonable
                             bb = element.bounding_box(timeout=50)
                             if bb and bb['width'] > 10 and bb['height'] > 10:
                                 potential_ads_locators.append(element)
                        except (PlaywrightError, TimeoutError): continue # Ignore bbox errors
                except (PlaywrightError, TimeoutError): continue # Ignore if selector times out quickly
                except Exception as e: error_logger.warning(f"Error checking ad selector '{selector}': {e}")

            # Check Iframes (more reliable heuristic for typical ad networks)
            try:
                if self.page.is_closed(): return None
                all_frames = self.page.frames
                for frame in all_frames:
                    if frame.is_detached(): continue

                    frame_loc = None
                    try:
                        # Try to get locator using frame object directly (more reliable in some cases)
                        frame_loc = frame.locator(':root').first # Get locator for the frame's document element
                        # Get the corresponding iframe element in the parent page
                        iframe_element = self.page.locator(f'iframe[src="{frame.url}"], iframe[name="{frame.name()}"]').first
                        if iframe_element.count() == 0: continue # Could not find the iframe tag in parent
                        frame_loc = iframe_element # Use locator for the iframe tag itself for clicks/bbox

                    except (PlaywrightError, TimeoutError):
                        continue # Failed to get locator for this frame

                    is_likely_ad = False
                    try:
                        # Check frame URL first
                        frame_url_lower = frame.url.lower()
                        if any(s in frame_url_lower for s in ['ads.', 'doubleclick.net', 'googleadservices', 'googlesyndication', 'adservice', 'yieldmo', 'adnxs']):
                             is_likely_ad = True

                        # Check attributes of the iframe tag itself using locator
                        attrs = {'id': '', 'name': '', 'aria-label': '', 'title': ''}
                        try: attrs['id'] = frame_loc.get_attribute('id', timeout=50) or ''
                        except: pass
                        try: attrs['name'] = frame_loc.get_attribute('name', timeout=50) or ''
                        except: pass
                        try: attrs['aria-label'] = frame_loc.get_attribute('aria-label', timeout=50) or ''
                        except: pass
                        try: attrs['title'] = frame_loc.get_attribute('title', timeout=50) or ''
                        except: pass

                        if any(s in attrs['id'].lower() for s in ['google_ads', 'ad_frame', 'google_ads', 'google_ad']) or \
                           any(s in attrs['name'].lower() for s in ['google_ads', 'ad_frame', 'google_ads', 'google_ad']) or \
                           any(s in attrs['aria-label'].lower() for s in ['ad', 'advertisement', 'google ad']) or \
                           any(s in attrs['title'].lower() for s in ['ad', 'advertisement', 'google ad']): is_likely_ad = True

                        # Check if frame itself contains typical ad markers (quick check)
                        if not is_likely_ad:
                            if frame.locator('body[id*="google_ads_iframe"], body[class*="adsbygoogle"]', timeout=50).count() > 0:
                                is_likely_ad = True

                        if is_likely_ad:
                            # Check if iframe tag is visible and has size
                            bb = frame_loc.bounding_box(timeout=100)
                            if bb and bb['width'] > 10 and bb['height'] > 10 and frame_loc.is_visible(timeout=50):
                                potential_ads_locators.append(frame_loc)
                    except (PlaywrightError, TimeoutError): continue # Ignore errors getting attributes/bbox for one iframe
            except (PlaywrightError, TimeoutError): pass # Ignore errors finding frames altogether
            except Exception as e: error_logger.warning(f"Error checking ad iframes: {e}")

            # --- Deduplicate and Choose ---
            unique_ads = {} # Use element handle or bounding box as key for rough uniqueness
            for loc in potential_ads_locators:
                try:
                    bb = loc.bounding_box(timeout=100)
                    if bb: unique_ads[tuple(bb.values())] = loc
                    # else: Use element handle? Can be complex. Bbox is simpler.
                except: continue

            final_ad_list = list(unique_ads.values())
            if final_ad_list:
                logging.info(f"Found {len(final_ad_list)} potential visible ad element(s).")
                # Prioritize larger ads or just random? Random is simpler.
                return random.choice(final_ad_list)
            else:
                 logging.debug("No likely ad elements found."); return None
        except Exception as e:
            if "Target page, context or browser has been closed" not in str(e):
                error_logger.exception(f"Error finding ad element:");
            return None

    def click_ad(self, browser_id, update_signal) -> bool:
        if not settings.get("AD_CLICK_ENABLED", False) or not self.page or self.page.is_closed(): return False
        if random.random() >= self.ad_click_probability: return False
        if random.random() < settings.get("SKIP_ACTION_PROBABILITY", 0.05): return False

        try:
            ad_element = self.find_ad_element()
            if ad_element:
                element_info = ""
                try: element_info = f"Tag: {ad_element.evaluate('el => el.tagName', timeout=100)}"
                except: pass
                update_signal.emit(f"Browser {browser_id+1}: Found potential ad ({element_info}), preparing to click.", browser_id)

                # Hover before click
                try:
                    box = ad_element.bounding_box(timeout=500)
                    if box:
                        tm = TextSelectionManager(self.page); tm.human_mouse_move(box['x'] + box['width']/2, box['y'] + box['height']/2, steps=random.randint(8, 16))
                        self.page.wait_for_timeout(random.uniform(300, 700))
                except (PlaywrightError, TimeoutError, Exception) as hover_err: logging.warning(f"Hover ad failed: {hover_err}")
                if self.page.is_closed(): return False # Check again after hover

                update_signal.emit(f"Browser {browser_id+1}: Clicking ad...", browser_id)
                new_page = None
                try:
                    # Use expect_page to handle popups/new tabs robustly
                    with self.page.context.expect_page(timeout=25000) as new_page_info: # Increased timeout
                         # Click the ad element
                         ad_element.click(timeout=15000, delay=random.uniform(50, 150))
                         # Wait slightly after click to ensure popup event triggers if it's slow
                         try: self.page.wait_for_timeout(1000)
                         except PlaywrightError: pass # Ignore if page closes during wait

                    # --- New Page Opened ---
                    new_page = new_page_info.value
                    update_signal.emit(f"Browser {browser_id+1}: Ad likely opened new tab/popup.", browser_id)
                    try:
                        # Wait for the new page to reach a stable state
                        new_page.wait_for_load_state('domcontentloaded', timeout=30000)
                        ad_url = new_page.url
                        update_signal.emit(f"Browser {browser_id+1}: Ad Tab URL: {ad_url[:80]}...", browser_id)

                        # Simulate interaction on the ad page
                        ad_page_wait = random.uniform(7, 18) # Spend longer on ad page
                        update_signal.emit(f"Browser {browser_id+1}: Waiting on ad page for {ad_page_wait:.1f}s...", browser_id)
                        start_wait = time.time()
                        while time.time() - start_wait < (ad_page_wait * 1000):
                             if new_page.is_closed(): break
                             # Simple random scroll on ad page
                             if random.random() < 0.3:
                                 try: new_page.mouse.wheel(0, random.randint(200, 800) * random.choice([-1, 1]))
                                 except: pass
                             new_page.wait_for_timeout(random.uniform(500, 1500))

                    except (PlaywrightError, TimeoutError) as ad_page_err:
                         if "Target page, context or browser has been closed" not in str(ad_page_err):
                            update_signal.emit(f"Browser {browser_id+1}: Error/Timeout interacting with ad page: {ad_page_err}", browser_id)
                    finally:
                        if new_page and not new_page.is_closed():
                             try: new_page.close(); update_signal.emit(f"Browser {browser_id+1}: Closed ad tab.", browser_id)
                             except PlaywrightError as close_err: error_logger.warning(f"Could not close ad tab: {close_err}")
                    return True # Success = interacted with new tab

                except TimeoutError: # No new page detected within timeout
                    update_signal.emit(f"Browser {browser_id+1}: Ad click - no new tab/popup detected. Assuming same tab nav or failed click.", browser_id)
                    # Wait a bit as if navigated in same tab, then maybe go back
                    try:
                        self.page.wait_for_load_state('domcontentloaded', timeout=15000)
                        update_signal.emit(f"Browser {browser_id+1}: Waited for potential same-tab nav. Current URL: {self.page.url[:80]}...", browser_id)
                        self.page.wait_for_timeout(random.uniform(4000, 9000)) # Stay longer
                        if random.random() < 0.7: # Higher chance to navigate back
                            try:
                                update_signal.emit(f"Browser {browser_id+1}: Attempting to navigate back...", browser_id)
                                self.page.go_back(timeout=20000, wait_until="domcontentloaded")
                                self.page.wait_for_timeout(random.uniform(500, 1500))
                                update_signal.emit(f"Browser {browser_id+1}: Navigated back.", browser_id)
                            except PlaywrightError as back_err:
                                 if "Target page, context or browser has been closed" not in str(back_err):
                                     error_logger.warning(f"Could not go back after assumed same-page ad nav: {back_err}")
                    except (PlaywrightError, TimeoutError) as same_page_wait_err:
                         if "Target page, context or browser has been closed" not in str(same_page_wait_err):
                            update_signal.emit(f"Browser {browser_id+1}: Error waiting after same-page ad click assumption: {same_page_wait_err}", browser_id)
                    return True # Ad click *attempted*

                except PlaywrightError as click_err:
                    if "Target page, context or browser has been closed" not in str(click_err):
                        error_logger.error(f"Browser {browser_id+1}: Error performing ad click action: {click_err}")
                        update_signal.emit(f"Browser {browser_id+1}: Error during ad click: {click_err}", browser_id)
                    return False # Click itself failed
            else:
                 return False # No ad found

        except Exception as e:
            if "Target page, context or browser has been closed" not in str(e):
                update_signal.emit(f"Browser {browser_id+1}: Unexpected error during ad click logic: {e}", browser_id)
                error_logger.exception(f"Browser {browser_id+1}: Unexpected ad click error:")
            return False


class FormFiller:
    """Fills out forms with realistic typos and corrections."""
    def __init__(self, page: Page):
        self.page = page
        self.typo_probability = 0.07
        self.correction_probability = 0.90

    def fill_form(self, browser_id, update_signal):
        if not settings.get("FORM_FILL_ENABLED", False) or not self.page or self.page.is_closed(): return

        update_signal.emit(f"Browser {browser_id+1}: Searching for form fields...", browser_id)
        try:
            # Selector for common fillable fields, excluding buttons, hidden, etc.
            field_selectors = """
                input:visible:enabled:not([type='submit']):not([type='button']):not([type='hidden']):not([type='reset']):not([type='checkbox']):not([type='radio']):not([type='file']):not([type='image']):not([readonly]),
                textarea:visible:enabled:not([readonly]),
                select:visible:enabled:not([readonly])
            """
            # Remove newlines for locator
            field_selectors = ",".join(s.strip() for s in field_selectors.splitlines())

            form_fields = self.page.locator(field_selectors).all(timeout=5000)

            if not form_fields:
                update_signal.emit(f"Browser {browser_id+1}: No suitable form fields found.", browser_id)
                return

            update_signal.emit(f"Browser {browser_id+1}: Found {len(form_fields)} potential fields.", browser_id)
            filled_count = 0
            # Try to fill a random subset of fields
            fields_to_fill = random.sample(form_fields, k=min(len(form_fields), random.randint(1, 4))) # Fill 1-4 fields

            for field in fields_to_fill:
                if not self.page or self.page.is_closed(): break
                try:
                    # Re-check visibility/enabled status just before interaction
                    if not field.is_visible(timeout=500) or not field.is_enabled(timeout=500):
                        continue

                    tag_name = field.evaluate("el => el.tagName.toLowerCase()")
                    field_type = field.get_attribute("type", timeout=50) or tag_name
                    # Try to get associated label text for better context
                    label_text = ""
                    field_id = field.get_attribute("id", timeout=50)
                    try:
                        if field_id:
                             label_loc = self.page.locator(f'label[for="{field_id}"]').first
                             if label_loc.is_visible(timeout=50): label_text = label_loc.text_content(timeout=100) or ""
                    except: pass
                    # Get other potential labels/hints
                    field_label_attr = field.get_attribute("aria-label", timeout=50) or field.get_attribute("placeholder", timeout=50) or field.get_attribute("name", timeout=50) or ""
                    full_label_info = f"{label_text} {field_label_attr}".strip() or f"({tag_name} / {field_type})"

                    logging.debug(f"Browser {browser_id+1}: Attempting field: '{full_label_info}' (Type: {field_type})")

                    if tag_name == "select":
                         options_loc = field.locator("option")
                         options_count = options_loc.count()
                         if options_count > 1: # More than one option available
                             try:
                                 # Get values/text of options (limit count for performance?)
                                 option_data = []
                                 for i in range(min(options_count, 20)): # Check first 20 options
                                      opt = options_loc.nth(i)
                                      is_disabled = opt.is_disabled(timeout=50)
                                      val = opt.get_attribute("value", timeout=50)
                                      txt = opt.inner_text(timeout=50)
                                      if not is_disabled and (val or txt): # Must have value or text
                                           option_data.append({'value': val, 'text': txt})

                                 # Filter out placeholder-like options (empty value AND generic text)
                                 valid_options = [
                                     opt for opt in option_data
                                     if not (not opt['value'] and any(placeholder in opt['text'].lower() for placeholder in ["select", "choose", "--"]))
                                 ]
                                 if not valid_options: valid_options = option_data # Use all if filtering removed everything

                                 if valid_options:
                                     option_to_select = random.choice(valid_options)
                                     value_attr = option_to_select['value']
                                     option_text = option_to_select['text']

                                     # Select by value if non-empty, otherwise by label/text
                                     if value_attr:
                                         field.select_option(value=value_attr, timeout=5000)
                                         update_signal.emit(f"Browser {browser_id+1}: Selected '{option_text[:30]}...' (value: {value_attr}) in dropdown '{full_label_info[:40]}...'.", browser_id)
                                     else: # Select by text/label
                                         field.select_option(label=option_text, timeout=5000)
                                         update_signal.emit(f"Browser {browser_id+1}: Selected '{option_text[:30]}...' (by label) in dropdown '{full_label_info[:40]}...'.", browser_id)
                                     filled_count += 1
                                 else: logging.debug(f"Skipping select '{full_label_info}', no valid/non-placeholder options found.")

                             except (PlaywrightError, TimeoutError) as select_err: error_logger.warning(f"Could not select option in '{full_label_info}': {select_err}")
                    else: # Input or Textarea
                        value = self._generate_input_value(field_type, full_label_info) # Pass label for context
                        if value:
                             self._type_with_typos(field, value)
                             filled_count += 1

                    self.page.wait_for_timeout(random.uniform(400, 1200)) # Pause after interacting with a field

                except (PlaywrightError, TimeoutError) as e:
                     if "Target page, context or browser has been closed" not in str(e):
                        error_logger.warning(f"Could not interact with form field '{full_label_info}': {e}")
                except Exception as e: error_logger.exception(f"Unexpected error filling field '{full_label_info}':")

            if filled_count > 0: update_signal.emit(f"Browser {browser_id+1}: Finished attempting to fill {filled_count} form fields.", browser_id)

        except PlaywrightError as e:
            if "Target page, context or browser has been closed" not in str(e):
                 error_logger.error(f"PlaywrightError during form field search: {e}")
        except Exception as e: error_logger.exception(f"Unexpected error during form filling process:")


    def _generate_input_value(self, field_type: str, field_label: str) -> str:
        field_type = field_type.lower(); field_label = field_label.lower()
        # More robust generation with common first/last names etc.
        first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah", "Charles", "Karen", "Thomas", "Nancy"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Martin", "Lee"]
        domains = ["testmail.com", "mailinator.com", "fakeinbox.net", "sample.org", "inbox.cc", "mail.com", "email.com"]
        cities = ["New York", "London", "Paris", "Tokyo", "Sydney", "Berlin", "Los Angeles", "Chicago", "Houston", "Toronto", "Mumbai", "Sao Paulo"]
        subjects = ["Inquiry", "Feedback", "Support Request", "Question", "Test Subject", "Regarding Your Product", "Website Issue", "Information Request"]
        search_terms = ["product reviews", "best deals", "how to fix", "local events", "python tutorials", "ai news", "travel ideas", "recipe", "weather forecast", "nearby restaurants", "online courses", "gift ideas"]

        # Use label hints first
        if "email" in field_type or "email" in field_label:
            name = random.choice(first_names).lower() + random.choice(['.', '', '_']) + random.choice(last_names).lower()
            return f"{name}{random.randint(10,999)}@{random.choice(domains)}"
        elif "password" in field_type or "password" in field_label:
            prefix = random.choice(["PassWord", "Secret", "MyP@ss", "LoginNow", "Test!Acc", "Secure1"])
            return f"{prefix}{random.randint(1000,99999)}!"
        elif "tel" in field_type or "phone" in field_label or "mobile" in field_label:
            return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"
        elif "search" in field_type or "query" in field_label or "keyword" in field_label:
            return random.choice(search_terms) + " " + random.choice(["", "tips", "online", str(datetime.date.today().year), "guide", "comparison"])
        elif "first" in field_label or "given" in field_label: return random.choice(first_names)
        elif "last" in field_label or "surname" in field_label or "family" in field_label: return random.choice(last_names)
        elif "name" in field_label or field_type == "text" and "name" in field_label: # Catch text fields labeled name
            if "user" in field_label or "login" in field_label: return random.choice(first_names) + str(random.randint(100,999))
            return f"{random.choice(first_names)} {random.choice(last_names)}" # Full name
        elif "zip" in field_label or "postal" in field_label: return str(random.randint(10000, 99999))
        elif "city" in field_label: return random.choice(cities)
        elif "subject" in field_label: return random.choice(subjects)
        elif field_type == "url" or "website" in field_label: return f"http://www.{random.choice(['example','test','sample','demo'])}{random.choice(['.com','.org','.net'])}"
        elif field_type == "number": return str(random.randint(1, 100))
        elif field_type == "date": return datetime.date.today().strftime('%Y-%m-%d')
        elif field_type == "textarea" or "message" in field_label or "comment" in field_label:
            sentences = ["Sample comment.", "Looking forward to hearing from you.", "Interesting point, could you elaborate?", "Could you provide more details please?", "Thank you for the information provided.", "This is a test message generated by automation.", "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "Just testing the form submission functionality.", "Seems to be working as expected."]
            return " ".join(random.sample(sentences, random.randint(1, 3)))
        else: # Default text input
            words = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat".split()
            return " ".join(random.sample(words, random.randint(3, 8)))


    def _type_with_typos(self, field: Locator, value: str):
        try:
            # field.click(timeout=5000); # Click to focus - fill does this implicitly
            # Clear field before typing new value
            field.fill("")
            self.page.wait_for_timeout(random.uniform(50, 150))

            current_value_typed = ""
            for i, char in enumerate(value):
                if not self.page or self.page.is_closed(): return
                # Simulate typo
                should_typo = random.random() < self.typo_probability
                if should_typo:
                    wrong_char = random.choice("abcdefghijklmnopqrstuvwxyz., /!?") # Simple typo chars + common symbols
                    if wrong_char != char:
                        field.press(wrong_char)
                        current_value_typed += wrong_char
                        self.page.wait_for_timeout(random.uniform(60, 180)) # Pause after typo

                        # Simulate correction
                        should_correct = random.random() < self.correction_probability
                        if should_correct:
                            field.press("Backspace")
                            current_value_typed = current_value_typed[:-1]
                            self.page.wait_for_timeout(random.uniform(120, 400)) # Pause after correction
                            # Press the correct char now
                            field.press(char)
                            current_value_typed += char
                            self.page.wait_for_timeout(random.uniform(40, 120)) # Delay after corrected char
                        else: # Typo not corrected, just press the intended char after it
                            field.press(char)
                            current_value_typed += char
                            self.page.wait_for_timeout(random.uniform(40, 120))
                    else: # Typo char was same as correct, just type normally
                        field.press(char)
                        current_value_typed += char
                        self.page.wait_for_timeout(random.uniform(40, 120))
                else: # No typo
                    field.press(char)
                    current_value_typed += char
                    self.page.wait_for_timeout(random.uniform(40, 120)) # Normal typing delay

            # Verify final value if needed (debug only, can be slow)
            # final_value = field.input_value(timeout=1000)
            # if final_value != value and final_value != current_value_typed: # Compare against potentially uncorrected typed value
            #     logging.warning(f"Field value mismatch after typing. Expected '{value}', Typed '{current_value_typed}', Final '{final_value}'")

        except (PlaywrightError, TimeoutError) as e:
             if "Target page, context or browser has been closed" not in str(e):
                 error_logger.warning(f"Error during typing with typos: {e}")
        except Exception as e: error_logger.exception(f"Unexpected typing error:")


class LicenseManager:
    """Verifies the license key (Basic Example)."""
    def __init__(self):
        self.expiration_date: Optional[datetime.date] = None
        self._activated = False
        self._activated_key = None

    def verify_license(self, license_key: str) -> Tuple[bool, Optional[str]]:
        """Verifies the license key and sets the expiration date."""
        # Simple static key check - REPLACE with actual validation logic
        valid_keys = {
            "AladdinBot_ValidKey_2024": 30,
            "HamzaAkmal_Special": 365,
            "TEST_KEY_7D": 7,
            "AladdinBot_PRO_YEAR": 366,
            "AladdinBot_Final_2025": 390, # Added key
        }

        if license_key in valid_keys:
            days = valid_keys[license_key]
            self.expiration_date = datetime.date.today() + datetime.timedelta(days=days)
            self._activated = True; self._activated_key = license_key
            logging.info(f"License key '{license_key}' accepted. Expires: {self.expiration_date}")
            return True, f"Activated - Expires: {self.expiration_date.strftime('%Y-%m-%d')}"
        else:
            self._activated = False; self.expiration_date = None; self._activated_key = None
            error_logger.error(f"Invalid license key entered: {license_key}")
            return False, "Invalid or unknown license key."

    def is_license_valid(self) -> Tuple[bool, Optional[str]]:
        """Checks if the license is currently activated and not expired."""
        if not self._activated or self.expiration_date is None or self._activated_key is None:
            return False, "License not activated."
        if datetime.date.today() > self.expiration_date:
            return False, f"License expired on {self.expiration_date.strftime('%Y-%m-%d')}."
        else:
            remaining = (self.expiration_date - datetime.date.today()).days
            return True, f"Active - Expires: {self.expiration_date.strftime('%Y-%m-%d')} ({remaining} days left)"

def random_delay(min_seconds: Optional[float] = None, max_seconds: Optional[float] = None):
    """Pauses execution for a random time using configured defaults."""
    min_s = min_seconds if min_seconds is not None else settings.get("MIN_DELAY", 1.0)
    max_s = max_seconds if max_seconds is not None else settings.get("MAX_DELAY", 3.0)
    min_s = max(0.1, min_s); max_s = max(min_s + 0.1, max_s) # Ensure min > 0 and max > min
    delay = random.uniform(min_s, max_s)
    time.sleep(delay)



# --- GUI Components and Logic ---

class CenteredLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

class BoldLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        font = self.font(); font.setBold(True); self.setFont(font)

class ProxyDelegate(QStyledItemDelegate):
    """Delegate for the proxy list to show proxy status visually."""
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        # data = index.data(Qt.ItemDataRole.UserRole) # Tuple: (address, type, status)
        # Robustly get data from model index
        model_index = index.model().index(index.row(), index.column())
        data = model_index.data(Qt.ItemDataRole.UserRole)

        status = "unchecked"
        if isinstance(data, tuple) and len(data) > 2:
            status = data[2]

        # Define colors
        valid_color = QColor("#4CAF50") # Green
        invalid_color = QColor("#F44336") # Red
        unchecked_color = QColor("#FF9800") # Orange for unchecked
        checking_color = QColor("#2196F3") # Blue for checking in progress? (or just use unchecked)

        # Choose color based on status
        color = {"valid": valid_color, "invalid": invalid_color, "checking": checking_color}.get(status, unchecked_color)

        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing); painter.setBrush(color); painter.setPen(Qt.PenStyle.NoPen)
        # Draw circle indicator
        rect = option.rect; diameter = min(rect.height() - 6, 10)
        x = rect.left() + 7; # Position circle on the left
        y = rect.top() + (rect.height() - diameter) / 2
        painter.drawEllipse(int(x), int(y), diameter, diameter)
        painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index); return QSize(size.width() + 25, size.height())

class BotThread(QThread):
    """Thread for running the bot logic in the background."""
    update_signal = pyqtSignal(str, int)  # Message, browser_id (-1 for general)
    finished_signal = pyqtSignal(int)     # browser_id
    error_signal = pyqtSignal(str, int)   # Error message, browser_id

    def __init__(self, browser_id: int, urls: List[str], config: Dict,
                 proxy: Optional[Tuple[str, str]], # (address, type)
                 user_agent_manager: FingerprintManager):
        super().__init__()
        self.browser_id = browser_id
        self.urls = urls
        self.config = config # Use passed config snapshot
        self.proxy_address, self.proxy_type = proxy if proxy else (None, None)
        self.user_agent_manager = user_agent_manager
        self._should_stop = False
        self._lock = threading.Lock()
        self.bot_instance: Optional[HumanTrafficBot] = None

    @property
    def should_stop(self):
        with self._lock: return self._should_stop

    def run(self):
        thread_name = f"BotThread-{self.browser_id + 1}"; threading.current_thread().name = thread_name
        logging.info(f"{thread_name} started.")
        try:
            with sync_playwright() as playwright:
                self.bot_instance = HumanTrafficBot(playwright, self.urls, self.config, self.proxy_address, self.proxy_type, self.user_agent_manager)
                self.bot_instance.run(self, self.update_signal, self.browser_id)
        except Exception as e:
            error_msg = f"Critical unhandled error in {thread_name}"
            error_logger.exception(error_msg)
            # Ensure error signal is emitted even if playwright context fails
            self.error_signal.emit(f"{error_msg}: {e}", self.browser_id)
        finally:
            log_msg = f"{thread_name} finished." + (" (Stop requested)" if self.should_stop else "")
            logging.info(log_msg); self.finished_signal.emit(self.browser_id)

    def stop(self):
        logging.info(f"Stop signal received for BotThread-{self.browser_id + 1}")
        with self._lock: self._should_stop = True
        # Attempt to close the browser associated with this thread via the bot instance
        # This helps speed up shutdown, especially if the bot is stuck in a long wait
        bot = self.bot_instance
        if bot and bot.browser_manager:
            logging.info(f"Attempting to close browser for Bot {self.browser_id + 1} due to stop signal.")
            # Run close in a separate thread to avoid potential deadlocks if called from signal handler?
            # No, direct call should be okay if careful.
            try:
                # Use a short timeout within close to prevent getting stuck here
                if bot.browser_manager.context:
                     bot.browser_manager.context.close(reason="Bot stopped by user") # Try closing context with reason
                elif bot.browser_manager.browser:
                     bot.browser_manager.browser.close(reason="Bot stopped by user") # Try closing browser if context gone
                # bot.browser_manager.close_browser() # This might wait too long
            except Exception as e:
                logging.error(f"Error during explicit browser close on stop signal for Bot {self.browser_id + 1}: {e}")


class HumanTrafficBot:
    """Main bot class orchestrating browser actions."""

    def __init__(self, playwright: Playwright, urls: List[str], config: Dict,
                 proxy_address: Optional[str], proxy_type: Optional[str],
                 user_agent_manager: FingerprintManager):
        self.config = config
        self.urls = urls
        self.user_agent_manager = user_agent_manager
        self.gemini_api_key = config.get("GEMINI_API_KEY")
        self.playwright = playwright
        self.proxy_address = proxy_address
        self.proxy_type = proxy_type
        self.browser_manager: Optional[BrowserManager] = None
        self.scrolling_manager: Optional[ScrollingManager] = None
        self.text_selection_manager: Optional[TextSelectionManager] = None
        self.form_filler: Optional[FormFiller] = None
        self.next_page_navigator: Optional[NextPageNavigator] = None
        self.ad_click_manager: Optional[AdClickManager] = None
        self.load_behavior_settings()

    def load_behavior_settings(self):
        """Load behavior settings from the config dictionary."""
        self.form_fill_enabled = self.config.get("FORM_FILL_ENABLED", False)
        self.impression_enabled = self.config.get("IMPRESSION_ENABLED", False)
        self.ad_click_enabled = self.config.get("AD_CLICK_ENABLED", False)
        self.behavioral_states_enabled = self.config.get("ENABLE_BEHAVIORAL_STATES", True)
        self.skip_action_probability = self.config.get("SKIP_ACTION_PROBABILITY", 0.05)
        self.min_delay = self.config.get("MIN_DELAY", 1.0)
        self.max_delay = self.config.get("MAX_DELAY", 3.0)

    def run(self, bot_thread: BotThread, update_signal: pyqtSignal, browser_id: int):
        current_behavior_state = "Scanning"; state_duration = random.uniform(15, 40); state_timer_start = time.time()
        page_instance = None # Track page instance for safety checks
        try:
            self.browser_manager = BrowserManager( # Initialize BrowserManager here
                self.playwright, self.proxy_address, self.proxy_type,
                self.config.get("HEADLESS", True), self.config.get("CHROMIUM_BLUE_ENABLED", False),
                self.config.get("CHROMIUM_BLUE_PATH", ""), self.config.get("CHROMIUM_BLUE_ARGS", "")
            )

            for i, url in enumerate(self.urls):
                if bot_thread.should_stop:
                    update_signal.emit(f"Browser {browser_id+1}: Stop requested before URL {i+1}.", browser_id)
                    break
                update_signal.emit(f"Browser {browser_id+1}: Processing URL {i+1}/{len(self.urls)}: {url}", browser_id)

                # Get Fingerprint (handles Generate & Use internally)
                user_agent, viewport_size, profile = self.user_agent_manager.get_fingerprint(browser_id, update_signal)

                # Start Browser
                try:
                    # Ensure previous instance is fully closed before starting new one
                    if self.browser_manager.page or self.browser_manager.context or self.browser_manager.browser:
                        update_signal.emit(f"Browser {browser_id+1}: Closing previous browser instance before new URL...", browser_id)
                        self.browser_manager.close_browser()
                        # Short pause might help ensure resources are released
                        try: bot_thread.msleep(200) # Use QThread's msleep
                        except: pass

                    self.browser_manager.start_browser(user_agent, viewport_size, profile)
                    if not self.browser_manager.page:
                        raise RuntimeError("Browser page object was not created after start_browser.")
                    page_instance = self.browser_manager.page # Store reference
                    update_signal.emit(f"Browser {browser_id+1}: Browser started. UA: {user_agent[:60]}...", browser_id)
                except Exception as start_err:
                     update_signal.emit(f"Browser {browser_id+1}: FATAL: Failed to start browser: {start_err}. Skipping URL.", browser_id)
                     error_logger.error(f"Browser {browser_id+1}: Start failed: {start_err}")
                     if self.browser_manager: self.browser_manager.close_browser() # Attempt cleanup
                     page_instance = None
                     # Signal error for this specific run/URL?
                     # bot_thread.error_signal.emit(f"Failed to start browser: {start_err}", browser_id) # Causes issues if thread finishes immediately after
                     continue # Skip to next URL

                # Initialize Page-Dependent Managers
                page = page_instance # Use the stored reference
                self.scrolling_manager = ScrollingManager(page)
                self.text_selection_manager = TextSelectionManager(page)
                self.next_page_navigator = NextPageNavigator(page)
                self.ad_click_manager = AdClickManager(page)
                self.form_filler = FormFiller(page) if self.form_fill_enabled else None

                # Navigate
                try:
                    self.browser_manager.navigate_to(url)
                    update_signal.emit(f"Browser {browser_id+1}: Navigation to {url} complete.", browser_id)
                except Exception as nav_err:
                    update_signal.emit(f"Browser {browser_id+1}: Failed to navigate to {url}: {nav_err}. Skipping interactions.", browser_id)
                    if self.browser_manager: self.browser_manager.take_screenshot(f"browser_{browser_id+1}_nav_fail_url_{i}.png")
                    page_instance = None # Page is likely unusable
                    continue # Go to next URL

                # --- Interaction Loop per Page ---
                interaction_start_time = time.time(); MAX_INTERACTION_TIME_PER_PAGE = random.randint(70, 140)
                page_interaction_count = 0; has_attempted_form_fill = False

                while time.time() - interaction_start_time < MAX_INTERACTION_TIME_PER_PAGE:
                    # Safety checks at loop start
                    if bot_thread.should_stop: break
                    # Check if page is still valid and attached
                    if not page_instance or page_instance.is_closed():
                        update_signal.emit(f"Browser {browser_id+1}: Page closed unexpectedly during interaction loop.", browser_id); break

                    page_interaction_count += 1

                    # Update Behavioral State
                    if self.behavioral_states_enabled and time.time() - state_timer_start > state_duration:
                        current_behavior_state = "Reading" if current_behavior_state == "Scanning" else "Scanning"
                        state_duration = random.uniform(8, 25) if current_behavior_state == "Reading" else random.uniform(15, 45)
                        state_timer_start = time.time()
                        update_signal.emit(f"Browser {browser_id+1}: State -> {current_behavior_state} (~{state_duration:.0f}s)", browser_id)

                    # Decide Action Order (slightly biased)
                    actions = ['scroll'] # Always scroll
                    if self.text_selection_manager: actions.append('mouse')
                    # Conditionally add actions based on settings and probability
                    if self.ad_click_manager and self.ad_click_enabled and random.random() < 0.5: actions.append('ad_click') # Less frequent attempt
                    if self.form_filler and not has_attempted_form_fill and random.random() < 0.3: actions.append('form_fill') # Less frequent attempt
                    if self.next_page_navigator and self.impression_enabled and random.random() < 0.6: actions.append('next_page') # Less frequent attempt
                    random.shuffle(actions)

                    action_performed_this_cycle = False
                    for action_type in actions:
                        # Re-check conditions before each action
                        if bot_thread.should_stop: break
                        if not page_instance or page_instance.is_closed(): break
                        if random.random() < self.skip_action_probability: continue

                        action_performed = False
                        try:
                            if action_type == 'scroll' and self.scrolling_manager:
                                scroll_params = {"min_scrolls": 2, "max_scrolls": 4, "duration_min": 700, "duration_max": 1800} # Reading
                                if current_behavior_state == "Scanning": scroll_params = {"min_scrolls": 3, "max_scrolls": 7, "duration_min": 400, "duration_max": 1200}
                                # Use Gemini scroll less frequently?
                                if self.gemini_api_key and random.random() < 0.4: self.scrolling_manager.gemini_scroll(browser_id, update_signal)
                                else: self.scrolling_manager.random_scroll(**scroll_params)
                                action_performed = True
                            elif action_type == 'mouse' and self.text_selection_manager:
                                if current_behavior_state == "Scanning" or random.random() < 0.4: # Less mouse when reading
                                     self.text_selection_manager.select_important_text(browser_id, update_signal); action_performed = True
                            elif action_type == 'ad_click' and self.ad_click_manager:
                                # click_ad already includes probability check internally
                                if self.ad_click_manager.click_ad(browser_id, update_signal):
                                    action_performed = True
                                    # Ad click might navigate or close page, re-check state after
                                    if not page_instance or page_instance.is_closed(): break
                            elif action_type == 'form_fill' and self.form_filler:
                                # Limit form fill attempts per page load
                                if not has_attempted_form_fill: # Only attempt once
                                     self.form_filler.fill_form(browser_id, update_signal); has_attempted_form_fill = True; action_performed = True
                            elif action_type == 'next_page' and self.next_page_navigator:
                                navigated = self.next_page_navigator.navigate_next_page(browser_id, update_signal)
                                if navigated:
                                     action_performed = True
                                     # Reset timers/state for new page load
                                     interaction_start_time = time.time(); state_timer_start = time.time(); page_interaction_count = 0; has_attempted_form_fill = False; MAX_INTERACTION_TIME_PER_PAGE = random.randint(70, 140)
                                     # Update page_instance IF browser_manager.page exists after nav
                                     page_instance = self.browser_manager.page if self.browser_manager else None
                                     if not page_instance or page_instance.is_closed():
                                         update_signal.emit(f"Browser {browser_id+1}: Page became invalid after next page navigation.", browser_id); break
                                     update_signal.emit(f"Browser {browser_id+1}: Resetting interaction timer for new page.", browser_id)
                                     # Re-init managers dependent on the page object? Assume they use the updated page ref.
                                     # Update manager references to be safe
                                     if page_instance:
                                         self.scrolling_manager = ScrollingManager(page_instance)
                                         self.text_selection_manager = TextSelectionManager(page_instance)
                                         self.next_page_navigator = NextPageNavigator(page_instance)
                                         self.ad_click_manager = AdClickManager(page_instance)
                                         self.form_filler = FormFiller(page_instance) if self.form_fill_enabled else None
                                     break # Restart interaction loop for new page immediately
                                else: # Next page not found/nav failed
                                     if self.impression_enabled: # If impressions mode is on, stop interaction for this base URL
                                         update_signal.emit(f"Browser {browser_id+1}: Next page not found/failed. Ending interaction for this URL.", browser_id)
                                         interaction_start_time = time.time() - MAX_INTERACTION_TIME_PER_PAGE - 1 # Force loop exit
                                         action_performed = True # The check itself was the action

                        except Exception as action_err:
                            if "Target page, context or browser has been closed" not in str(action_err):
                                error_logger.exception(f"Browser {browser_id+1}: Error during action '{action_type}':")
                                update_signal.emit(f"Browser {browser_id+1}: Error during '{action_type}': {action_err}", browser_id)
                            try: # Short pause after error
                                if page_instance and not page_instance.is_closed():
                                     page_instance.wait_for_timeout(random.uniform(1000, 2000))
                            except: pass

                        if action_performed:
                             action_performed_this_cycle = True
                             # Delay after performing an action
                             delay_multiplier = 1.5 if current_behavior_state == "Reading" else 1.0
                             min_d = self.min_delay * delay_multiplier; max_d = self.max_delay * delay_multiplier
                             try:
                                 if page_instance and not page_instance.is_closed():
                                      page_instance.wait_for_timeout(random.uniform(min_d, max_d) * 1000)
                             except PlaywrightError: break # Stop delay if page closes
                             except: pass # Ignore other errors during delay

                    # Break outer loops if needed
                    if bot_thread.should_stop: break
                    if not page_instance or page_instance.is_closed(): break

                    # If no action was performed in this cycle (all skipped/failed), add a small base delay
                    if not action_performed_this_cycle:
                         try:
                             if page_instance and not page_instance.is_closed():
                                 page_instance.wait_for_timeout(random.uniform(self.min_delay, self.max_delay) * 1000)
                         except PlaywrightError: break # Stop delay if page closes
                         except: pass

                # --- End of interaction loop ---
                if bot_thread.should_stop:
                    update_signal.emit(f"Browser {browser_id+1}: Stop requested during interaction for URL {i+1}.", browser_id)
                    # No break needed here, will exit URL loop naturally
                elif not page_instance or page_instance.is_closed():
                    update_signal.emit(f"Browser {browser_id+1}: Page was closed, ended interaction for URL {i+1}.", browser_id)
                    # No screenshot here, page is gone
                else:
                    update_signal.emit(f"Browser {browser_id+1}: Finished interaction for URL {i+1} ({page_interaction_count} cycles).", browser_id)
                    if random.random() < 0.15: # Reduced screenshot frequency
                        self.browser_manager.take_screenshot(f"browser_{browser_id+1}_end_url_{i}.png")

                # Explicitly close browser instance after EACH URL is processed
                if self.browser_manager:
                    self.browser_manager.close_browser()
                    page_instance = None # Clear reference
                    update_signal.emit(f"Browser {browser_id+1}: Browser instance closed after URL {i+1}.", browser_id)

                # Pause between URLs if not the last one
                if i < len(self.urls) - 1 and not bot_thread.should_stop:
                    inter_url_delay = random.uniform(2, 5)
                    update_signal.emit(f"Browser {browser_id+1}: Pausing {inter_url_delay:.1f}s before next URL...", browser_id)
                    try:
                         bot_thread.msleep(int(inter_url_delay * 1000))
                    except: pass # Ignore errors during sleep


            # --- End of URL loop ---
            if not bot_thread.should_stop:
                update_signal.emit(f"Browser {browser_id+1}: Finished all URLs.", browser_id)

        except Exception as e:
            # Catch any unexpected errors during the main run logic
            error_msg = f"Browser {browser_id+1}: Unhandled Exception in Bot Run"
            error_logger.exception(error_msg)
            update_signal.emit(f"{error_msg}: {e}", browser_id)
            bot_thread.error_signal.emit(f"Unhandled Exception: {e}", browser_id) # Emit error signal
            try: # Attempt screenshot if browser/page still exist
                if self.browser_manager and self.browser_manager.page and not self.browser_manager.page.is_closed():
                    self.browser_manager.take_screenshot(f"browser_{browser_id+1}_critical_run_error.png")
            except: pass
        finally:
            # Ensure final cleanup happens regardless of where errors occurred
            if self.browser_manager:
                self.browser_manager.close_browser()
            logging.info(f"Bot {browser_id+1} run method exiting.")


class ProxyValidator:
    """Validates proxy format and checks connectivity."""

    def validate_proxy_format(self, proxy: str) -> bool:
        # Allows user:pass@host:port, host:port, ip:port
        # Updated regex to better handle domains/IPs and optional user:pass
        return bool(re.match(r"^(?:([^:@/]+)(?::([^:@/]+))?@)?([\w.-]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5})$", proxy))

    def check_proxy(self, proxy_address: str, proxy_type: str, timeout_ms: int = 10000) -> bool:
        """Checks proxy connectivity using Playwright."""
        proxy_type = proxy_type.lower()
        if proxy_type not in ("http", "socks4", "socks5"): # Note: 'https' should already be mapped to 'http'
            logging.warning(f"Invalid proxy type '{proxy_type}' for check: {proxy_address}")
            return False
        if not self.validate_proxy_format(proxy_address):
            logging.warning(f"Invalid proxy format for check: '{proxy_address}'")
            return False

        # Use 'http' scheme for http proxies, 'socks5'/'socks4' for socks
        scheme = proxy_type if proxy_type.startswith("socks") else "http"
        full_proxy_url = f"{scheme}://{proxy_address}"
        # Target URL for IP check (consider alternatives like ipinfo.io/json)
        test_url = "https://api.ipify.org?format=json"
        # test_url = "https://ipinfo.io/json" # Alternative, gives more info

        logging.debug(f"Checking proxy: {full_proxy_url} -> {test_url} (Timeout: {timeout_ms}ms)")

        try:
            with sync_playwright() as p_check:
                browser = context = page = None
                try:
                    browser = p_check.chromium.launch(
                        proxy={"server": full_proxy_url},
                        headless=True,
                        timeout=timeout_ms + 5000, # Launch timeout slightly longer
                        args=["--no-sandbox"] # Ensure sandbox arg
                    )
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36", # Simple UA
                        viewport={"width": 100, "height": 100},
                        locale="en-US", timezone_id="UTC", # Minimal context
                        java_script_enabled=False, # No JS needed for simple IP check
                        accept_downloads=False,
                        bypass_csp=True # Bypass CSP for simple IP check page
                    )
                    # Set timeouts for the context and navigation
                    context.set_default_navigation_timeout(timeout_ms)
                    context.set_default_timeout(timeout_ms)

                    page = context.new_page()
                    response = page.goto(test_url, wait_until="domcontentloaded")

                    is_ok = False
                    if response and response.ok:
                        try:
                             # Get response body safely
                             body = response.body()
                             content = body.decode('utf-8', errors='ignore')
                             # Check if content looks like JSON and contains an IP
                             if content.strip().startswith('{') and '"ip":' in content.lower():
                                 try:
                                     ip_data = json.loads(content)
                                     returned_ip = ip_data.get('ip')
                                     if returned_ip:
                                         logging.info(f"Proxy check OK: {full_proxy_url} -> IP: {returned_ip}")
                                         is_ok = True
                                     else:
                                         logging.warning(f"Proxy check {full_proxy_url}: Status OK but JSON has no 'ip' key: {content[:100]}...")
                                         is_ok = False # Consider no IP key as failure
                                 except json.JSONDecodeError:
                                      logging.warning(f"Proxy check {full_proxy_url}: Status OK but failed to parse JSON response: {content[:100]}...")
                                      is_ok = False # Stricter: parse failure is invalid
                             else:
                                 logging.warning(f"Proxy check {full_proxy_url}: Status OK but response doesn't look like JSON IP: {content[:100]}...")
                                 is_ok = False # Response format incorrect
                        except (PlaywrightError, TimeoutError) as content_err:
                             logging.warning(f"Proxy check {full_proxy_url}: Status OK but failed get content: {content_err}")
                             # Treat as OK if status was good but content failed? Risky. Let's mark as False.
                             is_ok = False
                    else:
                        status_code = response.status if response else 'N/A'
                        logging.warning(f"Proxy check FAIL: {full_proxy_url} - Status: {status_code}")
                        is_ok = False
                    return is_ok

                except TimeoutError as e:
                    # Extract concise timeout reason if possible
                    reason = str(e).splitlines()[0] if str(e) else "Timeout"
                    logging.warning(f"Proxy check TIMEOUT: {full_proxy_url} - {reason}")
                    return False
                except PlaywrightError as e:
                    err_str = str(e)
                    if "ERR_PROXY_CONNECTION_FAILED" in err_str or \
                       "ERR_TUNNEL_CONNECTION_FAILED" in err_str or \
                       "socksConnectionFailed" in err_str:
                         logging.warning(f"Proxy check CONNECTION FAILED: {full_proxy_url}")
                    elif "executable doesn't exist" in err_str:
                         logging.error(f"Playwright browser executable missing for proxy check! Run 'playwright install'")
                    elif "Target page, context or browser has been closed" in err_str:
                         logging.warning(f"Proxy check interrupted by closure: {full_proxy_url}") # Less severe
                    else:
                         logging.warning(f"Proxy check PlaywrightError: {full_proxy_url} - {err_str.splitlines()[0]}")
                    return False
                except Exception as e:
                    logging.error(f"Unexpected proxy check error {full_proxy_url}: {e}", exc_info=False)
                    return False
                finally: # Ensure cleanup
                    try:
                        if page and not page.is_closed(): page.close()
                    except Exception as pe: error_logger.debug(f"Error closing page during proxy check cleanup: {pe}")
                    try:
                        if context: context.close()
                    except Exception as ce: error_logger.debug(f"Error closing context during proxy check cleanup: {ce}")
                    try:
                        if browser: browser.close()
                    except Exception as be: error_logger.debug(f"Error closing browser during proxy check cleanup: {be}")
        except Exception as outer_e:
             logging.error(f"Playwright initialization error during proxy check: {outer_e}"); return False


# --- Main Window ---

class MainWindow(QWidget):
    """Main application window."""

    # Define signal for cross-thread communication (e.g., for manual generation results)
    manual_gen_ua_result_signal = pyqtSignal(bool, str)
    manual_gen_fp_result_signal = pyqtSignal(bool, str)
    proxy_check_progress_signal = pyqtSignal(int, int) # checked_count, total_count
    proxy_check_finished_signal = pyqtSignal(dict) # results dictionary

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aladdin Traffic Bot v2.4") # Version bump
        try: self.setWindowIcon(QIcon("resources/chromium_blue.png"))
        except Exception as e: print(f"Could not load window icon: {e}")
        self.setGeometry(100, 100, 1250, 950)
        self.sys_settings = QSettings("AladdinBot", "TrafficBotAppV2_4") # Updated app name slightly for settings isolation
        self.license_manager = LicenseManager()
        # Initialize FM with potentially empty API key, will be updated before runs
        self.fingerprint_manager = FingerprintManager(user_agents, generated_user_agents, "")
        self.bot_threads: Dict[int, BotThread] = {}
        self.browser_logs: Dict[int, QTextEdit] = {}
        self.proxy_validator = ProxyValidator()
        self.proxies: List[Tuple[str, str, str]] = [] # (address, type, status: "unchecked"|"valid"|"invalid"|"checking")
        self.current_proxy_index = 0
        self.is_checking_proxies = False # Flag to prevent concurrent checks
        self.proxy_check_results_cache: Dict[int, Tuple[str, str, str]] = {} # Cache for check results
        self.total_runs_planned = 0
        self.runs_completed_count = 0
        self.run_failed_count = 0


        self.setup_ui()
        self.connect_signals() # Connect custom signals
        self.load_state() # Load persistent UI settings first
        self.load_proxies_from_file(show_message=False) # Load initial proxies without message box
        self.update_fingerprint_combo() # Populate fingerprint dropdown based on loaded profiles/options
        self.update_license_status_display() # Update license status based on loaded state

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.main_tab = QWidget(); self.setup_main_tab(); self.tabs.addTab(self.main_tab, "🚦 Main Control")
        self.settings_tab = QWidget(); self.setup_settings_tab(); self.tabs.addTab(self.settings_tab, "⚙️ Settings")
        self.license_tab = QWidget(); self.setup_license_tab(); self.tabs.addTab(self.license_tab, "🔑 License")
        main_layout.addWidget(self.tabs)

    def setup_main_tab(self):
        """Sets up the main control tab's UI."""
        layout = QVBoxLayout(self.main_tab)
        top_h_layout = QHBoxLayout()

        # URL Input
        url_group = QGroupBox("🎯 URLs"); url_layout = QVBoxLayout()
        self.url_text_edit = QTextEdit(); self.url_text_edit.setPlaceholderText("Enter target URLs, one per line (http:// or https://)")
        url_layout.addWidget(self.url_text_edit); url_group.setLayout(url_layout); top_h_layout.addWidget(url_group, stretch=2)

        # Proxy List
        proxy_group = QGroupBox("🔌 Proxies"); proxy_layout = QVBoxLayout()
        self.proxy_list_widget = QListWidget(); self.proxy_list_widget.setItemDelegate(ProxyDelegate()); self.proxy_list_widget.setAlternatingRowColors(True)
        proxy_layout.addWidget(self.proxy_list_widget)
        # Proxy Controls
        proxy_ctrl_layout = QHBoxLayout()
        self.load_proxies_button = QPushButton("📂 Load"); self.load_proxies_button.setToolTip(f"Load proxies from {settings.get('PROXY_FILE')}")
        proxy_ctrl_layout.addWidget(self.load_proxies_button)
        self.check_proxies_button = QPushButton("🧪 Check"); self.check_proxies_button.setToolTip("Check connectivity for all loaded proxies")
        proxy_ctrl_layout.addWidget(self.check_proxies_button)
        self.clear_proxies_button = QPushButton("❌ Clear"); self.clear_proxies_button.setToolTip("Clear the proxy list")
        proxy_ctrl_layout.addWidget(self.clear_proxies_button)
        proxy_layout.addLayout(proxy_ctrl_layout)
        proxy_group.setLayout(proxy_layout); top_h_layout.addWidget(proxy_group, stretch=1)

        layout.addLayout(top_h_layout)

        # Run Configuration
        run_config_group = QGroupBox("🚀 Run Configuration"); run_config_layout = QFormLayout()
        self.total_runs_spinbox = QSpinBox(); self.total_runs_spinbox.setRange(1, 10000); self.total_runs_spinbox.setValue(1)
        run_config_layout.addRow(BoldLabel("Total Bot Instances:"), self.total_runs_spinbox)
        self.run_group_size_spinbox = QSpinBox(); self.run_group_size_spinbox.setRange(1, 50); self.run_group_size_spinbox.setValue(1); self.run_group_size_spinbox.setToolTip("Number of bots to launch and run concurrently in each group.")
        run_config_layout.addRow(BoldLabel("Concurrent Instances per Group:"), self.run_group_size_spinbox)
        run_config_group.setLayout(run_config_layout); layout.addWidget(run_config_group)

        # Controls
        controls_layout = QHBoxLayout()
        self.start_button = QPushButton("▶️ Start Bot(s)"); self.start_button.setObjectName("start_button"); self.start_button.setStyleSheet("font-size: 14pt;") # Specific style via object name
        controls_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("⏹️ Stop All Bots"); self.stop_button.setObjectName("stop_button"); self.stop_button.setStyleSheet("font-size: 14pt;")
        self.stop_button.setEnabled(False); controls_layout.addWidget(self.stop_button)
        layout.addLayout(controls_layout)

        # Browser Logs
        log_group = QGroupBox("📜 Bot Logs"); log_layout = QVBoxLayout()
        self.browser_logs_tab_widget = QTabWidget(); self.browser_logs_tab_widget.setTabsClosable(False) # Keep tabs persistent
        log_layout.addWidget(self.browser_logs_tab_widget); log_group.setLayout(log_layout); layout.addWidget(log_group, stretch=1)

        # Connect main tab buttons here
        self.load_proxies_button.clicked.connect(lambda: self.load_proxies_from_file(show_message=True))
        self.check_proxies_button.clicked.connect(self.check_all_proxies)
        self.clear_proxies_button.clicked.connect(self.clear_proxy_list)
        self.start_button.clicked.connect(self.start_bots_grouped)
        self.stop_button.clicked.connect(self.stop_all_bots)

    def setup_settings_tab(self):
        """ Sets up the settings tab's UI using QScrollArea."""
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        settings_widget = QWidget(); layout = QVBoxLayout(settings_widget)

        # --- General ---
        general_group = QGroupBox("🌐 General"); general_layout = QFormLayout()
        self.headless_check = QCheckBox("Headless Mode"); self.headless_check.setToolTip("Run browsers without a visible window.")
        general_layout.addRow(self.headless_check)
        self.min_delay_spin = QDoubleSpinBox(); self.min_delay_spin.setRange(0.1, 60.0); self.min_delay_spin.setSingleStep(0.1); self.min_delay_spin.setDecimals(1)
        general_layout.addRow(BoldLabel("Min Action Delay (s):"), self.min_delay_spin)
        self.max_delay_spin = QDoubleSpinBox(); self.max_delay_spin.setRange(0.2, 120.0); self.max_delay_spin.setSingleStep(0.1); self.max_delay_spin.setDecimals(1)
        general_layout.addRow(BoldLabel("Max Action Delay (s):"), self.max_delay_spin)
        general_group.setLayout(general_layout); layout.addWidget(general_group)

        # --- Proxy ---
        proxy_group = QGroupBox("🔌 Proxy"); proxy_layout = QFormLayout()
        self.proxy_enabled_check = QCheckBox("Enable Proxy Usage")
        self.proxy_enabled_check.setToolTip("Use proxies from Main tab list (must be loaded).\nSOCKS5 recommended for privacy (hides public IP with WebRTC flag).\nHTTP/S proxies only hide local IP with WebRTC flag.")
        proxy_layout.addRow(self.proxy_enabled_check)
        proxy_group.setLayout(proxy_layout); layout.addWidget(proxy_group)

        # --- Fingerprinting & Evasion ---
        fp_group = QGroupBox("🛡️ Fingerprinting & Evasion"); fp_layout = QFormLayout()
        self.fingerprint_profile_combo = self.create_fingerprint_profile_combo() # Create instance
        fp_layout.addRow(BoldLabel("Fingerprint Mode:"), self.fingerprint_profile_combo)
        self.disable_automation_flags_check = QCheckBox("Hide Automation Flags"); self.disable_automation_flags_check.setToolTip("Injects JS to hide navigator.webdriver and other flags.")
        fp_layout.addRow(self.disable_automation_flags_check)
        self.prevent_webrtc_leak_check = QCheckBox("Attempt WebRTC IP Leak Prevention");
        self.prevent_webrtc_leak_check.setToolTip("Uses Chrome/Chromium launch flag (--force-webrtc-ip-handling-policy=default_public_interface_only) AND JS overrides to prevent local IP leaks via WebRTC. SOCKS5 proxy recommended to also hide public IP.")
        fp_layout.addRow(self.prevent_webrtc_leak_check)

        self.manual_fp_gen_button = QPushButton("✨ Generate New Profile (via Gemini)")
        self.manual_fp_gen_button.setObjectName("generate_fp_button") # For potential specific styling
        self.manual_fp_gen_button.setToolTip(f"Generates a new full fingerprint profile using the Gemini API key below and saves it to {settings.get('FINGERPRINT_FILE')}. Requires a valid API Key.")
        fp_layout.addRow(self.manual_fp_gen_button)

        fp_group.setLayout(fp_layout); layout.addWidget(fp_group)


        # --- User Agent Generation ---
        ua_gen_group = QGroupBox("🤖 User Agent Generation (Gemini)"); ua_gen_layout = QFormLayout()
        self.gemini_api_key_input = QLineEdit(); self.gemini_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_api_key_input.setToolTip("Required for ALL generation features (User Agents and Fingerprint Profiles).")
        ua_gen_layout.addRow(BoldLabel("Google Gemini API Key:"), self.gemini_api_key_input)
        self.user_agent_generation_check = QCheckBox("Enable Auto-Generation (when UAs exhausted)")
        ua_gen_layout.addRow(self.user_agent_generation_check)
        self.user_agent_generation_count_spin = QSpinBox(); self.user_agent_generation_count_spin.setRange(5, 50); self.user_agent_generation_count_spin.setSingleStep(5); self.user_agent_generation_count_spin.setValue(10)
        ua_gen_layout.addRow(BoldLabel("    Auto-Gen Count:"), self.user_agent_generation_count_spin) # Indented

        # Manual UA Generation Section
        manual_gen_layout = QHBoxLayout()
        manual_gen_layout.addWidget(QLabel("Generate Manually:"))
        self.manual_ua_gen_count_spin = QSpinBox(); self.manual_ua_gen_count_spin.setRange(1, 100); self.manual_ua_gen_count_spin.setValue(10)
        manual_gen_layout.addWidget(self.manual_ua_gen_count_spin)
        manual_gen_layout.addWidget(QLabel("UAs"))
        self.manual_ua_gen_button = QPushButton("✨ Generate Now")
        self.manual_ua_gen_button.setObjectName("generate_ua_button")
        self.manual_ua_gen_button.setToolTip("Generate new User Agents using Gemini and save to generated_user_agents.json. Requires a valid API Key.")
        manual_gen_layout.addWidget(self.manual_ua_gen_button)
        manual_gen_layout.addStretch()
        ua_gen_layout.addRow(manual_gen_layout)

        ua_gen_group.setLayout(ua_gen_layout); layout.addWidget(ua_gen_group)

        # --- Behavior ---
        behavior_group = QGroupBox("🚶 Behavior Simulation"); behavior_layout = QFormLayout()
        self.mouse_movement_check = QCheckBox("Enable Mouse Movement"); behavior_layout.addRow(self.mouse_movement_check)
        self.scroll_duration_min_spin = QSpinBox(); self.scroll_duration_min_spin.setRange(200, 5000); self.scroll_duration_min_spin.setSingleStep(50); self.scroll_duration_min_spin.setSuffix(" ms")
        behavior_layout.addRow(BoldLabel("Min Scroll Anim Duration:"), self.scroll_duration_min_spin)
        self.scroll_duration_max_spin = QSpinBox(); self.scroll_duration_max_spin.setRange(300, 10000); self.scroll_duration_max_spin.setSingleStep(50); self.scroll_duration_max_spin.setSuffix(" ms")
        behavior_layout.addRow(BoldLabel("Max Scroll Anim Duration:"), self.scroll_duration_max_spin)
        self.behavioral_states_check = QCheckBox("Enable Behavioral States"); behavior_layout.addRow(self.behavioral_states_check)
        self.skip_action_prob_spin = QDoubleSpinBox(); self.skip_action_prob_spin.setRange(0.0, 0.5); self.skip_action_prob_spin.setSingleStep(0.01); self.skip_action_prob_spin.setDecimals(2)
        behavior_layout.addRow(BoldLabel("Skip Action Probability:"), self.skip_action_prob_spin)
        behavior_group.setLayout(behavior_layout); layout.addWidget(behavior_group)

        # --- Interaction Features ---
        interact_group = QGroupBox("🖱️ Interaction Features"); interact_layout = QFormLayout()
        self.form_fill_check = QCheckBox("Enable Form Filling"); interact_layout.addRow(self.form_fill_check)
        self.impression_enabled_check = QCheckBox("Enable Impression/Pagination"); interact_layout.addRow(self.impression_enabled_check)
        interact_layout.addRow(BoldLabel("Next Page Selectors (CSS, one per line):"))
        self.next_page_selectors_edit = QTextEdit(); self.next_page_selectors_edit.setAcceptRichText(False); self.next_page_selectors_edit.setFixedHeight(60)
        interact_layout.addRow(self.next_page_selectors_edit)
        self.next_page_text_fallback_edit = QLineEdit(); self.next_page_text_fallback_edit.setPlaceholderText("Comma-separated: Next, >")
        interact_layout.addRow(BoldLabel("Next Page Text Fallback (Link/Button Text):"), self.next_page_text_fallback_edit)
        interact_layout.addWidget(QLabel("---")); # Separator
        self.ad_click_enabled_check = QCheckBox("Enable Ad Clicking"); interact_layout.addRow(self.ad_click_enabled_check)
        self.ad_click_probability_spin = QDoubleSpinBox(); self.ad_click_probability_spin.setRange(0.0, 1.0); self.ad_click_probability_spin.setSingleStep(0.01); self.ad_click_probability_spin.setDecimals(2)
        interact_layout.addRow(BoldLabel("Ad Click Probability:"), self.ad_click_probability_spin)
        interact_layout.addRow(BoldLabel("Ad Selectors (CSS, one per line):"))
        self.ad_selectors_edit = QTextEdit(); self.ad_selectors_edit.setAcceptRichText(False); self.ad_selectors_edit.setFixedHeight(60)
        interact_layout.addRow(self.ad_selectors_edit)
        interact_group.setLayout(interact_layout); layout.addWidget(interact_group)

        # --- Advanced/Chromium Blue ---
        advanced_group = QGroupBox("🚀 Advanced / Custom Browser"); advanced_layout = QFormLayout()
        self.chromium_blue_check = QCheckBox("Use Custom Chromium Build");
        advanced_layout.addRow(self.chromium_blue_check)
        cb_path_layout = QHBoxLayout()
        self.chromium_blue_path_input = QLineEdit(); self.chromium_blue_path_input.setPlaceholderText("/path/to/chromium or C:\\path\\to\\chrome.exe"); self.chromium_blue_path_input.setEnabled(False)
        cb_path_layout.addWidget(self.chromium_blue_path_input)
        self.cb_browse_button = QPushButton("Browse..."); self.cb_browse_button.setEnabled(False)
        cb_path_layout.addWidget(self.cb_browse_button)
        advanced_layout.addRow(BoldLabel("    Executable Path:"), cb_path_layout)
        self.chromium_blue_args_input = QLineEdit(); self.chromium_blue_args_input.setPlaceholderText("--disable-features=MyFeature --ignore-gpu-blocklist"); self.chromium_blue_args_input.setEnabled(False)
        advanced_layout.addRow(BoldLabel("    Extra Arguments:"), self.chromium_blue_args_input)
        advanced_group.setLayout(advanced_layout); layout.addWidget(advanced_group)

        # --- Save/Load Buttons ---
        settings_buttons_layout = QHBoxLayout()
        self.load_settings_button = QPushButton("🔄 Load Settings"); self.load_settings_button.setToolTip("Load from config/settings.py")
        settings_buttons_layout.addWidget(self.load_settings_button)
        self.save_settings_button = QPushButton("💾 Save Settings"); self.save_settings_button.setToolTip("Save to config/settings.py")
        settings_buttons_layout.addWidget(self.save_settings_button)
        layout.addLayout(settings_buttons_layout)

        layout.addStretch(); scroll_area.setWidget(settings_widget)
        settings_tab_layout = QVBoxLayout(self.settings_tab); settings_tab_layout.addWidget(scroll_area)

        # Connect settings tab buttons/widgets here
        self.chromium_blue_check.stateChanged.connect(self.toggle_chromium_blue_fields)
        self.cb_browse_button.clicked.connect(self.browse_chromium_path)
        self.manual_fp_gen_button.clicked.connect(self.trigger_manual_fp_generation)
        self.manual_ua_gen_button.clicked.connect(self.trigger_manual_ua_generation)
        self.load_settings_button.clicked.connect(self.load_settings_from_file_ui)
        self.save_settings_button.clicked.connect(self.save_settings_to_file_ui)


    def create_fingerprint_profile_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setToolTip("Select Fingerprint: Specific profile, Random, or Generate new UA per bot")
        # Populate later in update_fingerprint_combo
        return combo

    def update_fingerprint_combo(self):
        """Loads profile names and updates the combo box."""
        self.fingerprint_manager.fingerprint_profiles = self.fingerprint_manager._load_fingerprint_profiles() # Reload profiles
        current_selection = self.fingerprint_profile_combo.currentText()
        self.fingerprint_profile_combo.clear()
        profile_names = self.fingerprint_manager.get_profile_names() # Includes "Random", "Generate & Use..."
        self.fingerprint_profile_combo.addItems(profile_names)

        # Try to restore previous selection or default from loaded settings file
        target_profile = current_selection or settings.get("FINGERPRINT_PROFILE_NAME", "Random")
        index = self.fingerprint_profile_combo.findText(target_profile)
        if index != -1: self.fingerprint_profile_combo.setCurrentIndex(index)
        elif profile_names: self.fingerprint_profile_combo.setCurrentIndex(0) # Default to first item ("Random") if target not found

    def setup_license_tab(self):
        layout = QVBoxLayout(self.license_tab); layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        license_group = QGroupBox("🔑 License Activation"); form_layout = QFormLayout()
        self.license_key_input = QLineEdit(); self.license_key_input.setPlaceholderText("Enter License Key")
        form_layout.addRow(BoldLabel("License Key:"), self.license_key_input)
        self.activate_button = QPushButton("🚀 Activate License")
        form_layout.addRow(self.activate_button)
        self.license_status_label = CenteredLabel("Status: Unknown"); font = self.license_status_label.font(); font.setPointSize(12); font.setBold(True); self.license_status_label.setFont(font)
        form_layout.addRow(self.license_status_label); license_group.setLayout(form_layout)
        layout.addWidget(license_group); layout.addStretch()

        # Connect license tab button
        self.activate_button.clicked.connect(self.activate_license)

    def connect_signals(self):
        """Connect custom signals to their handler slots."""
        self.manual_gen_ua_result_signal.connect(self._show_manual_gen_result)
        self.manual_gen_fp_result_signal.connect(self._show_manual_fp_gen_result)
        self.proxy_check_progress_signal.connect(self.update_proxy_check_progress)
        self.proxy_check_finished_signal.connect(self.finalize_proxy_check)

    # --- Toggle / Browse / Manual Gen Functions ---

    def toggle_chromium_blue_fields(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.chromium_blue_path_input.setEnabled(enabled)
        self.chromium_blue_args_input.setEnabled(enabled)
        self.cb_browse_button.setEnabled(enabled)

    def browse_chromium_path(self):
        # Use native path separators
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Chromium Executable", "", "Applications (*.exe);;All Files (*)")
        if filepath: self.chromium_blue_path_input.setText(os.path.normpath(filepath))

    # --- Manual UA Generation Handlers ---
    def trigger_manual_ua_generation(self):
        """Handles the 'Generate Now' UA button click."""
        api_key = self.gemini_api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API Key Missing", "Please enter your Google Gemini API Key in the Settings tab to generate user agents.")
            return

        count = self.manual_ua_gen_count_spin.value()
        # Disable button and show progress
        self.manual_ua_gen_button.setEnabled(False)
        self.manual_ua_gen_button.setText("Generating...")
        QApplication.processEvents() # Update UI immediately

        # Run generation in a separate thread
        threading.Thread(target=self._run_manual_ua_gen, args=(count, api_key), daemon=True).start()

    def _run_manual_ua_gen(self, count, api_key):
        """Worker function for manual UA generation."""
        self.fingerprint_manager.gemini_api_key = api_key # Ensure FM has the key
        success, message = self.fingerprint_manager.generate_user_agents_manual(count, api_key)
        # Emit signal to update UI from main thread
        self.manual_gen_ua_result_signal.emit(success, message)

    # SLOT for manual_gen_ua_result_signal
    def _show_manual_gen_result(self, success, message):
        """Shows the result of manual UA generation in the main thread."""
        if success:
            QMessageBox.information(self, "Manual UA Generation Complete", message)
        else:
            QMessageBox.warning(self, "Manual UA Generation Failed", message)
        # Re-enable the button
        self.manual_ua_gen_button.setEnabled(True)
        self.manual_ua_gen_button.setText("✨ Generate Now")

    # --- Manual FP Profile Generation Handlers ---
    def trigger_manual_fp_generation(self):
        """Handles the 'Generate New Profile' button click."""
        api_key = self.gemini_api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API Key Missing", "Please enter your Google Gemini API Key in the Settings tab to generate profiles.")
            return

        self.manual_fp_gen_button.setEnabled(False)
        self.manual_fp_gen_button.setText("Generating...")
        QApplication.processEvents()

        # Run generation in a separate thread
        threading.Thread(target=self._run_manual_fp_gen, args=(api_key,), daemon=True).start()

    def _run_manual_fp_gen(self, api_key):
        """Worker function for manual FP profile generation."""
        self.fingerprint_manager.gemini_api_key = api_key # Ensure FM has the key
        success, message = self.fingerprint_manager.generate_fingerprint_profile_manual(api_key)
        # Emit signal to update UI from main thread
        self.manual_gen_fp_result_signal.emit(success, message)

    # SLOT for manual_gen_fp_result_signal
    def _show_manual_fp_gen_result(self, success, message):
        """Shows the result of manual FP profile generation in the main thread."""
        if success:
            QMessageBox.information(self, "Profile Generation Complete", message)
            self.update_fingerprint_combo() # Refresh dropdown list
        else:
            QMessageBox.warning(self, "Profile Generation Failed", message)
        # Re-enable the button
        self.manual_fp_gen_button.setEnabled(True)
        self.manual_fp_gen_button.setText("✨ Generate New Profile (via Gemini)")

    # --- Core Actions ---
    def activate_license(self):
        license_key = self.license_key_input.text().strip()
        if not license_key: QMessageBox.warning(self, "License", "Please enter a license key."); return
        is_valid, message = self.license_manager.verify_license(license_key)
        self.update_license_status_display()
        if is_valid:
            QMessageBox.information(self, "License Activated", f"License activated successfully!\n{message}")
            self.save_state() # Save state including the activated key
        else:
            QMessageBox.warning(self, "License Error", message)

    def update_license_status_display(self):
         is_valid_now, message_now = self.license_manager.is_license_valid()
         self.license_status_label.setText(f"Status: {message_now}")
         color = "green" if "Active" in message_now else ("red" if "Expired" in message_now else "darkorange")
         self.license_status_label.setStyleSheet(f"color: {color}; font-size: 12pt; font-weight: bold;") # Ensure style applied

    def load_proxies_from_file(self, show_message: bool = True):
        """Loads proxies from file and updates the list widget."""
        if self.is_checking_proxies:
             if show_message: QMessageBox.warning(self, "Busy", "Cannot load proxies while check is in progress.")
             return
        proxy_file = settings.get('PROXY_FILE', 'config/proxies.txt')
        loaded_proxies_tuples = load_proxies()
        # Reset status to "unchecked" on load
        self.proxies = [(addr, ptype, "unchecked") for addr, ptype in loaded_proxies_tuples]
        self.proxy_list_widget.clear()

        if not self.proxies:
            if show_message and os.path.exists(proxy_file):
                 QMessageBox.information(self, "Load Proxies", f"No valid proxies found or file is empty:\n{proxy_file}")
            elif not os.path.exists(proxy_file):
                 logging.warning(f"Proxy file not found: {proxy_file}")
            return

        for proxy_address, proxy_type, status in self.proxies:
            item = QListWidgetItem(f"{proxy_address} ({proxy_type})")
            item.setData(Qt.ItemDataRole.UserRole, (proxy_address, proxy_type, status))
            self.proxy_list_widget.addItem(item)

        if self.proxies and show_message:
            QMessageBox.information(self, "Proxies Loaded", f"Loaded {len(self.proxies)} proxies from {proxy_file}.\nStatus reset to 'unchecked'.")
        self.proxy_list_widget.viewport().update()


    def clear_proxy_list(self):
        if self.is_checking_proxies:
             QMessageBox.warning(self, "Busy", "Cannot clear proxies while check is in progress.")
             return
        # Confirm before clearing
        reply = QMessageBox.question(self, "Clear Proxies", "Are you sure you want to clear the proxy list?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.proxies.clear(); self.proxy_list_widget.clear()
            QMessageBox.information(self, "Proxies Cleared", "Proxy list cleared.")


    def check_all_proxies(self):
        """Checks validity of all loaded proxies using a background thread."""
        if not self.proxies:
            QMessageBox.warning(self, "Check Proxies", "No proxies loaded to check.")
            return
        if self.is_checking_proxies:
            QMessageBox.warning(self, "Check Proxies", "Proxy check already in progress.")
            return

        self.is_checking_proxies = True
        num_proxies = len(self.proxies)
        self.proxy_check_results_cache.clear() # Clear previous results

        # Update UI to show checking state
        self.check_proxies_button.setEnabled(False)
        self.check_proxies_button.setText(f"Checking 0/{num_proxies}...")
        self.load_proxies_button.setEnabled(False)
        self.clear_proxies_button.setEnabled(False)
        self.proxy_list_widget.setEnabled(False)

        # Set initial status to "checking" visually
        for i in range(num_proxies):
             item = self.proxy_list_widget.item(i)
             if item:
                 addr, ptype, _ = self.proxies[i]
                 item.setData(Qt.ItemDataRole.UserRole, (addr, ptype, "checking"))
        self.proxy_list_widget.viewport().update()
        QApplication.processEvents()

        # Start the background checking thread
        threading.Thread(target=self._run_proxy_check_thread, daemon=True).start()

    def _run_proxy_check_thread(self):
        """Worker thread for checking proxies concurrently."""
        num_proxies = len(self.proxies)
        max_threads = 15 # Concurrent checks limit
        threads = []
        results_lock = threading.Lock()
        checked_count = [0] # List to allow modification in nested func

        def check_proxy_worker(index, address, proxy_type):
            is_valid = self.proxy_validator.check_proxy(address, proxy_type)
            status = "valid" if is_valid else "invalid"
            with results_lock:
                self.proxy_check_results_cache[index] = (address, proxy_type, status)
                checked_count[0] += 1
                # Emit progress signal periodically
                if checked_count[0] % 5 == 0 or checked_count[0] == num_proxies:
                    self.proxy_check_progress_signal.emit(checked_count[0], num_proxies)

        # Launch worker threads in batches
        for i, (address, original_type, _) in enumerate(self.proxies):
             thread = threading.Thread(target=check_proxy_worker, args=(i, address, original_type), daemon=True)
             threads.append(thread); thread.start()
             if len(threads) >= max_threads: # Wait for batch
                  for t in threads: t.join()
                  threads = []
                  # Allow GUI to update between batches slightly
                  time.sleep(0.05)

        # Wait for any remaining threads
        for t in threads: t.join()

        # Signal completion with results
        self.proxy_check_finished_signal.emit(self.proxy_check_results_cache.copy())


    # SLOT for proxy_check_progress_signal
    def update_proxy_check_progress(self, checked_count, total_count):
        """Updates the check button text and list visuals during proxy check."""
        self.check_proxies_button.setText(f"Checking {checked_count}/{total_count}...")
        # Update visuals based on partial results in cache
        for i, data in self.proxy_check_results_cache.items():
             item = self.proxy_list_widget.item(i)
             if item: item.setData(Qt.ItemDataRole.UserRole, data)
        self.proxy_list_widget.viewport().update()
        QApplication.processEvents()


    # SLOT for proxy_check_finished_signal
    def finalize_proxy_check(self, results: Dict):
        """Updates UI and state after all proxy checks are done."""
        valid_count = 0
        invalid_count = 0
        # Update the main self.proxies list and the QListWidget items
        for i in range(len(self.proxies)):
            if i in results:
                 address, checked_type, status = results[i]
                 self.proxies[i] = (address, checked_type, status) # Update internal list
                 item = self.proxy_list_widget.item(i)
                 if item: item.setData(Qt.ItemDataRole.UserRole, self.proxies[i]) # Update item data
                 if status == "valid": valid_count += 1
                 elif status == "invalid": invalid_count += 1
            else: # Should not happen if logic is correct
                 logging.warning(f"Proxy check result missing for index {i}")

        # Reset UI state
        self.proxy_list_widget.viewport().update() # Final visual refresh
        self.check_proxies_button.setEnabled(True); self.check_proxies_button.setText("🧪 Check")
        self.load_proxies_button.setEnabled(True); self.clear_proxies_button.setEnabled(True)
        self.proxy_list_widget.setEnabled(True)
        self.is_checking_proxies = False # Reset flag

        QMessageBox.information(self, "Proxy Check Complete", f"Checked {len(self.proxies)} proxies.\nValid: {valid_count}\nInvalid: {invalid_count}")


    def get_next_valid_proxy(self) -> Optional[Tuple[str, str]]:
        """Gets the next available *valid* proxy, cycling through them."""
        valid_proxies = [(addr, ptype) for addr, ptype, status in self.proxies if status == "valid"]
        if not valid_proxies: logging.warning("get_next_valid_proxy: No valid proxies available."); return None
        num_valid = len(valid_proxies)
        proxy_to_use = valid_proxies[self.current_proxy_index % num_valid]
        self.current_proxy_index += 1
        return proxy_to_use

    def get_next_any_proxy(self) -> Optional[Tuple[str, str]]:
        """Gets the next proxy from the list (any status except maybe 'invalid'?), cycling."""
        # Option 1: Use only 'valid' or 'unchecked'
        # usable_proxies = [(addr, ptype) for addr, ptype, status in self.proxies if status != "invalid"]
        # Option 2: Use strictly ANY proxy
        usable_proxies = [(addr, ptype) for addr, ptype, status in self.proxies]

        if not usable_proxies: logging.warning("get_next_any_proxy: No usable proxies loaded."); return None
        num_usable = len(usable_proxies)
        proxy_data = usable_proxies[self.current_proxy_index % num_usable]
        self.current_proxy_index += 1
        return (proxy_data[0], proxy_data[1]) # (address, type)


    def start_bots_grouped(self):
        logging.info("Start button clicked.")
        if self.is_checking_proxies:
            QMessageBox.warning(self, "Busy", "Cannot start bots while proxy check is in progress.")
            return
        # 1. License Check
        is_valid, license_msg = self.license_manager.is_license_valid()
        if not is_valid:
            QMessageBox.critical(self, "License Error", f"Cannot start: {license_msg}"); self.tabs.setCurrentWidget(self.license_tab); return

        # 2. Get URLs
        urls = [url.strip() for url in self.url_text_edit.toPlainText().splitlines() if url.strip().startswith(('http://', 'https://'))]
        if not urls: QMessageBox.warning(self, "Input Error","Enter valid URLs (http:// or https://)."); return
        logging.info(f"Target URLs ({len(urls)}): {urls[:3]}...")

        # 3. Get Run Config
        total_runs = self.total_runs_spinbox.value(); run_group_size = self.run_group_size_spinbox.value()
        logging.info(f"Run config: Total Instances={total_runs}, Group Size={run_group_size}")

        # 4. Update Internal Settings from GUI (Snapshot for this run)
        current_run_config = self.get_current_settings_from_ui()
        global settings; settings = current_run_config.copy() # Update global settings context for this run

        # 5. Proxy Prep
        use_proxy = current_run_config.get("PROXY_ENABLED", False)
        get_proxy_func = None
        if use_proxy:
            logging.info("Proxy usage enabled.")
            if not self.proxies: QMessageBox.warning(self, "Proxy Error", "Proxy enabled, but no proxies loaded."); return
            valid_proxies_count = sum(1 for _, _, status in self.proxies if status == "valid")
            if valid_proxies_count > 0:
                 logging.info(f"{valid_proxies_count} valid proxies available. Will cycle through valid ones.")
                 get_proxy_func = self.get_next_valid_proxy
            else:
                 reply = QMessageBox.question(self, "Proxy Warning",
                                              "Proxies enabled, but none are marked 'valid'.\n\nDo you want to proceed using ANY loaded proxy (including unchecked or invalid)?",
                                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.Yes:
                     logging.warning("Proceeding with unchecked/invalid proxies. Will cycle through all.")
                     get_proxy_func = self.get_next_any_proxy
                 else:
                     QMessageBox.information(self, "Proxy Info", "Operation cancelled. Please load/check proxies or disable proxy usage.")
                     return
        else: logging.info("Proxy usage disabled.")

        # 6. Update UI State & Initialize Run Tracking
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.browser_logs_tab_widget.clear()
        self.browser_logs.clear()
        if self.bot_threads: self.bot_threads.clear() # Clear old refs
        self.current_proxy_index = 0
        self.total_runs_planned = total_runs
        self.runs_completed_count = 0
        self.run_failed_count = 0

        # 7. Initialize/Update Fingerprint Manager for this run group
        self.fingerprint_manager.gemini_api_key = current_run_config.get("GEMINI_API_KEY", "")
        self.fingerprint_manager.user_agents = load_user_agents()
        self.fingerprint_manager.generated_user_agents = load_generated_user_agents()
        self.fingerprint_manager.used_user_agents_in_run = set()

        # 8. Grouped Execution Loop
        runs_launched_in_total = 0; group_num = 0
        try:
            while runs_launched_in_total < self.total_runs_planned:
                 group_num += 1
                 QApplication.processEvents() # Process stop signal between groups
                 if not self.stop_button.isEnabled():
                     logging.warning("Stop requested between groups. Halting launch.")
                     break

                 current_group_target_size = min(run_group_size, self.total_runs_planned - runs_launched_in_total)
                 if current_group_target_size <= 0: break

                 logging.info(f"--- Starting Group {group_num} ({current_group_target_size} bots) ---")
                 self.update_log(f"--- Starting Group {group_num}/{-( -self.total_runs_planned // run_group_size)} ({current_group_target_size} bots) ---", -1)

                 active_group_threads = {}
                 for i in range(current_group_target_size):
                     browser_id = runs_launched_in_total + i

                     # Create Log Tab
                     log_text_edit = QTextEdit(); log_text_edit.setReadOnly(True); log_text_edit.setFontFamily("Consolas, Courier New, monospace")
                     self.browser_logs[browser_id] = log_text_edit
                     scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); scroll_area.setWidget(log_text_edit)
                     tab_title = f"Bot {browser_id + 1}"
                     tab_index = self.browser_logs_tab_widget.addTab(scroll_area, tab_title)
                     self.browser_logs_tab_widget.setTabText(tab_index, tab_title)
                     self.browser_logs_tab_widget.setTabToolTip(tab_index, f"Log for Bot Instance {browser_id + 1}")
                     # Reset tab color (in case it was red from previous run)
                     self.browser_logs_tab_widget.tabBar().setTabTextColor(tab_index, QColor("black"))
                     if i == 0 and group_num == 1: self.browser_logs_tab_widget.setCurrentIndex(tab_index)

                     # Assign Proxy
                     proxy_tuple = None
                     if use_proxy and get_proxy_func:
                         proxy_tuple = get_proxy_func()
                         if proxy_tuple: self.update_log(f"Assigning Proxy: {proxy_tuple[1]}://{proxy_tuple[0]}", browser_id)
                         else: self.update_log("WARNING: No more usable proxies available to assign.", browser_id)

                     # Create and Start Thread
                     bot_thread = BotThread(browser_id, list(urls), current_run_config.copy(), proxy_tuple, self.fingerprint_manager)
                     bot_thread.update_signal.connect(self.update_log)
                     bot_thread.finished_signal.connect(self.on_bot_finished)
                     bot_thread.error_signal.connect(self.on_bot_error)
                     self.bot_threads[browser_id] = bot_thread
                     active_group_threads[browser_id] = bot_thread
                     bot_thread.start(); self.update_log(f"Bot {browser_id + 1} thread started.", browser_id)
                     QApplication.processEvents() # Keep UI responsive

                 # Wait for Current Group to finish
                 logging.info(f"Waiting for {len(active_group_threads)} bots in group {group_num}..."); self.update_log(f"--- Waiting for Group {group_num} to complete... ---", -1)
                 active_ids = list(active_group_threads.keys())
                 while any(self.bot_threads.get(bid) and self.bot_threads[bid].isRunning() for bid in active_ids):
                     QApplication.processEvents()
                     if not self.stop_button.isEnabled(): break
                     time.sleep(0.1) # Brief sleep

                 if not self.stop_button.isEnabled():
                     logging.warning("Stop requested while waiting for group completion.")
                     break

                 runs_launched_in_total += current_group_target_size
                 logging.info(f"Group {group_num} finished. Total launched so far: {runs_launched_in_total}/{self.total_runs_planned}"); self.update_log(f"--- Group {group_num} finished. ({runs_launched_in_total}/{self.total_runs_planned} launched) ---", -1)
                 if runs_launched_in_total < self.total_runs_planned:
                      self.fingerprint_manager.used_user_agents_in_run = set()
                      logging.debug("Reset used UA pool for the next group.")

        except Exception as launch_err:
            error_logger.exception("Error during bot launch loop:"); QMessageBox.critical(self, "Launch Error", f"Error during launch:\n{launch_err}")
            self.check_if_all_finished() # Attempt to reset UI state

        # 9. Final Check (after loop finishes or breaks)
        self.check_if_all_finished() # Check if UI needs reset

    def check_if_all_finished(self):
        """Checks if all threads are done and resets UI if necessary."""
        # This might be called multiple times, ensure it only acts once
        total_accounted = self.runs_completed_count + self.run_failed_count
        no_active_threads = not any(t.isRunning() for t in self.bot_threads.values())

        # Check if the run is complete OR if stop was pressed (stop_button is disabled)
        run_is_over = (total_accounted >= self.total_runs_planned and no_active_threads) or not self.stop_button.isEnabled()

        if run_is_over and self.start_button.isEnabled() is False: # Only act if start button is currently disabled
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            # Determine final message based on whether stop was pressed or run completed naturally
            if not self.stop_button.isEnabled(): # Stop was pressed
                logging.info("Bot execution halted by user request.")
                QMessageBox.information(self, "Bots Stopped", f"Bot execution stopped by user.") # Simple message
            elif total_accounted >= self.total_runs_planned:
                logging.info("All planned bot runs completed or failed.")
                success_msg = f"All {self.total_runs_planned} bot instances completed processing."
                if self.run_failed_count > 0:
                    success_msg += f"\n({self.runs_completed_count} succeeded, {self.run_failed_count} failed - check red tabs/logs)."
                QMessageBox.information(self, "Bots Finished", success_msg)
            else: # Loop finished early without stop (e.g., launch error)
                 logging.warning(f"Bot run finished unexpectedly after {total_accounted}/{self.total_runs_planned} were accounted for.")
                 if total_accounted > 0 : QMessageBox.warning(self, "Bots Finished Early", f"Process finished early after {total_accounted}/{self.total_runs_planned} runs were processed (check logs for errors).")

            # Final cleanup of thread references
            self.bot_threads.clear()


    def stop_all_bots(self):
        active_threads = [t for t in self.bot_threads.values() if t and t.isRunning()]
        if not active_threads:
            logging.info("Stop: No active bot threads found.")
            self.check_if_all_finished() # Ensure UI is reset
            return

        logging.info(f"Stopping all {len(active_threads)} active bot threads...")
        # Update UI immediately to disable start and reflect stop action
        self.start_button.setEnabled(True) # Allow restart later
        self.stop_button.setEnabled(False) # Disable stop button itself

        for bot_thread in active_threads:
            logging.debug(f"Sending stop signal to Bot {bot_thread.browser_id + 1}")
            self.update_log(f"Stop signal sent.", bot_thread.browser_id)
            bot_thread.stop() # Set the internal flag and attempt browser close

        # Give threads a brief moment to acknowledge the stop signal
        wait_start_time = time.time(); max_wait_sec = 5 # Shorter wait
        logging.info(f"Waiting up to {max_wait_sec}s for threads to terminate gracefully...")

        while any(t.isRunning() for t in active_threads) and (time.time() - wait_start_time < max_wait_sec):
             QApplication.processEvents(); time.sleep(0.1)
             active_threads = [t for t in self.bot_threads.values() if t and t.isRunning()] # Re-check

        still_running = [t.browser_id + 1 for t in active_threads]
        if still_running:
             logging.warning(f"Stop timeout ({max_wait_sec}s). Threads still potentially running: {still_running}. They should terminate eventually.")
             QMessageBox.warning(self, "Stop Timeout", f"Threads ({len(still_running)}) did not stop within {max_wait_sec}s. They may take longer to close.")
        else:
             logging.info("All active bot threads stopped successfully.");
             QMessageBox.information(self, "Bots Stopped", "All running instances stopped.")

        self.check_if_all_finished() # Final UI state check/reset


    # --- Signal Handlers ---
    def update_log(self, message: str, browser_id: int):
        """Appends a message to the correct log widget or general log."""
        log_widget = self.browser_logs.get(browser_id)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {message}"

        if log_widget:
            try:
                 log_widget.append(log_line)
                 scrollbar = log_widget.verticalScrollBar()
                 if scrollbar.value() >= scrollbar.maximum() - 30:
                     scrollbar.setValue(scrollbar.maximum())
            except RuntimeError: # Handle cases where widget might be deleted
                 logging.warning(f"Log widget for Bot {browser_id+1} seems to be deleted. Msg: {message}")
        elif browser_id == -1: # General message
             logging.info(f"[--- General Log ---] {message}")
             # Append to the currently selected tab's log
             current_index = self.browser_logs_tab_widget.currentIndex()
             if current_index >= 0:
                 try:
                     scroll_area = self.browser_logs_tab_widget.widget(current_index)
                     if isinstance(scroll_area, QScrollArea):
                         log_widget_current = scroll_area.widget()
                         if isinstance(log_widget_current, QTextEdit):
                             log_widget_current.append(f"--- {message} ---")
                             scrollbar = log_widget_current.verticalScrollBar()
                             if scrollbar.value() >= scrollbar.maximum() - 30: scrollbar.setValue(scrollbar.maximum())
                 except Exception: pass # Ignore errors updating current tab if it's gone
        else: # Log widget not found for specific browser ID
            logging.warning(f"Log widget for Bot {browser_id+1} not found. Msg: {message}")

    def on_bot_finished(self, browser_id: int):
        self.update_log(f"Bot {browser_id + 1}: Task Finished.", browser_id)
        self.runs_completed_count += 1
        if browser_id in self.bot_threads:
             del self.bot_threads[browser_id] # Remove thread ref
        else:
             logging.warning(f"Finished signal received for unknown/already removed browser_id: {browser_id}")
        self.check_if_all_finished() # Check if all runs are done

    def on_bot_error(self, error_message: str, browser_id: int):
       self.update_log(f"Bot {browser_id+1} CRITICAL ERROR: {error_message}", browser_id)
       self.run_failed_count += 1
       # Highlight the tab in red
       for i in range(self.browser_logs_tab_widget.count()):
            tab_text = self.browser_logs_tab_widget.tabText(i)
            is_target_tab = tab_text == f"Bot {browser_id + 1}" or tab_text.startswith(f"Bot {browser_id + 1} (")
            if is_target_tab:
                 try:
                     self.browser_logs_tab_widget.tabBar().setTabTextColor(i, QColor("red"))
                     if "(Error)" not in tab_text:
                          self.browser_logs_tab_widget.setTabText(i, f"Bot {browser_id + 1} (Error)")
                 except Exception as e: logging.error(f"Failed to highlight error tab {i}: {e}")
                 break

       if browser_id in self.bot_threads:
           del self.bot_threads[browser_id] # Remove thread ref
       self.check_if_all_finished() # Check if all runs are done


    # --- State/Settings Persistence ---
    def get_current_settings_from_ui(self) -> Dict:
        s = DEFAULT_SETTINGS.copy()
        # Overwrite with UI values
        s["PROXY_ENABLED"] = self.proxy_enabled_check.isChecked()
        s["HEADLESS"] = self.headless_check.isChecked()
        s["MIN_DELAY"] = self.min_delay_spin.value(); s["MAX_DELAY"] = self.max_delay_spin.value()
        s["GEMINI_API_KEY"] = self.gemini_api_key_input.text().strip()
        s["LICENSE_KEY"] = self.license_key_input.text().strip()
        s["FINGERPRINT_PROFILE_NAME"] = self.fingerprint_profile_combo.currentText() or "Random"
        s["DISABLE_AUTOMATION_FLAGS"] = self.disable_automation_flags_check.isChecked()
        s["PREVENT_WEBRTC_IP_LEAK"] = self.prevent_webrtc_leak_check.isChecked() # <<< UPDATED
        s["USER_AGENT_GENERATION_ENABLED"] = self.user_agent_generation_check.isChecked()
        s["USER_AGENT_GENERATION_COUNT"] = self.user_agent_generation_count_spin.value()
        s["CHROMIUM_BLUE_ENABLED"] = self.chromium_blue_check.isChecked()
        s["CHROMIUM_BLUE_PATH"] = self.chromium_blue_path_input.text().strip()
        s["CHROMIUM_BLUE_ARGS"] = self.chromium_blue_args_input.text().strip()
        s["MOUSE_MOVEMENT_ENABLED"] = self.mouse_movement_check.isChecked()
        s["SCROLL_DURATION_MIN"] = self.scroll_duration_min_spin.value()
        s["SCROLL_DURATION_MAX"] = self.scroll_duration_max_spin.value()
        s["ENABLE_BEHAVIORAL_STATES"] = self.behavioral_states_check.isChecked()
        s["SKIP_ACTION_PROBABILITY"] = self.skip_action_prob_spin.value()
        s["FORM_FILL_ENABLED"] = self.form_fill_check.isChecked()
        s["IMPRESSION_ENABLED"] = self.impression_enabled_check.isChecked()
        s["NEXT_PAGE_SELECTORS"] = [ln.strip() for ln in self.next_page_selectors_edit.toPlainText().splitlines() if ln.strip()]
        s["NEXT_PAGE_TEXT_FALLBACK"] = [t.strip() for t in self.next_page_text_fallback_edit.text().split(',') if t.strip()]
        s["AD_CLICK_ENABLED"] = self.ad_click_enabled_check.isChecked()
        s["AD_CLICK_PROBABILITY"] = self.ad_click_probability_spin.value()
        s["AD_SELECTORS"] = [ln.strip() for ln in self.ad_selectors_edit.toPlainText().splitlines() if ln.strip()]
        s["TOTAL_RUNS"] = self.total_runs_spinbox.value(); s["RUN_GROUP_SIZE"] = self.run_group_size_spinbox.value()

        # Ensure correct types for settings (handle potential load errors)
        for k in ["VIEWPORT_MIN_WIDTH", "VIEWPORT_MAX_WIDTH", "VIEWPORT_MIN_HEIGHT", "VIEWPORT_MAX_HEIGHT", "SCROLL_DURATION_MIN", "SCROLL_DURATION_MAX", "USER_AGENT_GENERATION_COUNT", "TOTAL_RUNS", "RUN_GROUP_SIZE"]:
            try: s[k] = int(s.get(k, DEFAULT_SETTINGS.get(k, 0)))
            except (ValueError, TypeError): s[k] = DEFAULT_SETTINGS.get(k, 0)
        for k in ["MIN_DELAY", "MAX_DELAY", "SKIP_ACTION_PROBABILITY", "AD_CLICK_PROBABILITY"]:
             try: s[k] = float(s.get(k, DEFAULT_SETTINGS.get(k, 0.0)))
             except (ValueError, TypeError): s[k] = DEFAULT_SETTINGS.get(k, 0.0)
        for k in ["PROXY_FILE", "FINGERPRINT_FILE", "USER_AGENTS_FILE", "GENERATED_USER_AGENTS_FILE", "IMPORTANT_WORDS_FILE", "CHROMIUM_BLUE_PATH", "CHROMIUM_BLUE_ARGS", "GEMINI_API_KEY", "LICENSE_KEY", "FINGERPRINT_PROFILE_NAME"]:
             s[k] = str(s.get(k, DEFAULT_SETTINGS.get(k, "")))
        for k in ["PROXY_ENABLED", "HEADLESS", "DISABLE_AUTOMATION_FLAGS", "PREVENT_WEBRTC_IP_LEAK", "USER_AGENT_GENERATION_ENABLED", "CHROMIUM_BLUE_ENABLED", "MOUSE_MOVEMENT_ENABLED", "ENABLE_BEHAVIORAL_STATES", "FORM_FILL_ENABLED", "IMPRESSION_ENABLED", "AD_CLICK_ENABLED"]:
             try: s[k] = bool(s.get(k, DEFAULT_SETTINGS.get(k, False)))
             except: s[k] = DEFAULT_SETTINGS.get(k, False)
        for k in ["NEXT_PAGE_SELECTORS", "NEXT_PAGE_TEXT_FALLBACK", "AD_SELECTORS"]:
             val = s.get(k, DEFAULT_SETTINGS.get(k, []))
             s[k] = list(val) if isinstance(val, (list, tuple)) else DEFAULT_SETTINGS.get(k, [])

        return s

    def save_state(self):
        """Saves UI state and license info to QSettings."""
        logging.info("Saving UI state..."); current_ui_settings = self.get_current_settings_from_ui()
        self.sys_settings.beginGroup("UIState")
        for key, value in current_ui_settings.items():
             if isinstance(value, (list, dict)): # Store complex types as JSON
                 try: self.sys_settings.setValue(key, json.dumps(value))
                 except TypeError as e: logging.error(f"Failed to JSON encode setting '{key}': {e}")
             else: self.sys_settings.setValue(key, value) # Save others directly
        # Save URL text too
        self.sys_settings.setValue("URL_TEXT", self.url_text_edit.toPlainText())
        self.sys_settings.endGroup()

        # Save License state
        self.sys_settings.beginGroup("License")
        self.sys_settings.setValue("activated", self.license_manager._activated)
        if self.license_manager._activated and self.license_manager.expiration_date and self.license_manager._activated_key:
            self.sys_settings.setValue("expiry_ordinal", self.license_manager.expiration_date.toordinal())
            self.sys_settings.setValue("key", self.license_manager._activated_key)
        else: self.sys_settings.remove("expiry_ordinal"); self.sys_settings.remove("key")
        self.sys_settings.endGroup()

        self.sys_settings.setValue("geometry", self.saveGeometry())
        logging.debug("UI state and geometry saved.")

    def load_state(self):
        """Loads UI state from QSettings."""
        logging.info("Loading UI state..."); self.sys_settings.beginGroup("UIState")

        def get_setting(key, default_value, target_type):
            saved_value = self.sys_settings.value(key)
            if saved_value is None: return default_value
            if target_type == list or target_type == dict: # Parse JSON
                 if isinstance(saved_value, str):
                      try: return json.loads(saved_value)
                      except json.JSONDecodeError: return default_value
                 else: return default_value
            if target_type == bool: # Handle bool conversion robustly
                 if isinstance(saved_value, str): return saved_value.lower() in ['true', '1', 'yes', 'on']
                 try: return bool(saved_value)
                 except: return default_value
            try: return target_type(saved_value) # General type conversion
            except (ValueError, TypeError): return default_value

        # Apply loaded settings to UI widgets
        self.proxy_enabled_check.setChecked(get_setting("PROXY_ENABLED", DEFAULT_SETTINGS["PROXY_ENABLED"], bool))
        self.headless_check.setChecked(get_setting("HEADLESS", DEFAULT_SETTINGS["HEADLESS"], bool))
        self.min_delay_spin.setValue(get_setting("MIN_DELAY", DEFAULT_SETTINGS["MIN_DELAY"], float))
        self.max_delay_spin.setValue(get_setting("MAX_DELAY", DEFAULT_SETTINGS["MAX_DELAY"], float))
        self.gemini_api_key_input.setText(get_setting("GEMINI_API_KEY", DEFAULT_SETTINGS["GEMINI_API_KEY"], str))
        # FP/Evasion
        profile_name = get_setting("FINGERPRINT_PROFILE_NAME", DEFAULT_SETTINGS["FINGERPRINT_PROFILE_NAME"], str) # Set combo later
        self.disable_automation_flags_check.setChecked(get_setting("DISABLE_AUTOMATION_FLAGS", DEFAULT_SETTINGS["DISABLE_AUTOMATION_FLAGS"], bool))
        self.prevent_webrtc_leak_check.setChecked(get_setting("PREVENT_WEBRTC_IP_LEAK", DEFAULT_SETTINGS["PREVENT_WEBRTC_IP_LEAK"], bool)) # <<< UPDATED
        self.user_agent_generation_check.setChecked(get_setting("USER_AGENT_GENERATION_ENABLED", DEFAULT_SETTINGS["USER_AGENT_GENERATION_ENABLED"], bool))
        self.user_agent_generation_count_spin.setValue(get_setting("USER_AGENT_GENERATION_COUNT", DEFAULT_SETTINGS["USER_AGENT_GENERATION_COUNT"], int))
        # Behavior
        self.mouse_movement_check.setChecked(get_setting("MOUSE_MOVEMENT_ENABLED", DEFAULT_SETTINGS["MOUSE_MOVEMENT_ENABLED"], bool))
        self.scroll_duration_min_spin.setValue(get_setting("SCROLL_DURATION_MIN", DEFAULT_SETTINGS["SCROLL_DURATION_MIN"], int))
        self.scroll_duration_max_spin.setValue(get_setting("SCROLL_DURATION_MAX", DEFAULT_SETTINGS["SCROLL_DURATION_MAX"], int))
        self.behavioral_states_check.setChecked(get_setting("ENABLE_BEHAVIORAL_STATES", DEFAULT_SETTINGS["ENABLE_BEHAVIORAL_STATES"], bool))
        self.skip_action_prob_spin.setValue(get_setting("SKIP_ACTION_PROBABILITY", DEFAULT_SETTINGS["SKIP_ACTION_PROBABILITY"], float))
        # Interaction
        self.form_fill_check.setChecked(get_setting("FORM_FILL_ENABLED", DEFAULT_SETTINGS["FORM_FILL_ENABLED"], bool))
        self.impression_enabled_check.setChecked(get_setting("IMPRESSION_ENABLED", DEFAULT_SETTINGS["IMPRESSION_ENABLED"], bool))
        self.next_page_selectors_edit.setPlainText("\n".join(get_setting("NEXT_PAGE_SELECTORS", DEFAULT_SETTINGS["NEXT_PAGE_SELECTORS"], list)))
        self.next_page_text_fallback_edit.setText(", ".join(get_setting("NEXT_PAGE_TEXT_FALLBACK", DEFAULT_SETTINGS["NEXT_PAGE_TEXT_FALLBACK"], list)))
        self.ad_click_enabled_check.setChecked(get_setting("AD_CLICK_ENABLED", DEFAULT_SETTINGS["AD_CLICK_ENABLED"], bool))
        self.ad_click_probability_spin.setValue(get_setting("AD_CLICK_PROBABILITY", DEFAULT_SETTINGS["AD_CLICK_PROBABILITY"], float))
        self.ad_selectors_edit.setPlainText("\n".join(get_setting("AD_SELECTORS", DEFAULT_SETTINGS["AD_SELECTORS"], list)))
        # Advanced
        self.chromium_blue_check.setChecked(get_setting("CHROMIUM_BLUE_ENABLED", DEFAULT_SETTINGS["CHROMIUM_BLUE_ENABLED"], bool))
        self.chromium_blue_path_input.setText(get_setting("CHROMIUM_BLUE_PATH", DEFAULT_SETTINGS["CHROMIUM_BLUE_PATH"], str))
        self.chromium_blue_args_input.setText(get_setting("CHROMIUM_BLUE_ARGS", DEFAULT_SETTINGS["CHROMIUM_BLUE_ARGS"], str))
        # Run Config
        self.total_runs_spinbox.setValue(get_setting("TOTAL_RUNS", DEFAULT_SETTINGS["TOTAL_RUNS"], int))
        self.run_group_size_spinbox.setValue(get_setting("RUN_GROUP_SIZE", DEFAULT_SETTINGS["RUN_GROUP_SIZE"], int))
        # URLs
        self.url_text_edit.setPlainText(get_setting("URL_TEXT", "", str))
        self.sys_settings.endGroup() # UIState

        # Load License state
        self.sys_settings.beginGroup("License")
        is_activated_saved = get_setting("activated", False, bool)
        saved_key = get_setting("key", "", str)
        expiry_ordinal = get_setting("expiry_ordinal", None, int) # Load as int
        self.sys_settings.endGroup()

        if is_activated_saved and saved_key and expiry_ordinal:
            try:
                expiry_date = datetime.date.fromordinal(expiry_ordinal)
                is_still_valid, _ = self.license_manager.verify_license(saved_key) # Re-verify
                if is_still_valid:
                     self.license_key_input.setText(saved_key)
                     logging.info(f"Loaded and verified active license key: {saved_key}")
                else: logging.warning(f"Saved license key '{saved_key}' is no longer valid."); self.license_manager._activated = False
            except (TypeError, ValueError) as e: logging.error(f"Error parsing saved license expiry date: {e}"); self.license_manager._activated = False
        else: logging.info("No valid saved license state found."); self.license_manager._activated = False

        # Load window geometry
        geom = self.sys_settings.value("geometry")
        if geom and isinstance(geom, (bytes, bytearray)): self.restoreGeometry(geom)

        # Update UI elements dependent on loaded values AFTER all values are loaded
        self.toggle_chromium_blue_fields(self.chromium_blue_check.checkState())
        self.update_fingerprint_combo() # Populate combo box
        idx = self.fingerprint_profile_combo.findText(profile_name); self.fingerprint_profile_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.update_license_status_display()
        logging.info("UI state loaded.")


    def save_settings_to_file_ui(self):
        filepath = "config/settings.py"
        reply = QMessageBox.question(self, "Save Settings", f"This will overwrite the settings file:\n{filepath}\n\nAre you sure you want to save the current UI settings to this file?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Cancel: return
        try:
            current_ui_settings = self.get_current_settings_from_ui()
            with open(filepath, 'w') as f:
                f.write(f"# Aladdin Traffic Bot Settings - Saved {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
                for key in DEFAULT_SETTINGS.keys(): # Maintain default order
                     if key in current_ui_settings:
                          value = current_ui_settings[key]
                          f.write(f"{key} = {repr(value)}\n")
                     else: f.write(f"# {key} = {repr(DEFAULT_SETTINGS[key])} # Default (Not found in UI)\n")
            QMessageBox.information(self, "Settings Saved", f"Current settings saved to\n{filepath}")
            global settings; settings = load_settings() # Reload global settings
        except Exception as e: QMessageBox.critical(self, "Error Saving Settings", f"Failed to save settings to file:\n{e}"); error_logger.exception("Failed saving settings to file")


    def load_settings_from_file_ui(self):
        filepath = "config/settings.py"
        if not os.path.exists(filepath): QMessageBox.warning(self, "Load Settings", f"Settings file not found:\n{filepath}"); return

        reply = QMessageBox.question(self, "Load Settings", f"This will load settings from:\n{filepath}\n\nUnsaved changes in the UI will be overwritten. Continue?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Cancel: return

        global settings; settings = load_settings() # Reload global settings
        if not settings: QMessageBox.critical(self, "Error Loading Settings", f"Failed to load settings from\n{filepath}\nCheck logs/errors.log."); return
        try:
            # Apply reloaded settings to UI (similar to load_state but uses 'settings' dict)
            self.proxy_enabled_check.setChecked(settings.get("PROXY_ENABLED", DEFAULT_SETTINGS["PROXY_ENABLED"]))
            self.headless_check.setChecked(settings.get("HEADLESS", DEFAULT_SETTINGS["HEADLESS"]))
            self.min_delay_spin.setValue(settings.get("MIN_DELAY", DEFAULT_SETTINGS["MIN_DELAY"])); self.max_delay_spin.setValue(settings.get("MAX_DELAY", DEFAULT_SETTINGS["MAX_DELAY"]))
            self.gemini_api_key_input.setText(settings.get("GEMINI_API_KEY", DEFAULT_SETTINGS["GEMINI_API_KEY"]))
            self.license_key_input.setText(settings.get("LICENSE_KEY", DEFAULT_SETTINGS["LICENSE_KEY"])) # Load key but don't activate

            profile_name = settings.get("FINGERPRINT_PROFILE_NAME", DEFAULT_SETTINGS["FINGERPRINT_PROFILE_NAME"])
            self.update_fingerprint_combo() # Update combo content first
            idx = self.fingerprint_profile_combo.findText(profile_name); self.fingerprint_profile_combo.setCurrentIndex(idx if idx != -1 else 0)

            self.disable_automation_flags_check.setChecked(settings.get("DISABLE_AUTOMATION_FLAGS", DEFAULT_SETTINGS["DISABLE_AUTOMATION_FLAGS"]))
            self.prevent_webrtc_leak_check.setChecked(settings.get("PREVENT_WEBRTC_IP_LEAK", DEFAULT_SETTINGS["PREVENT_WEBRTC_IP_LEAK"])) # <<< UPDATED
            self.user_agent_generation_check.setChecked(settings.get("USER_AGENT_GENERATION_ENABLED", DEFAULT_SETTINGS["USER_AGENT_GENERATION_ENABLED"]))
            self.user_agent_generation_count_spin.setValue(settings.get("USER_AGENT_GENERATION_COUNT", DEFAULT_SETTINGS["USER_AGENT_GENERATION_COUNT"]))
            self.mouse_movement_check.setChecked(settings.get("MOUSE_MOVEMENT_ENABLED", DEFAULT_SETTINGS["MOUSE_MOVEMENT_ENABLED"]))
            self.scroll_duration_min_spin.setValue(settings.get("SCROLL_DURATION_MIN", DEFAULT_SETTINGS["SCROLL_DURATION_MIN"]))
            self.scroll_duration_max_spin.setValue(settings.get("SCROLL_DURATION_MAX", DEFAULT_SETTINGS["SCROLL_DURATION_MAX"]))
            self.behavioral_states_check.setChecked(settings.get("ENABLE_BEHAVIORAL_STATES", DEFAULT_SETTINGS["ENABLE_BEHAVIORAL_STATES"]))
            self.skip_action_prob_spin.setValue(settings.get("SKIP_ACTION_PROBABILITY", DEFAULT_SETTINGS["SKIP_ACTION_PROBABILITY"]))
            self.form_fill_check.setChecked(settings.get("FORM_FILL_ENABLED", DEFAULT_SETTINGS["FORM_FILL_ENABLED"]))
            self.impression_enabled_check.setChecked(settings.get("IMPRESSION_ENABLED", DEFAULT_SETTINGS["IMPRESSION_ENABLED"]))
            self.next_page_selectors_edit.setPlainText("\n".join(settings.get("NEXT_PAGE_SELECTORS", DEFAULT_SETTINGS["NEXT_PAGE_SELECTORS"])))
            self.next_page_text_fallback_edit.setText(", ".join(settings.get("NEXT_PAGE_TEXT_FALLBACK", DEFAULT_SETTINGS["NEXT_PAGE_TEXT_FALLBACK"])))
            self.ad_click_enabled_check.setChecked(settings.get("AD_CLICK_ENABLED", DEFAULT_SETTINGS["AD_CLICK_ENABLED"]))
            self.ad_click_probability_spin.setValue(settings.get("AD_CLICK_PROBABILITY", DEFAULT_SETTINGS["AD_CLICK_PROBABILITY"]))
            self.ad_selectors_edit.setPlainText("\n".join(settings.get("AD_SELECTORS", DEFAULT_SETTINGS["AD_SELECTORS"])))
            self.chromium_blue_check.setChecked(settings.get("CHROMIUM_BLUE_ENABLED", DEFAULT_SETTINGS["CHROMIUM_BLUE_ENABLED"]))
            self.chromium_blue_path_input.setText(settings.get("CHROMIUM_BLUE_PATH", DEFAULT_SETTINGS["CHROMIUM_BLUE_PATH"]))
            self.chromium_blue_args_input.setText(settings.get("CHROMIUM_BLUE_ARGS", DEFAULT_SETTINGS["CHROMIUM_BLUE_ARGS"]))
            self.total_runs_spinbox.setValue(settings.get("TOTAL_RUNS", DEFAULT_SETTINGS["TOTAL_RUNS"]))
            self.run_group_size_spinbox.setValue(settings.get("RUN_GROUP_SIZE", DEFAULT_SETTINGS["RUN_GROUP_SIZE"]))

            # Update dependent UI states
            self.toggle_chromium_blue_fields(self.chromium_blue_check.checkState())
            # Re-verify license based on key loaded from file, update status display
            loaded_license_key = settings.get("LICENSE_KEY", "")
            if loaded_license_key: self.license_manager.verify_license(loaded_license_key)
            else: self.license_manager._activated = False # Ensure inactive if key is empty
            self.update_license_status_display()

            # Reload FM with potentially new API key from file
            self.fingerprint_manager = FingerprintManager(user_agents, load_generated_user_agents(), settings.get("GEMINI_API_KEY"))

            QMessageBox.information(self, "Settings Loaded", f"Settings loaded successfully from\n{filepath}")
        except Exception as e: QMessageBox.critical(self, "Error Applying Settings", f"Failed to apply loaded settings to UI:\n{e}"); error_logger.exception("Error applying settings from file to UI")


    def closeEvent(self, event):
        """Handles window close event: saves state, confirms exit if bots running."""
        logging.info("Close event triggered.")
        self.save_state() # Save UI state first

        active_threads = [t for t in self.bot_threads.values() if t and t.isRunning()]
        confirm_msg = "Are you sure you want to exit?"
        # Change message if bots are active
        if active_threads:
             confirm_msg = f"{len(active_threads)} bot(s) are currently running.\n\nStopping them might take a moment.\n\nAre you sure you want to exit?"

        reply = QMessageBox.question(self, 'Confirm Exit', confirm_msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No if active_threads else QMessageBox.StandardButton.Yes)

        if reply == QMessageBox.StandardButton.Yes:
            logging.info("Close accepted. Stopping any active bots and closing application.");
            if active_threads:
                self.stop_all_bots() # Attempt graceful shutdown
                # Give stop_all_bots a very short time to process signals
                QApplication.processEvents()
                time.sleep(0.5) # Small delay to allow threads to potentially receive stop
            event.accept() # Allow window to close
            logging.info("Application closing now.")
        else:
            logging.info("Close ignored by user.");
            event.ignore() # Prevent window from closing


# --- Main Execution ---
def check_playwright_install():
     """Checks if Playwright browsers seem installed and usable."""
     try:
        logging.debug("Checking Playwright browser installation...");
        with sync_playwright() as p:
            # Try launching with minimal args, handle potential errors
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
            browser.close()
        logging.info("Playwright browser check successful."); return True
     except PlaywrightError as pe:
         # More specific error messages
        err_str = str(pe)
        if "Executable doesn't exist" in err_str:
            err_msg = "Playwright browser executable not found.\nPlease run 'playwright install' in your terminal/command prompt."
        elif "Host system is missing dependencies" in err_str:
            err_msg = "Missing OS dependencies for Playwright browsers.\nPlease run 'playwright install-deps' in your terminal (may require sudo/admin)."
        else:
            err_msg = f"Playwright browser launch failed.\nTry running 'playwright install' in your terminal.\n\n(Error: {err_str.splitlines()[0]})"
        logging.error(err_msg)
        # Show error in a message box *before* main app window if check fails
        QMessageBox.critical(None, "Playwright Installation Error", err_msg)
        return False
     except Exception as e:
        err_msg = f"Unexpected error during Playwright check.\nTry running 'playwright install'.\n\n(Error: {e})"
        logging.error(err_msg, exc_info=True)
        QMessageBox.critical(None, "Playwright Error", err_msg)
        return False

def main():
    """Entry point for the GUI application."""
    # Setup logging level based on environment variable or default
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(log_level) # Set level for root logger
    # Ensure file handlers respect this level too
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(log_level)
    logging.info(f"--- Application Starting --- Log level: {logging.getLevelName(log_level)}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Apply style

    # Consolidated Stylesheet
    app.setStyleSheet("""
        QWidget { font-size: 10pt; }
        QGroupBox {
            font-size: 11pt; font-weight: bold;
            border: 1px solid #B0B0B0; border-radius: 6px;
            margin-top: 12px; padding-top: 10px; /* Added padding-top */
            background-color: #FDFDFD;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top center;
            padding: 4px 10px; background-color: #EAEAEA;
            border: 1px solid #B0B0B0; border-radius: 4px;
            color: #222; font-weight: bold;
        }
        QTabBar::tab {
            background: #E0E0E0; color: #333; border: 1px solid #B0B0B0;
            padding: 9px 20px; border-bottom: none;
            border-top-left-radius: 5px; border-top-right-radius: 5px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #FDFDFD; color: #000;
            border-bottom: 1px solid #FDFDFD; font-weight: bold;
        }
        QTabWidget::pane {
            border: 1px solid #B0B0B0; border-top: none;
            background-color: #FDFDFD;
        }
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid #BDBDBD; border-radius: 4px; padding: 6px;
            background-color: #FFF;
            selection-background-color: #AAD; selection-color: #FFF;
        }
        QTextEdit { min-height: 50px; }
        QPushButton {
            background-color: #D8D8D8; border: 1px solid #A0A0A0;
            padding: 8px 16px; border-radius: 4px; font-weight: bold;
        }
        QPushButton:hover { background-color: #E2E2E2; }
        QPushButton:pressed { background-color: #C8C8C8; }
        QPushButton:disabled { background-color: #EDEDED; color: #999; border-color: #C0C0C0; }
        /* Specific buttons styled by object name */
        QPushButton#start_button { background-color: #4CAF50; color: white; border-color: #388E3C; }
        QPushButton#start_button:hover { background-color: #66BB6A; }
        QPushButton#start_button:pressed { background-color: #388E3C; }
        QPushButton#stop_button { background-color: #f44336; color: white; border-color: #D32F2F; }
        QPushButton#stop_button:hover { background-color: #E57373; }
        QPushButton#stop_button:pressed { background-color: #D32F2F; }
        QPushButton#generate_ua_button, QPushButton#generate_fp_button { /* Gemini buttons */
            background-color: #8E44AD; color: white; border-color: #7B2496;
        }
        QPushButton#generate_ua_button:hover, QPushButton#generate_fp_button:hover {
            background-color: #9B59B6;
        }
        QPushButton#generate_ua_button:pressed, QPushButton#generate_fp_button:pressed {
            background-color: #7B2496;
        }
        QListWidget { border: 1px solid #B0B0B0; border-radius: 4px; }
        QListWidget::item:alternate { background-color: #F9F9F9; }
        QScrollArea { border: none; }
        QLabel { padding-top: 4px; padding-bottom: 2px; }
        BoldLabel { font-weight: bold; }
        QFormLayout { label-alignment: AlignRight; vertical-spacing: 8px; }
        QMessageBox { font-size: 10pt; } /* Ensure message boxes use base font size */
    """)

    # Perform Playwright check *after* QApplication is initialized for QMessageBox
    if not check_playwright_install():
        logging.critical("Playwright browser check failed. Exiting.")
        sys.exit(1)

    try:
        window = MainWindow()
        window.show()
        logging.info("Main window displayed.")
        exit_code = app.exec()
        logging.info(f"--- Application Exited --- Code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
         logging.exception("Fatal error during app startup or execution.")
         try: QMessageBox.critical(None, "Fatal Error", f"A critical error occurred:\n{e}\n\nPlease check logs/errors.log for details.")
         except: pass # If GUI fails very early, message box might also fail
         sys.exit(1)

if __name__ == "__main__":
    # Ensure structure and defaults are set before anything else
    _ensure_structure()
    # Proceed to main application entry point
    main()