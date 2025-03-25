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

# --- Other Libraries ---
try:
    from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext, Playwright, Error as PlaywrightError
    import google.generativeai as genai  # For Gemini API integration
except ImportError:
    print("Missing required libraries.  Please run 'pip install -r requirements.txt'")
    sys.exit(1)

# --- Project Structure --- (remains unchanged)
def _ensure_structure():
    for dir_path in ("core", "config", "logs", "data", "tests", "resources"):
        os.makedirs(dir_path, exist_ok=True)
    for file_path in ("config/settings.py", "config/proxies.txt", "logs/bot_activity.log", "logs/errors.log",
                      "data/user_agents.json", "data/important_words.json", "data/generated_user_agents.json"):
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                if file_path.endswith(".json"):
                    f.write("{}")
                else:
                    f.write("")

_ensure_structure()

#Copy logo
if not os.path.exists("resources/chromium_blue.png"):
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (64, 64), color='blue')
        img.save('resources/chromium_blue.png')
    except Exception as e:
        print(f"Error during create a simple blue logo: {e}. Please provide a valid logo file.")

# --- Logging Setup --- (remains unchanged)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bot_activity.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler("logs/errors.log")
error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)


# --- Configuration (config/settings.py) --- (remains unchanged)
DEFAULT_SETTINGS = {
    "PROXY_ENABLED": False,
    "PROXY_TYPE": "http",
    "PROXY_FILE": "config/proxies.txt",
    "HEADLESS": True,
    "MIN_DELAY": 2,
    "MAX_DELAY": 5,
    "GEMINI_API_KEY": "",
    "LICENSE_KEY": "",
    "USER_AGENTS_FILE": "data/user_agents.json",
    "IMPORTANT_WORDS_FILE": "data/important_words.json",
    "VIEWPORT_MIN_WIDTH": 800,
    "VIEWPORT_MAX_WIDTH": 1920,
    "VIEWPORT_MIN_HEIGHT": 600,
    "VIEWPORT_MAX_HEIGHT": 1080,
    "CHROMIUM_BLUE_ENABLED": False,
    "CHROMIUM_BLUE_PATH": "",
    "CHROMIUM_BLUE_ARGS": "",
    "MOUSE_MOVEMENT_ENABLED": True,
    "CONCURRENT_BROWSERS": 1,
    "SCROLL_DURATION_MIN": 500,
    "SCROLL_DURATION_MAX": 1500,
    "FORM_FILL_ENABLED": False,
    "IMPRESSION_ENABLED": False,
    "NEXT_PAGE_SELECTOR": ".next-page, .next",
    "AD_CLICK_ENABLED": False,
    "AD_SELECTOR": ".ad-link, .advertisement",
    "AD_CLICK_PROBABILITY": 0.1,
    "TOTAL_RUNS": 1,
    "RUN_GROUP_SIZE": 1,
    "USER_AGENT_GENERATION_ENABLED": False,
    "GENERATED_USER_AGENTS_FILE": "data/generated_user_agents.json",
    "USER_AGENT_GENERATION_COUNT": 10,
}


def load_settings() -> Dict:
    """Loads settings from config/settings.py."""
    try:
        with open("config/settings.py", "r") as f:
            settings_module = {}
            exec(f.read(), settings_module)
            settings = DEFAULT_SETTINGS.copy()
            settings.update({k: v for k, v in settings_module.items() if k.isupper()})
            return settings
    except (FileNotFoundError, Exception) as e:
        error_logger.error(f"Error loading settings: {e}. Using default settings.")
        return DEFAULT_SETTINGS


settings = load_settings()


# --- Load Data Files --- (remains unchanged, but included for completeness)

def load_user_agents() -> List[str]:
    """Loads user agents from file."""
    try:
        with open(settings["USER_AGENTS_FILE"], "r") as f:
            return json.load(f).get("user_agents", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        error_logger.error(f"Error loading user agents: {e}. Returning [].")
        return []

def load_generated_user_agents() -> List[str]:
    """Loads generated user agents."""
    try:
        with open(settings["GENERATED_USER_AGENTS_FILE"], "r") as f:
            return json.load(f).get("generated_user_agents", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        error_logger.error(f"Error loading generated agents: {e}. Returning [].")
        return []
    except Exception as e:
        error_logger.error(f"An unexpected error occurred: {e}")
        return []

def save_generated_user_agents(user_agents_list: List[str]):
    """Saves generated user agents."""
    try:
        with open(settings["GENERATED_USER_AGENTS_FILE"], "w") as f:
            json.dump({"generated_user_agents": user_agents_list}, f, indent=4)
    except Exception as e:
        error_logger.error(f"Error saving generated user agents: {e}")


user_agents = load_user_agents()
generated_user_agents = load_generated_user_agents()


def load_important_words() -> List[str]:
    """Loads important words from file."""
    try:
        with open(settings["IMPORTANT_WORDS_FILE"], "r") as f:
            return json.load(f).get("important_words", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        error_logger.error(f"Error loading important words: {e}. Returning [].")
        return []
    except Exception as e: # added to handle unexpected errors
        error_logger.error(f"An unexpected error occurred: {e}")
        return []

def load_proxies() -> List[Tuple[str, str]]:
    """Loads proxies from the configured proxy file (config/proxies.txt)."""
    proxies: List[Tuple[str, str]] = []  # List of (proxy_address, proxy_type) tuples
    try:
        with open(settings["PROXY_FILE"], "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    try:
                        # Attempt to split into address and type
                        parts = line.split()
                        address = parts[0]
                        proxy_type = "http"  # Default type
                        if len(parts) > 1:
                            proxy_type = parts[1].lower()
                            if proxy_type not in ("http", "https", "socks4", "socks5"):
                                error_logger.warning(f"Invalid type '{proxy_type}' in line: {line}.  Skipping.")
                                continue

                        proxies.append((address, proxy_type))
                    except Exception as e:
                        error_logger.warning(f"Error parsing proxy line: {line}.  Skipping. Error: {e}")
    except FileNotFoundError:
        error_logger.error(f"Proxy file not found: {settings['PROXY_FILE']}")
    except Exception as e:
        error_logger.error(f"Error loading proxies: {e}")
    return proxies


important_words = load_important_words()
# --- Core Classes and Functions ---
class BrowserManager:
    """Manages Chromium/Chromium Blue browser instances."""

    def __init__(self, playwright: Playwright, proxy_server: Optional[str] = None,
                 proxy_type: Optional[str] = None,
                 headless: bool = settings["HEADLESS"],
                 use_chromium_blue: bool = settings["CHROMIUM_BLUE_ENABLED"],
                 chromium_blue_path: str = settings["CHROMIUM_BLUE_PATH"],
                 chromium_blue_args: str = settings["CHROMIUM_BLUE_ARGS"]):
        self.playwright = playwright
        self.proxy_server = proxy_server
        self.proxy_type = proxy_type.lower() if proxy_type else None
        self.headless = headless
        self.use_chromium_blue = use_chromium_blue
        self.chromium_blue_path = chromium_blue_path
        self.chromium_blue_args = chromium_blue_args.split() if chromium_blue_args else []
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def start_browser(self, user_agent: str = None, viewport_size: Dict = None):
        """Starts a new browser instance (Chromium or Chromium Blue)."""
        launch_options = {
            "headless": self.headless,
        }

        if self.proxy_server:
            if self.proxy_type in ("http", "https", "socks4", "socks5"):
                launch_options["proxy"] = {"server": f"{self.proxy_type}://{self.proxy_server}"}
            elif self.proxy_type is not None:
                raise ValueError(f"Invalid proxy type: {self.proxy_type}")

        if self.use_chromium_blue:
            if not os.path.exists(self.chromium_blue_path):
                raise FileNotFoundError(f"Chromium Blue not found at {self.chromium_blue_path}")
            launch_options["executable_path"] = self.chromium_blue_path
            launch_options["args"] = self.chromium_blue_args

        try:
            browser_type = self.playwright.chromium
            self.browser = browser_type.launch(**launch_options)
            self.context = self.browser.new_context(
                user_agent=user_agent,
                viewport=viewport_size
            )
            self.page = self.context.new_page()

            self.page.set_extra_http_headers({'X-Forwarded-For': generate_random_ip()})
            logging.info(f"Browser started. Headless: {self.headless}, Proxy: {self.proxy_server}, Type: {self.proxy_type}, Chromium Blue: {self.use_chromium_blue}")
        except PlaywrightError as e:
            error_logger.error(f"Failed to start browser: {e}")
            raise
        except FileNotFoundError as e:
            error_logger.error(str(e))
            raise
        except Exception as e:
            error_logger.error(f"Unexpected error during browser start: {e}")
            raise

    def navigate_to(self, url: str):
        """Navigates to the given URL."""
        if not self.page:
            raise Exception("Browser not started. Call start_browser() first.")
        try:
            self.page.goto(url, timeout=60000)
            logging.info(f"Navigated to: {url}")
        except PlaywrightError as e:
            error_logger.error(f"Navigation error: {e}")
            raise
        except Exception as e:
            error_logger.error(f"Unexpected error: {e}")
            raise


    def close_browser(self):
        """Closes the browser instance."""
        if self.browser:
            self.browser.close()
            self.browser = None
            self.context = None
            self.page = None
            logging.info("Browser closed.")

    def take_screenshot(self, filename: str = "screenshot.png"):
        """Takes a screenshot of the current page."""
        if self.page:
            try:
                screenshot_path = os.path.join("logs", filename)
                self.page.screenshot(path=screenshot_path)
                logging.info(f"Screenshot saved to {screenshot_path}")
            except PlaywrightError as e:
                error_logger.error(f"Error taking screenshot: {e}")
            except Exception as e:
                error_logger.error(f"Unexpected error while taking a screen shot: {e}")


class FingerprintManager:
    """Manages User Agents, including generation."""

    def __init__(self, initial_user_agents: List[str], generated_user_agents: List[str], gemini_api_key: str):
        self.user_agents = initial_user_agents
        self.generated_user_agents = generated_user_agents
        self.gemini_api_key = gemini_api_key
        self.used_user_agents_in_run = set()

    def get_random_viewport_size(self) -> Dict:
        """Returns a random viewport size."""
        width = random.randint(settings["VIEWPORT_MIN_WIDTH"], settings["VIEWPORT_MAX_WIDTH"])
        height = random.randint(settings["VIEWPORT_MIN_HEIGHT"], settings["VIEWPORT_MAX_HEIGHT"])
        return {"width": width, "height": height}

    def get_user_agent(self, browser_id, update_signal) -> str:
        """Selects a unique random user agent, generates more if needed."""

        available_agents = [ua for ua in self.user_agents + self.generated_user_agents if ua not in self.used_user_agents_in_run]

        if not available_agents:
            if settings["USER_AGENT_GENERATION_ENABLED"]:
                update_signal.emit(f"Browser {browser_id}: All user agents exhausted. Generating...", browser_id)
                self.generate_user_agents_gemini(browser_id, update_signal)
                self.generated_user_agents = load_generated_user_agents()
                available_agents = [ua for ua in self.user_agents + self.generated_user_agents if ua not in self.used_user_agents_in_run]

                if not available_agents:
                    error_logger.error(f"Browser {browser_id}: Failed to generate new user agents.")
                    update_signal.emit(f"Browser {browser_id}: Warning: Failed to generate. Using default.", browser_id)
                    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/Default"
            else:
                update_signal.emit(f"Browser {browser_id}: Warning: All user agents used, generation disabled. Default.", browser_id)
                return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/Default"

        chosen_agent = random.choice(available_agents)
        self.used_user_agents_in_run.add(chosen_agent)
        return chosen_agent


    def generate_user_agents_gemini(self, browser_id, update_signal):
        """Generates new user agents using Gemini API and saves them."""
        if not self.gemini_api_key:
            update_signal.emit(f"Browser {browser_id}: Gemini API key missing. Cannot generate.", browser_id)
            return

        prompt = f"""
        Generate a list of {settings["USER_AGENT_GENERATION_COUNT"]} diverse and realistic user agent strings for web browsers.
        Include a variety of browser types (Chrome, Firefox, Safari, Edge), operating systems (Windows, macOS, Linux, Android, iOS), and versions.
        Each user agent should be on a new line.
        """

        try:
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            generated_ua_text = response.text

            new_user_agents = [ua.strip() for ua in generated_ua_text.strip().splitlines() if ua.strip()]

            if len(new_user_agents) < settings["USER_AGENT_GENERATION_COUNT"]:
                update_signal.emit(f"Browser {browser_id}: Gemini generated fewer agents than requested. {len(new_user_agents)}/{settings['USER_AGENT_GENERATION_COUNT']}", browser_id)

            if new_user_agents:
                current_generated_agents = load_generated_user_agents()
                updated_agents = current_generated_agents + new_user_agents
                save_generated_user_agents(updated_agents)
                update_signal.emit(f"Browser {browser_id}: Generated and saved {len(new_user_agents)} new user agents.", browser_id)
            else:
                update_signal.emit(f"Browser {browser_id}: Gemini API returned no user agents.", browser_id)


        except Exception as e:
            error_logger.error(f"Browser {browser_id}: Error generating user agents with Gemini: {e}")
            update_signal.emit(f"Browser {browser_id}: Error from Gemini API: {e}", browser_id)
            update_signal.emit(f"Browser {browser_id}: Gemini API Details: {e}", browser_id)


class ScrollingManager:
    """Handles human-like scrolling behavior."""

    def __init__(self, page: Page):
        self.page = page


    def random_scroll(self, min_scrolls: int = 3, max_scrolls: int = 6, duration_min:int = settings["SCROLL_DURATION_MIN"], duration_max:int = settings["SCROLL_DURATION_MAX"]):
        """Performs a series of random scrolls, mimicking human behavior, with variable scroll counts and durations."""
        num_scrolls = random.randint(min_scrolls, max_scrolls)
        viewport_height = self.page.viewport_size["height"]
        total_scroll_time = 0
        for _ in range(num_scrolls):
            if total_scroll_time >= 40000: # 40 seconds
                break
            scroll_amount = random.randint(int(viewport_height * 0.3), int(viewport_height * 0.8))
            scroll_direction = random.choice([1, -1])
            scroll_to = self.page.evaluate("window.pageYOffset") + scroll_amount * scroll_direction
            max_scroll = self.page.evaluate("document.body.scrollHeight - window.innerHeight")
            scroll_to = max(0, min(scroll_to, max_scroll))
            duration = random.randint(duration_min, duration_max)
            self.smooth_scroll_to(scroll_to, duration)
            scroll_delay = duration / 1000 + random.uniform(0.5, 1.5)
            total_scroll_time += (duration + scroll_delay*1000)
            random_delay(scroll_delay, scroll_delay + 1)

    def smooth_scroll_to(self, scroll_to, duration):
        """Smoothly scrolls to the given position using JS, similar to random_scroll."""

        current_scroll = self.page.evaluate("window.pageYOffset")
        scroll_distance = abs(scroll_to - current_scroll)

        # If the scroll distance is very small, just jump to it
        if scroll_distance < 50:
            self.page.evaluate(f"window.scrollTo(0, {scroll_to})")
            return
        self.page.evaluate(f"""
            (function() {{
                const start = window.pageYOffset;
                const change = {scroll_to} - start;
                const startTime = performance.now();

                function easeInOutQuad(t) {{
                    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
                }}

                function animateScroll() {{
                    const currentTime = performance.now();
                    const time = Math.min(1, (currentTime - startTime) / {duration});
                    const easedTime = easeInOutQuad(time);
                    window.scrollTo(0, start + change * easedTime);

                    if (time < 1) {{
                        requestAnimationFrame(animateScroll);
                    }}
                }}

                requestAnimationFrame(animateScroll);
            }})();
        """)
        random_delay(duration/1000, duration/1000 + 0.5)


    def gemini_scroll(self, browser_id, update_signal):
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
        else:  # Larger page
            scroll_instruction_prompt = """
            Describe a natural, varied human-like scroll action on a webpage for approximately 60 seconds.
            Simulate a person browsing, reading, and occasionally going back to re-read sections.
            Include a mix of:
                - Varied scroll speeds (fast for skimming, slow for reading)
                - Short pauses (simulating reading or thinking)
                - Changes in direction (scroll up briefly to re-read)
                - Longer pauses (simulating focused reading of a section)
                - Movement towards the bottom/footer of the page, then back up

            Give instructions *strictly* in the following format, and *only* this format:
            'Scroll down [pixels] pixels over [seconds] seconds, pause [seconds] seconds' OR
            'Scroll up [pixels] pixels over [seconds] seconds, pause [seconds] seconds'

            Provide a sequence of instructions that will take approximately 60 seconds to execute. Separate instructions with commas. Do *not* number the instructions.

            Example (this is just an example, generate a *different*, 60-second sequence):
            'Scroll down 300 pixels over 1.0 seconds, pause 0.5 seconds, Scroll down 500 pixels over 1.8 seconds, pause 1.2 seconds, Scroll up 100 pixels over 0.4 seconds, pause 0.3 seconds, Scroll down 250 pixels over 0.8 seconds, pause 0.7 seconds, Scroll down 700 pixels over 2.5 seconds, pause 1.5 seconds, Scroll up 200 pixels over 0.6 seconds, pause 0.4 seconds, ..., Scroll down 400 pixels over 1.3 seconds, pause 0.9 seconds'  (continue in this format for ~60 seconds total)

            Be concise. Do not include any introductory text or explanations. The output should be *only* the comma-separated instructions.
            """

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


class TextSelectionManager:
    """Handles text selection on the page."""

    def __init__(self, page: Page):
        self.page = page

    def select_important_text(self, browser_id, update_signal):
        """Selects text containing important words (e.g., price, date) on the page."""
        if settings["MOUSE_MOVEMENT_ENABLED"]:  # Check the setting.
          self.gemini_mouse_movement(browser_id, update_signal) # if enabled, use gemini mouse
        else: # otherwise use a simple random mouse move
            self.simple_random_mouse_movement_header(browser_id, update_signal)

    def select_random_text(self):
        """Selects a random portion of text on the page."""
        pass #removed as we are using only gemini mouse movements

    def simple_random_mouse_movement_header(self, browser_id, update_signal):
        """Simple fallback mouse movement - targets header area, robust header detection."""
        try:
            # Try to locate the header using various selectors
            header_element = None
            for selector in ['header', '[role="banner"]', '.header', '.navbar', '#header', '#navbar']:
                try:
                    header_element = self.page.locator(selector).first
                    if header_element.is_visible():
                        break  # Found a header, exit loop
                except:
                    pass

            if header_element:
                bounding_box = header_element.bounding_box()
                if bounding_box:
                    header_x = bounding_box['x'] + random.randint(50, int(bounding_box['width'] - 50))
                    header_y = bounding_box['y'] + int(bounding_box['height'] / 2)
                else:
                    update_signal.emit(f"Browser {browser_id}: Header element found but has no bounding box. Using fallback.", browser_id)
                    header_x = random.randint(50, 300)
                    header_y = random.randint(20, 60)
            else:
                update_signal.emit(f"Browser {browser_id}: No header element found, using fallback coordinates.", browser_id)
                header_x = random.randint(50, 300)
                header_y = random.randint(20, 60)

            self.page.mouse.move(header_x, header_y)
            random_delay(1, 2)
            update_signal.emit(f"Browser {browser_id}: Using simple random mouse movement (header fallback).", browser_id)
        except Exception as e:
            update_signal.emit(f"Browser {browser_id}: Error in fallback mouse movement: {e}", browser_id)


    def gemini_mouse_movement(self, browser_id, update_signal):
        """Generates a human-like mouse path using Gemini API - targeting header area."""
        page_width = self.page.evaluate("document.body.scrollWidth")
        page_height = self.page.evaluate("document.body.scrollHeight")

        # Target the header area (adjust coordinates as needed for different websites)
        header_y_coordinate = 80  # Adjusted: Assuming header is within top 80 pixels
        header_x_start = 50
        header_x_end = page_width - 50

        prompt = f"Describe a very short, natural human-like mouse movement path. Start at x={header_x_start}, y={header_y_coordinate} (within website header area). Briefly move horizontally within the header area, then pause. Describe as a sequence of x,y coordinates and very short pauses. Example: 'x=100, y={header_y_coordinate}, 0.1; x=200, y={header_y_coordinate}, 0.15; x=150, y={header_y_coordinate}, 0.2'. Be very concise."

        try:
            response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
            mouse_path_instructions_text = response.text
            update_signal.emit(f"Browser {browser_id}: Gemini Mouse Path Instructions (Header): {mouse_path_instructions_text}", browser_id)
            self.execute_mouse_path_instructions(mouse_path_instructions_text, browser_id, update_signal) # Execute here
        except Exception as e:
            update_signal.emit(f"Browser {browser_id}: Error generating mouse path from Gemini API: {e}", browser_id)  # More specific error message
            update_signal.emit(f"Browser {browser_id}: Gemini API Error Details: {e}", browser_id)  # Print exception details
            self.simple_random_mouse_movement_header(browser_id, update_signal) # Fallback

    def execute_mouse_path_instructions(self, instructions_text, browser_id, update_signal):
        """Executes mouse path instructions generated by Gemini."""
        if not instructions_text:
            self.simple_random_mouse_movement_header(browser_id, update_signal)  # Fallback
            return

        current_x = 0
        current_y = 0
        try:
            instructions = instructions_text.strip().split(';')  # Semicolon separated instructions
            for instruction in instructions:
                instruction = instruction.strip()
                parts = instruction.split(',')  # e.g., ['x=150', ' y=200', ' 0.2']
                if len(parts) == 3 :
                    try:
                        x = int(parts[0].split('=')[1])  # Extract x value
                        y = int(parts[1].split('=')[1])  # Extract y value
                        pause_duration = float(parts[2])  # Extract pause duration

                        # Calculate relative movement
                        move_x = x - current_x
                        move_y = y - current_y

                        self.page.mouse.move(x, y, steps=20)  # Use absolute move with steps for smoothness
                        current_x = x
                        current_y = y
                        time.sleep(pause_duration)


                    except (ValueError, IndexError) as e:
                        update_signal.emit(f"Browser {browser_id}: Error parsing mouse instruction: {instruction}, error: {e}", browser_id)
                        continue  # Skip invalid instruction
                else:
                                        update_signal.emit(f"Browser {browser_id}: Invalid mouse instruction format: {instruction}", browser_id)

        except Exception as e:
            update_signal.emit(f"Browser {browser_id}: Error executing mouse path instructions: {e}", browser_id)
            self.simple_random_mouse_movement_header(browser_id, update_signal)  # Fallback on error


    def human_mouse_move(self, x, y, steps=None):
        """Moves the mouse to (x, y) with a more human-like trajectory."""
        # This function is now primarily used for the selection.  The core
        # movement is now handled by gemini_mouse_movement and
        # execute_mouse_path_instructions, so we just use page.mouse.move.
        if not settings["MOUSE_MOVEMENT_ENABLED"]:
             self.page.mouse.move(x,y)
             return
        self.page.mouse.move(x, y, steps=steps or 20)  # Ensure steps is defined


class NextPageNavigator:
    """Handles navigation to the next page."""

    def __init__(self, page: Page):
        self.page = page
        self.next_page_selector = settings["NEXT_PAGE_SELECTOR"]

    def navigate_next_page(self, browser_id, update_signal):
        """Navigates to the next page if found."""
        if not settings["IMPRESSION_ENABLED"]:
            return

        try:
            next_page_element = self.page.locator(self.next_page_selector).first
            if next_page_element.is_visible():
                update_signal.emit(f"Browser {browser_id}: Navigating to the  next page.", browser_id)
                next_page_element.click(timeout=15000)
                random_delay()
                update_signal.emit(f"Browser {browser_id}: Navigated to the next page.", browser_id)
                return True
            else:
                update_signal.emit(f"Browser {browser_id}: Next page element not found.", browser_id)
                return False
        except PlaywrightError as e:
            update_signal.emit(f"Browser {browser_id}: Error navigating: {e}", browser_id)
            return False
        except Exception as e:
            update_signal.emit(f"Browser {browser_id}: Unexpected error: {e}", browser_id)
            return False

class AdClickManager:
    """Handles ad detection and clicking."""

    def __init__(self, page: Page):
        self.page = page
        self.ad_selector = settings["AD_SELECTOR"]
        self.ad_click_probability = settings["AD_CLICK_PROBABILITY"]

    def click_ad(self, browser_id, update_signal):
        """Attempts to find and click an ad based on probability."""
        if not settings["AD_CLICK_ENABLED"]:
            return False

        if random.random() >= self.ad_click_probability:
            update_signal.emit(f"Browser {browser_id}: Not clicking ad (probability).", browser_id)
            return False

        try:
            ad_elements = self.page.locator(self.ad_selector).all()
            if ad_elements:
                ad_element_to_click = random.choice(ad_elements)

                if ad_element_to_click.is_visible():
                    update_signal.emit(f"Browser {browser_id}: Clicking an ad.", browser_id)
                    ad_element_to_click.click(timeout=15000)
                    random_delay(2, 5)
                    self.page.go_back()
                    random_delay()
                    update_signal.emit(f"Browser {browser_id}: Clicked ad and navigated back.", browser_id)
                    return True
                else:
                    update_signal.emit(f"Browser {browser_id}: Ad found but not visible, skipping.", browser_id)
                    return False
            else:
                update_signal.emit(f"Browser {browser_id}: No ad elements found.", browser_id)
                return False

        except PlaywrightError as e:
            update_signal.emit(f"Browser {browser_id}: Error during ad click: {e}", browser_id)
            return False
        except Exception as e:
            update_signal.emit(f"Browser {browser_id}: Unexpected error during ad click: {e}", browser_id)
            return False


class FormFiller:
    """Fills out forms with realistic typos and corrections."""

    def __init__(self, page: Page):
        self.page = page
        self.typo_probability = 0.1
        self.correction_probability = 0.8

    def fill_form(self):
        """Locates and fills form fields."""
        form_fields = self.page.locator(
            "input[type='text'], input[type='email'], input[type='password'], textarea").all()
        if not form_fields:
            logging.info("No form fields found.")
            return

        for field in form_fields:
            try:
                field_type = field.get_attribute("type") or "text"
                if field.is_visible() and field.is_enabled():
                    value = self._generate_input_value(field_type)
                    self._type_with_typos(field, value)
                    random_delay()

            except PlaywrightError as e:
                error_logger.warning(f"Could not fill form field: {e}")
            except Exception as e:
                error_logger.error(f"Unexpected error filling form: {e}")

    def _generate_input_value(self, field_type: str) -> str:
        """Generates input values based on field type."""
        if field_type == "email":
            return f"test{random.randint(1, 1000)}@example.com"
        elif field_type == "password":
            return "P@$$wOrd" + str(random.randint(1000, 9999))
        else:
            lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            words = lorem_ipsum.split()
            return " ".join(random.sample(words, random.randint(3, 7)))

    def _type_with_typos(self, field, value: str):
        """Types with simulated typos and corrections."""
        for i, char in enumerate(value):
            field.press(char)
            random_delay(0.05, 0.2)

            if random.random() < self.typo_probability:
                typo = random.choice("abcdefghijklmnopqrstuvwxyz")
                field.press(typo)
                random_delay(0.1, 0.3)

                if random.random() < self.correction_probability:
                    field.press("Backspace")
                    random_delay(0.2, 0.5)
                    field.press(char)
                    random_delay(0.05, 0.2)


class LicenseManager:
    """Verifies the license key."""

    def __init__(self):
        self.expiration_date: Optional[datetime.date] = None

    def verify_license(self, license_key: str) -> Tuple[bool, Optional[str]]:
        """Verifies the license key and sets the expiration date."""
        if license_key == "HamzaAkmal":
            self.expiration_date = datetime.date.today() + datetime.timedelta(days=5)
            logging.info("License key is valid. Expiry Date: %s", self.expiration_date)
            return True, None  # Valid, no error message
        else:
            error_logger.error("Invalid license key.")
            return False, "Invalid license key."  # Invalid, with error message

    def is_license_expired(self) -> Tuple[bool, Optional[str]]:
        """Checks if the license has expired."""
        if self.expiration_date is None:
            return True, "License not activated."  # Expired, no activation

        if datetime.date.today() > self.expiration_date:
            return False, "License has expired."  # Expired
        else:
            return False, None  # Not expired, no error message


def random_delay(min_seconds: float = settings["MIN_DELAY"], max_seconds: float = settings["MAX_DELAY"]):
    """Pauses execution for a random time."""
    time.sleep(random.uniform(min_seconds, max_seconds))


def generate_random_ip() -> str:
    """Generates a random IPv4 address."""
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

# --- GUI Components and Logic ---
try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                 QLineEdit, QPushButton, QTextEdit, QCheckBox, QSpinBox,
                                 QDoubleSpinBox, QFileDialog, QMessageBox, QTabWidget, QGroupBox, QFormLayout,
                                 QTableWidget, QTableWidgetItem, QComboBox, QListWidget, QListWidgetItem,
                                 QStyledItemDelegate, QStyle, QScrollArea)

    from PyQt6.QtCore import QThread, pyqtSignal, QSettings, QDate, QTime, Qt, QSize
    from PyQt6.QtGui import QIcon, QPixmap, QColor
except ImportError:
    print("Missing PyQt6. Please run 'pip install PyQt6'")
    sys.exit(1)

class CenteredLabel(QLabel):
    """Custom QLabel that centers its text."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

class BoldLabel(QLabel):
    """Custom QLabel that makes the text bold."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        font = self.font()
        font.setBold(True)
        self.setFont(font)


class ProxyDelegate(QStyledItemDelegate):
    """Delegate for the proxy list to show proxy status."""
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        status = index.data(Qt.ItemDataRole.UserRole)

        if status == "valid":
            color = QColor("green")
        elif status == "invalid":
            color = QColor("red")
        else:
            color = QColor("gray")

        painter.setBrush(color)
        rect = option.rect
        diameter = min(rect.height(), 16)
        x = rect.left() + 5
        y = rect.top() + (rect.height() - diameter) / 2
        painter.drawEllipse(int(x), int(y), diameter, diameter)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width() + 30, size.height())

class BotThread(QThread):
    """Thread for running the bot in the background."""
    update_signal = pyqtSignal(str, int)  # Signal for updating the GUI (message, browser_id)
    finished_signal = pyqtSignal(int)  # Signal when the bot finishes (browser_id)
    error_signal = pyqtSignal(str, int) #Signal for error (error_message, browser_id)


    def __init__(self, browser_id: int, urls: List[str], proxy: Optional[str], proxy_type: Optional[str], headless: bool, use_chromium_blue: bool,
                 chromium_blue_path: str, chromium_blue_args: str, gemini_api_key: str, user_agent_manager: FingerprintManager):
        super().__init__()
        self.browser_id = browser_id
        self.urls = urls
        self.proxy = proxy
        self.proxy_type = proxy_type  # Can now be None
        self.headless = headless
        self.use_chromium_blue = use_chromium_blue
        self.chromium_blue_path = chromium_blue_path
        self.chromium_blue_args = chromium_blue_args
        self.gemini_api_key = gemini_api_key
        self.user_agent_manager = user_agent_manager
        self.should_stop = False  # Flag to signal the thread to stop

    def run(self):
        """Runs the bot logic."""
        try:
            with sync_playwright() as playwright:
                # Create bot instance in run() method.
                self.bot = HumanTrafficBot(playwright, self.urls, self.proxy, self.proxy_type, self.headless,
                                      self.use_chromium_blue, self.chromium_blue_path, self.chromium_blue_args,
                                      self.gemini_api_key, self.user_agent_manager)
                # PASS self (the BotThread instance) to bot.run()
                self.bot.run(self, self.update_signal, self.browser_id)  # Pass 'self' here

        except Exception as e:
            self.error_signal.emit(str(e), self.browser_id)

        finally:
            self.finished_signal.emit(self.browser_id)

    def stop(self):
        """Sets the flag to stop the thread."""
        self.should_stop = True
        # In Playwright, there's no direct way to interrupt a running operation.
        # Best approach is to close the browser/context.
        if hasattr(self, 'bot') and self.bot.browser_manager:
            self.bot.browser_manager.close_browser()

class HumanTrafficBot:
    """Main bot class."""

    def __init__(self, playwright: Playwright, urls: List[str], proxy: Optional[str] = None,
                 proxy_type: Optional[str] = None,
                 headless: bool = settings["HEADLESS"],
                 use_chromium_blue: bool = settings["CHROMIUM_BLUE_ENABLED"],
                 chromium_blue_path: str = settings["CHROMIUM_BLUE_PATH"],
                 chromium_blue_args: str = settings["CHROMIUM_BLUE_ARGS"],
                 gemini_api_key: str = settings["GEMINI_API_KEY"],
                 user_agent_manager: FingerprintManager = None):

        self.browser_manager = BrowserManager(playwright, proxy_server=proxy, proxy_type=proxy_type, headless=headless,
                                               use_chromium_blue=use_chromium_blue,
                                               chromium_blue_path=chromium_blue_path,
                                               chromium_blue_args=chromium_blue_args)
        self.fingerprint_manager = user_agent_manager if user_agent_manager else FingerprintManager(user_agents, generated_user_agents, gemini_api_key)
        self.scrolling_manager: Optional[ScrollingManager] = None
        self.text_selection_manager: Optional[TextSelectionManager] = None
        self.form_filler: Optional[FormFiller] = None
        self.next_page_navigator: Optional[NextPageNavigator] = None
        self.ad_click_manager: Optional[AdClickManager] = None
        self.urls = urls
        self.license_manager = LicenseManager()
        self.gemini_api_key = gemini_api_key
        self.form_fill_enabled = settings["FORM_FILL_ENABLED"]
        self.impression_enabled = settings["IMPRESSION_ENABLED"]
        self.ad_click_enabled = settings["AD_CLICK_ENABLED"]

    def run(self, bot_thread, update_signal: pyqtSignal, browser_id: int):  # Add bot_thread parameter
        """Runs the bot, now with access to the BotThread."""
        try:
            for i, url in enumerate(self.urls):
                # Check for the stop signal from the BotThread instance:
                if bot_thread.should_stop:  # Access it directly via bot_thread
                    update_signal.emit(f"Browser {browser_id}: Stopping...", browser_id)
                    break

                update_signal.emit(f"Browser {browser_id}: Processing URL {i + 1}/{len(self.urls)}: {url}",
                                    browser_id)
                user_agent = self.fingerprint_manager.get_user_agent(browser_id, update_signal)
                viewport_size = self.fingerprint_manager.get_random_viewport_size()
                self.browser_manager.start_browser(user_agent=user_agent, viewport_size=viewport_size)
                self.scrolling_manager = ScrollingManager(self.browser_manager.page)
                self.text_selection_manager = TextSelectionManager(self.browser_manager.page)
                self.next_page_navigator = NextPageNavigator(self.browser_manager.page)
                self.ad_click_manager = AdClickManager(self.browser_manager.page)

                self.browser_manager.navigate_to(url)
                page_content = self.browser_manager.page.content()

                if self.gemini_api_key:
                    try:
                        genai.configure(api_key=self.gemini_api_key)
                        self.scrolling_manager.gemini_scroll(browser_id, update_signal)
                    except Exception as e:
                        update_signal.emit(f"Browser {browser_id}: Gemini API error: {e}. Random scroll.", browser_id)
                        self.scrolling_manager.random_scroll()
                else:
                    self.scrolling_manager.random_scroll()

                self.text_selection_manager.select_important_text(browser_id, update_signal)
                if self.form_fill_enabled:
                    self.form_filler = FormFiller(self.browser_manager.page)
                    self.form_filler.fill_form()
                if self.ad_click_enabled:
                    self.ad_click_manager.click_ad(browser_id, update_signal)

                if random.random() < 0.2:
                    self.browser_manager.take_screenshot(f"browser_{browser_id}_screenshot.png")

                if self.impression_enabled:
                    if not self.next_page_navigator.navigate_next_page(browser_id, update_signal):
                        update_signal.emit(f"Browser {browser_id}: Reached end of pages or next page not found. Next URL.", browser_id)
                        break
                    else:
                        update_signal.emit(f"Browser {browser_id}: Staying on the same URL base.", browser_id)

                random_delay()

        except Exception as e:
            update_signal.emit(f"Browser {browser_id}: Error: {e}", browser_id)
            self.browser_manager.take_screenshot(f"browser_{browser_id}_error_screenshot.png")
        finally:
            self.browser_manager.close_browser()



class MainWindow(QWidget):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aladdin Traffic Bot")
        self.setWindowIcon(QIcon("resources/chromium_blue.png"))
        self.setGeometry(100, 100, 1100, 850)
        self.settings = QSettings("MyCompany", "HumanTrafficBot")
        self.license_manager = LicenseManager()
        self.bot_threads: List[BotThread] = []
        self.browser_logs: Dict[int, QTextEdit] = {}
        self.proxy_validator = ProxyValidator()
        self.proxies: List[Tuple[str, str]] = []  # Store loaded proxies (address, type)
        self.proxy_statuses: Dict[str, str] = {}
        self.setup_ui()
        self.load_state()
        self.load_proxies_from_file()

    def setup_ui(self):
        """Creates the GUI layout and widgets."""
        layout = QVBoxLayout()

        # --- Tabs ---
        self.tabs = QTabWidget()

        # --- Main Tab ---
        self.main_tab = QWidget()
        self.setup_main_tab()
        self.tabs.addTab(self.main_tab, "Main")

        # --- Settings Tab ---
        self.settings_tab = QWidget()
        self.setup_settings_tab()
        self.tabs.addTab(self.settings_tab, "Settings")

        # --- Visualization Tab ---
        self.visualization_tab = QWidget()
        self.setup_visualization_tab()
        self.tabs.addTab(self.visualization_tab, "Visualization")

        # --- License Tab ---
        self.license_tab = QWidget()
        self.setup_license_tab()
        self.tabs.addTab(self.license_tab, "License")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def setup_main_tab(self):
        """Sets up the main tab's UI."""
        main_layout = QVBoxLayout()

        # --- URL Input ---
        url_group = QGroupBox("URLs")
        url_layout = QVBoxLayout()
        self.url_text_edit = QTextEdit()
        self.url_text_edit.setPlaceholderText("Enter URLs, one per line")
        url_layout.addWidget(self.url_text_edit)
        url_group.setLayout(url_layout)
        main_layout.addWidget(url_group)

         # --- Proxy List ---
        proxy_group = QGroupBox("Proxies")
        proxy_layout = QVBoxLayout()

        self.proxy_list_widget = QListWidget()
        self.proxy_list_widget.setItemDelegate(ProxyDelegate())
        proxy_layout.addWidget(self.proxy_list_widget)

        self.proxy_type_combo_main = QComboBox()
        self.proxy_type_combo_main.addItems(["http", "https", "socks4", "socks5"])
        self.proxy_type_combo_main.setEnabled(True)
        proxy_layout.addWidget(self.proxy_type_combo_main)


        proxy_buttons_layout = QHBoxLayout()
        self.load_proxies_button = QPushButton("Load Proxies")
        self.load_proxies_button.clicked.connect(self.load_proxies_from_file)
        proxy_buttons_layout.addWidget(self.load_proxies_button)

        self.check_proxies_button = QPushButton("Check Proxies")
        self.check_proxies_button.clicked.connect(self.check_all_proxies)
        proxy_buttons_layout.addWidget(self.check_proxies_button)

        proxy_layout.addLayout(proxy_buttons_layout)
        proxy_group.setLayout(proxy_layout)
        main_layout.addWidget(proxy_group)

        # --- Run Configuration ---
        run_config_group = QGroupBox("Run Configuration")
        run_config_layout = QFormLayout()

        self.total_runs_spinbox = QSpinBox()
        self.total_runs_spinbox.setRange(1, 1000)
        self.total_runs_spinbox.setValue(settings["TOTAL_RUNS"])
        run_config_layout.addRow(BoldLabel("Total Runs:"), self.total_runs_spinbox)

        self.run_group_size_spinbox = QSpinBox()
        self.run_group_size_spinbox.setRange(1, 10)
        self.run_group_size_spinbox.setValue(settings["RUN_GROUP_SIZE"])
        run_config_layout.addRow(BoldLabel("Run Group Size:"), self.run_group_size_spinbox)

        run_config_group.setLayout(run_config_layout)
        main_layout.addWidget(run_config_group)


        # --- Controls ---
        controls_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_bots_grouped)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop All")
        self.stop_button.clicked.connect(self.stop_all_bots)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        main_layout.addLayout(controls_layout)

        # --- Browser Logs ---
        self.browser_logs_tab_widget = QTabWidget()
        main_layout.addWidget(self.browser_logs_tab_widget)

        self.main_tab.setLayout(main_layout)

    def setup_settings_tab(self):
        """ Sets up the settings tab's UI."""
        settings_layout = QFormLayout()

        self.proxy_enabled_check = QCheckBox("Enable Proxy (Use Proxies from File)")
        self.proxy_enabled_check.stateChanged.connect(self.toggle_proxy_fields)
        settings_layout.addRow(self.proxy_enabled_check)

        self.headless_check = QCheckBox("Headless Mode")
        settings_layout.addRow(self.headless_check)

        self.min_delay_spin = QDoubleSpinBox()
        self.min_delay_spin.setRange(0.1, 60.0)
        self.min_delay_spin.setSingleStep(0.1)
        settings_layout.addRow(BoldLabel("Min Delay (s):"), self.min_delay_spin)

        self.max_delay_spin = QDoubleSpinBox()
        self.max_delay_spin.setRange(0.1, 60.0)
        self.max_delay_spin.setSingleStep(0.1)
        settings_layout.addRow(BoldLabel("Max Delay (s):"), self.max_delay_spin)

        self.gemini_api_key_input = QLineEdit()
        self.gemini_api_key_input.setPlaceholderText("Gemini API Key")
        settings_layout.addRow(BoldLabel("Gemini API Key:"), self.gemini_api_key_input)

        self.chromium_blue_check = QCheckBox("Use Chromium Blue")
        self.chromium_blue_check.stateChanged.connect(self.toggle_chromium_blue_fields)
        settings_layout.addRow(self.chromium_blue_check)

        self.chromium_blue_path_input = QLineEdit()
        self.chromium_blue_path_input.setPlaceholderText("Path to Chromium Blue executable")
        self.chromium_blue_path_input.setEnabled(False)
        settings_layout.addRow(BoldLabel("Chromium Blue Path:"), self.chromium_blue_path_input)

        self.chromium_blue_args_input = QLineEdit()
        self.chromium_blue_args_input.setPlaceholderText("Chromium Blue Arguments (Optional)")
        self.chromium_blue_args_input.setEnabled(False)
        settings_layout.addRow(BoldLabel("Chromium Blue Args:"), self.chromium_blue_args_input)

        self.mouse_movement_check = QCheckBox("Enable Mouse Movement")
        settings_layout.addRow(self.mouse_movement_check)

        self.concurrent_browsers_spin = QSpinBox()
        self.concurrent_browsers_spin.setRange(1, 5)
        settings_layout.addRow(BoldLabel("Concurrent Browsers:"), self.concurrent_browsers_spin)

        self.scroll_duration_min_spin = QSpinBox()
        self.scroll_duration_min_spin.setRange(200, 2000)
        self.scroll_duration_min_spin.setSingleStep(50)
        self.scroll_duration_min_spin.setValue(settings["SCROLL_DURATION_MIN"])
        settings_layout.addRow(BoldLabel("Min Scroll Duration (ms):"), self.scroll_duration_min_spin)

        self.scroll_duration_max_spin = QSpinBox()
        self.scroll_duration_max_spin.setRange(500, 5000)
        self.scroll_duration_max_spin.setSingleStep(50)
        self.scroll_duration_max_spin.setValue(settings["SCROLL_DURATION_MAX"])
        settings_layout.addRow(BoldLabel("Max Scroll Duration (ms):"), self.scroll_duration_max_spin)

        self.form_fill_check = QCheckBox("Enable Form Filling")
        settings_layout.addRow(self.form_fill_check)

        self.impression_enabled_check = QCheckBox("Enable Impression (Auto Next Page)")
        settings_layout.addRow(self.impression_enabled_check)

        self.next_page_selector_input = QLineEdit()
        self.next_page_selector_input.setPlaceholderText("CSS Selector for Next Page")
        settings_layout.addRow(BoldLabel("Next Page Selector:"), self.next_page_selector_input)

        self.ad_click_enabled_check = QCheckBox("Enable Ad Click")
        settings_layout.addRow(self.ad_click_enabled_check)

        self.ad_selector_input = QLineEdit()
        self.ad_selector_input.setPlaceholderText("CSS Selector for Ads")
        settings_layout.addRow(BoldLabel("Ad Selector:"), self.ad_selector_input)

        self.ad_click_probability_spin = QDoubleSpinBox()
        self.ad_click_probability_spin.setRange(0.0, 1.0)
        self.ad_click_probability_spin.setSingleStep(0.05)
        settings_layout.addRow(BoldLabel("Ad Click Probability:"), self.ad_click_probability_spin)

        self.user_agent_generation_check = QCheckBox("Enable User Agent Generation")
        settings_layout.addRow(self.user_agent_generation_check)

        self.user_agent_generation_count_spin = QSpinBox()
        self.user_agent_generation_count_spin.setRange(5, 50)
        self.user_agent_generation_count_spin.setSingleStep(5)
        self.user_agent_generation_count_spin.setValue(settings["USER_AGENT_GENERATION_COUNT"])
        settings_layout.addRow(BoldLabel("User Agent Generation Count:"), self.user_agent_generation_count_spin)

        settings_buttons_layout = QHBoxLayout()

        self.load_settings_button = QPushButton("Load Settings")
        self.load_settings_button.clicked.connect(self.load_settings_from_file)
        settings_buttons_layout.addWidget(self.load_settings_button)

        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.clicked.connect(self.save_settings_to_file)
        settings_buttons_layout.addWidget(self.save_settings_button)

        settings_layout.addRow(settings_buttons_layout)
        self.settings_tab.setLayout(settings_layout)

    def setup_visualization_tab(self):
        """Sets up the visualization tab UI."""
        visualization_layout = QVBoxLayout()
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(2)
        self.data_table.setHorizontalHeaderLabels(["X-Axis", "Y-Axis"])
        visualization_layout.addWidget(self.data_table)
        self.visualization_tab.setLayout(visualization_layout)

    def setup_license_tab(self):
        """Sets up the license tab's UI."""
        license_layout = QVBoxLayout()

        self.license_key_input = QLineEdit()
        self.license_key_input.setPlaceholderText("Enter License Key")
        license_layout.addWidget(self.license_key_input)

        self.activate_button = QPushButton("Activate")
        self.activate_button.clicked.connect(self.activate_license)
        license_layout.addWidget(self.activate_button)

        self.license_status_label = CenteredLabel("License Status: Not Activated")
        license_layout.addWidget(self.license_status_label)

        self.license_tab.setLayout(license_layout)

    def toggle_proxy_fields(self, state):
        """Enables/disables proxy input based on checkbox."""
        checked = state == Qt.CheckState.Checked.value
        self.load_proxies_button.setEnabled(checked)
        self.check_proxies_button.setEnabled(checked)
        self.proxy_list_widget.setEnabled(checked)


    def toggle_chromium_blue_fields(self, state):
        self.chromium_blue_path_input.setEnabled(state == Qt.CheckState.Checked.value)
        self.chromium_blue_args_input.setEnabled(state == Qt.CheckState.Checked.value)

    def activate_license(self):
        """Activates the license key."""
        license_key = self.license_key_input.text()
        is_valid, message = self.license_manager.verify_license(license_key)
        if is_valid:
            self.license_status_label.setText(
                f"License: Activated (Expires: {self.license_manager.expiration_date.strftime('%Y-%m-%d')})")
            self.save_state()
        else:
            self.license_status_label.setText(f"License Status: {message}")


    def load_proxies_from_file(self):
        """Loads proxies and updates the list widget."""
        self.proxies = load_proxies()
        self.proxy_list_widget.clear()
        self.proxy_statuses.clear()

        for proxy_address, proxy_type in self.proxies:
            item = QListWidgetItem(f"{proxy_address} ({proxy_type})")
            item.setData(Qt.ItemDataRole.UserRole, "")
            self.proxy_list_widget.addItem(item)
            self.proxy_statuses[proxy_address] = ""

    def check_all_proxies(self):
        """Checks validity of all loaded proxies and updates the UI."""
        selected_proxy_type = self.proxy_type_combo_main.currentText()

        for row in range(self.proxy_list_widget.count()):
            item = self.proxy_list_widget.item(row)
            proxy_address, _ = self.proxies[row]  # Get address, ignore original type

            is_valid = self.proxy_validator.check_proxy(proxy_address, selected_proxy_type)
            status = "valid" if is_valid else "invalid"
            self.proxy_statuses[proxy_address] = status

            item.setData(Qt.ItemDataRole.UserRole, status)
            self.proxies[row] = (proxy_address, selected_proxy_type)
            item.setText(f"{proxy_address} ({selected_proxy_type})")

        self.proxy_list_widget.viewport().update()


    def get_next_proxy(self) -> Tuple[Optional[str], Optional[str]]:
        """Gets the next available proxy, cycling through them."""
        if not self.proxies:
            return None, None

        for _ in range(len(self.proxies)):
            proxy_address, proxy_type = self.proxies[self.current_proxy_index]
            if self.proxy_statuses.get(proxy_address) == "valid":
                self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
                return proxy_address, proxy_type
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)

        return None, None

    def start_bots_grouped(self):
        """Starts bot threads in groups, with different proxies."""
        is_expired, message = self.license_manager.is_license_expired()
        if is_expired:
            QMessageBox.warning(self, "License Error", message)
            return

        urls_text = self.url_text_edit.toPlainText()
        urls = [url.strip() for url in urls_text.splitlines() if url.strip()]
        if not urls:
            QMessageBox.warning(self, "No URLs","Please enter at least one URL.")
            return

        total_runs = self.total_runs_spinbox.value()
        run_group_size = self.run_group_size_spinbox.value()

        self.current_proxy_index = 0  # Initialize proxy cycling
        if self.proxy_enabled_check.isChecked():
            if not self.proxies:
                QMessageBox.warning(self, "No Proxies", "Please load proxies from a file.")
                return
            if not all(status in ("valid", "invalid") for status in self.proxy_statuses.values()):
                reply = QMessageBox.question(self, "Check Proxies",
                                            "Proxies unchecked. Check now?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    self.check_all_proxies()
            if not any(status == "valid" for status in self.proxy_statuses.values()):
                QMessageBox.critical(self, "No Valid Proxies", "No valid proxies found.  Load and check valid proxies.")
                return
            proxy_type = self.proxy_type_combo_main.currentText()
        else:
            proxy_address, proxy_type = None, None

        headless = self.headless_check.isChecked()
        use_chromium_blue = self.chromium_blue_check.isChecked()
        chromium_blue_path = self.chromium_blue_path_input.text().strip()
        chromium_blue_args = self.chromium_blue_args_input.text().strip()
        gemini_api_key = self.gemini_api_key_input.text().strip()
        impression_enabled = self.impression_enabled_check.isChecked()
        next_page_selector = self.next_page_selector_input.text().strip()
        ad_click_enabled = self.ad_click_enabled_check.isChecked()
        ad_selector = self.ad_selector_input.text().strip()
        ad_click_probability = self.ad_click_probability_spin.value()
        user_agent_generation_enabled = self.user_agent_generation_check.isChecked()
        user_agent_generation_count = self.user_agent_generation_count_spin.value()

        settings["PROXY_ENABLED"] = self.proxy_enabled_check.isChecked()
        settings["HEADLESS"] = self.headless_check.isChecked()
        settings["MIN_DELAY"] = self.min_delay_spin.value()
        settings["MAX_DELAY"] = self.max_delay_spin.value()
        settings["GEMINI_API_KEY"] = self.gemini_api_key_input.text().strip()
        settings["CHROMIUM_BLUE_ENABLED"] = self.chromium_blue_check.isChecked()
        settings["CHROMIUM_BLUE_PATH"] = self.chromium_blue_path_input.text().strip()
        settings["CHROMIUM_BLUE_ARGS"] = self.chromium_blue_args_input.text().strip()
        settings["MOUSE_MOVEMENT_ENABLED"] = self.mouse_movement_check.isChecked()
        settings["CONCURRENT_BROWSERS"] = 1
        settings["SCROLL_DURATION_MIN"] = self.scroll_duration_min_spin.value()
        settings["SCROLL_DURATION_MAX"] = self.scroll_duration_max_spin.value()
        settings["FORM_FILL_ENABLED"] = self.form_fill_check.isChecked()
        settings["IMPRESSION_ENABLED"] = self.impression_enabled_check.isChecked()
        settings["NEXT_PAGE_SELECTOR"] = self.next_page_selector_input.text().strip()
        settings["AD_CLICK_ENABLED"] = self.ad_click_enabled_check.isChecked()
        settings["AD_SELECTOR"] = self.ad_selector_input.text().strip()
        settings["AD_CLICK_PROBABILITY"] = self.ad_click_probability_spin.value()
        settings["TOTAL_RUNS"] = self.total_runs_spinbox.value()
        settings["RUN_GROUP_SIZE"] = self.run_group_size_spinbox.value()
        settings["USER_AGENT_GENERATION_ENABLED"] = self.user_agent_generation_check.isChecked()
        settings["USER_AGENT_GENERATION_COUNT"] = self.user_agent_generation_count_spin.value()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.browser_logs_tab_widget.clear()

        user_agent_manager = FingerprintManager(user_agents, generated_user_agents, gemini_api_key)

        runs_completed = 0
        while runs_completed < total_runs:
            current_group_size = min(run_group_size, total_runs - runs_completed)
            group_threads = []
            for i in range(current_group_size):
                browser_id = runs_completed + i
                log_text_edit = QTextEdit()
                log_text_edit.setReadOnly(True)
                self.browser_logs[browser_id] = log_text_edit
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)
                scroll_area.setWidget(log_text_edit)
                self.browser_logs_tab_widget.addTab(scroll_area, f"Browser {browser_id + 1}")

                if self.proxy_enabled_check.isChecked():
                    proxy_address, proxy_type = self.get_next_proxy()
                    if proxy_address is None:
                        QMessageBox.critical(self, "Error", "No valid proxies.")
                        return
                else:
                    proxy_address, proxy_type = None, None

                bot_thread = BotThread(browser_id, urls, proxy_address, proxy_type, headless, use_chromium_blue,
                                      chromium_blue_path, chromium_blue_args, gemini_api_key, user_agent_manager)
                bot_thread.update_signal.connect(self.update_log)
                bot_thread.finished_signal.connect(self.on_bot_finished)
                bot_thread.error_signal.connect(self.on_bot_error)
                group_threads.append(bot_thread)
                bot_thread.start()

            for thread in group_threads:
                thread.wait()  # Correctly wait for each thread in the group
            runs_completed += current_group_size
            user_agent_manager.used_user_agents_in_run = set()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.information(self, "Bots Finished", f"All {total_runs} runs completed.")

    def stop_all_bots(self):
        """Stops all running bot threads."""
        for bot_thread in self.bot_threads:
            if bot_thread and bot_thread.isRunning():
                bot_thread.stop()  # Signal the thread to stop
                bot_thread.wait()  # **CRITICAL:** Wait for thread to finish
        self.bot_threads = []  # Clear AFTER stopping
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.information(self, "Bots Stopped", "All bots stopped.")

    def update_log(self, message: str, browser_id: int):
        """Appends a message to the correct log text edit."""
        if browser_id in self.browser_logs:
            log_text_edit = self.browser_logs[browser_id]
            log_text_edit.append(message)
            cursor = log_text_edit.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            log_text_edit.setTextCursor(cursor)
        else:
            error_logger.warning(f"No log found for browser ID: {browser_id}")


    def on_bot_finished(self, browser_id: int):
        """Handles a bot finishing."""
        self.update_log(f"Bot {browser_id + 1}: Finished.", browser_id)

    def on_bot_error(self, error_message: str, browser_id: int):
       """Handles errors from a bot thread."""
       QMessageBox.critical(self, f"Bot {browser_id + 1} Error", f"Bot {browser_id + 1}: {error_message}")
       self.update_log(f"Bot {browser_id}: Error: {error_message}", browser_id)


    def save_state(self):
        """Saves the UI state (settings and license)."""
        self.settings.setValue("proxy_enabled", self.proxy_enabled_check.isChecked())
        self.settings.setValue("headless", self.headless_check.isChecked())
        self.settings.setValue("min_delay", self.min_delay_spin.value())
        self.settings.setValue("max_delay", self.max_delay_spin.value())
        self.settings.setValue("gemini_api_key", self.gemini_api_key_input.text())
        self.settings.setValue("chromium_blue_enabled", self.chromium_blue_check.isChecked())
        self.settings.setValue("chromium_blue_path", self.chromium_blue_path_input.text())
        self.settings.setValue("chromium_blue_args", self.chromium_blue_args_input.text())
        self.settings.setValue("mouse_movement_enabled", self.mouse_movement_check.isChecked())
        self.settings.setValue("concurrent_browsers", 1)
        self.settings.setValue("scroll_duration_min", self.scroll_duration_min_spin.value())
        self.settings.setValue("scroll_duration_max", self.scroll_duration_max_spin.value())
        self.settings.setValue("form_fill_enabled", self.form_fill_check.isChecked())
        self.settings.setValue("impression_enabled", self.impression_enabled_check.isChecked())
        self.settings.setValue("next_page_selector", self.next_page_selector_input.text().strip())
        self.settings.setValue("ad_click_enabled", self.ad_click_enabled_check.isChecked())
        self.settings.setValue("ad_selector", self.ad_selector_input.text().strip())
        self.settings.setValue("ad_click_probability", self.ad_click_probability_spin.value())
        self.settings.setValue("total_runs", self.total_runs_spinbox.value())
        self.settings.setValue("run_group_size", self.run_group_size_spinbox.value())
        self.settings.setValue("user_agent_generation_enabled", self.user_agent_generation_check.isChecked())
        self.settings.setValue("user_agent_generation_count", self.user_agent_generation_count_spin.value())

        if self.license_manager.expiration_date:
            self.settings.setValue("license_expiry",
                                   self.license_manager.expiration_date.toordinal())

    def load_state(self):
        """Loads the UI state."""
        self.proxy_enabled_check.setChecked(self.settings.value("proxy_enabled", False, type=bool))

        self.load_proxies_button.setEnabled(self.proxy_enabled_check.isChecked())
        self.check_proxies_button.setEnabled(self.proxy_enabled_check.isChecked())
        self.proxy_list_widget.setEnabled(self.proxy_enabled_check.isChecked())

        self.headless_check.setChecked(self.settings.value("headless", True, type=bool))
        self.min_delay_spin.setValue(self.settings.value("min_delay", 2.0, type=float))
        self.max_delay_spin.setValue(self.settings.value("max_delay", 5.0, type=float))
        self.gemini_api_key_input.setText(self.settings.value("gemini_api_key", ""))
        self.license_key_input.setText(self.settings.value("license_key", ""))
        self.chromium_blue_check.setChecked(self.settings.value("chromium_blue_enabled", False, type=bool))
        self.chromium_blue_path_input.setText(self.settings.value("chromium_blue_path", ""))
        self.chromium_blue_path_input.setEnabled(self.chromium_blue_check.isChecked())
        self.chromium_blue_args_input.setText(self.settings.value("chromium_blue_args", ""))
        self.chromium_blue_args_input.setEnabled(self.chromium_blue_check.isChecked())
        self.mouse_movement_check.setChecked(self.settings.value("mouse_movement_enabled", True, type=bool))
        self.concurrent_browsers_spin.setValue(1)
        self.scroll_duration_min_spin.setValue(self.settings.value("scroll_duration_min", settings["SCROLL_DURATION_MIN"], type=int))
        self.scroll_duration_max_spin.setValue(self.settings.value("scroll_duration_max", settings["SCROLL_DURATION_MAX"], type=int))
        self.form_fill_check.setChecked(self.settings.value("form_fill_enabled", settings["FORM_FILL_ENABLED"], type=bool))
        self.impression_enabled_check.setChecked(self.settings.value("impression_enabled", settings["IMPRESSION_ENABLED"], type=bool))
        self.next_page_selector_input.setText(self.settings.value("next_page_selector", settings["NEXT_PAGE_SELECTOR"]))
        self.ad_click_enabled_check.setChecked(self.settings.value("ad_click_enabled", settings["AD_CLICK_ENABLED"], type=bool))
        self.ad_selector_input.setText(self.settings.value("ad_selector", settings["AD_SELECTOR"]))
        self.ad_click_probability_spin.setValue(self.settings.value("ad_click_probability", settings["AD_CLICK_PROBABILITY"], type=float))
        self.total_runs_spinbox.setValue(self.settings.value("total_runs", settings["TOTAL_RUNS"], type=int))
        self.run_group_size_spinbox.setValue(self.settings.value("run_group_size", settings["RUN_GROUP_SIZE"], type=int))
        self.user_agent_generation_check.setChecked(self.settings.value("user_agent_generation_enabled", settings["USER_AGENT_GENERATION_ENABLED"], type=bool))
        self.user_agent_generation_count_spin.setValue(self.settings.value("user_agent_generation_count", settings["USER_AGENT_GENERATION_COUNT"], type=int))

        expiry_ordinal = self.settings.value("license_expiry")
        if expiry_ordinal is not None:
            try:
                self.license_manager.expiration_date = datetime.date.fromordinal(expiry_ordinal)
                self.license_status_label.setText(
                    f"License: Activated (Expires: {self.license_manager.expiration_date.strftime('%Y-%m-%d')})")
            except (TypeError, ValueError):
                self.license_status_label.setText("License Status: Not Activated")
        else:
            self.license_status_label.setText("License Status: Not Activated")

    def closeEvent(self, event):
        """Handles the window closing event."""
        self.stop_all_bots()  # Stop all threads before closing
        self.save_state()
        event.accept()

    def save_settings_to_file(self):
        """Saves current settings to config/settings.py"""
        try:
            filepath = "config/settings.py"
            with open(filepath, 'w') as f:
                f.write(f"PROXY_ENABLED = {self.proxy_enabled_check.isChecked()}\n")
                f.write(f"PROXY_FILE = 'config/proxies.txt'\n")
                f.write(f"HEADLESS = {self.headless_check.isChecked()}\n")
                f.write(f"MIN_DELAY = {self.min_delay_spin.value()}\n")
                f.write(f"MAX_DELAY = {self.max_delay_spin.value()}\n")
                f.write(f"GEMINI_API_KEY = '{self.gemini_api_key_input.text().strip()}'\n")
                f.write(f"LICENSE_KEY = '{self.license_key_input.text().strip()}'\n")
                f.write(f"USER_AGENTS_FILE = 'data/user_agents.json'\n")
                f.write(f"IMPORTANT_WORDS_FILE = 'data/important_words.json'\n")
                f.write(f"GENERATED_USER_AGENTS_FILE = 'data/generated_user_agents.json'\n")
                f.write(f"VIEWPORT_MIN_WIDTH = {settings['VIEWPORT_MIN_WIDTH']}\n")
                f.write(f"VIEWPORT_MAX_WIDTH = {settings['VIEWPORT_MAX_WIDTH']}\n")
                f.write(f"VIEWPORT_MIN_HEIGHT = {settings['VIEWPORT_MIN_HEIGHT']}\n")
                f.write(f"VIEWPORT_MAX_HEIGHT = {settings['VIEWPORT_MAX_HEIGHT']}\n")
                f.write(f"CHROMIUM_BLUE_ENABLED = {self.chromium_blue_check.isChecked()}\n")
                f.write(f"CHROMIUM_BLUE_PATH = '{self.chromium_blue_path_input.text().strip()}'\n")
                f.write(f"CHROMIUM_BLUE_ARGS = '{self.chromium_blue_args_input.text().strip()}'\n")
                f.write(f"MOUSE_MOVEMENT_ENABLED = {self.mouse_movement_check.isChecked()}\n")
                f.write(f"CONCURRENT_BROWSERS = 1\n")
                f.write(f"SCROLL_DURATION_MIN = {self.scroll_duration_min_spin.value()}\n")
                f.write(f"SCROLL_DURATION_MAX = {self.scroll_duration_max_spin.value()}\n")
                f.write(f"FORM_FILL_ENABLED = {self.form_fill_check.isChecked()}\n")
                f.write(f"IMPRESSION_ENABLED = {self.impression_enabled_check.isChecked()}\n")
                f.write(f"NEXT_PAGE_SELECTOR = '{self.next_page_selector_input.text().strip()}'\n")
                f.write(f"AD_CLICK_ENABLED = {self.ad_click_enabled_check.isChecked()}\n")
                f.write(f"AD_SELECTOR = '{self.ad_selector_input.text().strip()}'\n")
                f.write(f"AD_CLICK_PROBABILITY = {self.ad_click_probability_spin.value()}\n")
                f.write(f"TOTAL_RUNS = {self.total_runs_spinbox.value()}\n")
                f.write(f"RUN_GROUP_SIZE = {self.run_group_size_spinbox.value()}\n")
                f.write(f"USER_AGENT_GENERATION_ENABLED = {self.user_agent_generation_check.isChecked()}\n")
                f.write(f"USER_AGENT_GENERATION_COUNT = {self.user_agent_generation_count_spin.value()}\n")


            QMessageBox.information(self, "Settings Saved", "Settings saved to config/settings.py")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")

    def load_settings_from_file(self):
        """Loads settings from config/settings.py."""
        global settings, generated_user_agents
        settings = load_settings()
        generated_user_agents = load_generated_user_agents()

        self.proxy_enabled_check.setChecked(settings["PROXY_ENABLED"])
        self.load_proxies_button.setEnabled(self.proxy_enabled_check.isChecked())
        self.check_proxies_button.setEnabled(self.proxy_enabled_check.isChecked())
        self.proxy_list_widget.setEnabled(self.proxy_enabled_check.isChecked())

        self.headless_check.setChecked(settings["HEADLESS"])
        self.min_delay_spin.setValue(settings["MIN_DELAY"])
        self.max_delay_spin.setValue(settings["MAX_DELAY"])
        self.gemini_api_key_input.setText(settings["GEMINI_API_KEY"])
        self.license_key_input.setText(settings["LICENSE_KEY"])
        self.chromium_blue_check.setChecked(settings["CHROMIUM_BLUE_ENABLED"])
        self.chromium_blue_path_input.setText(settings["CHROMIUM_BLUE_PATH"])
        self.chromium_blue_args_input.setText(settings["CHROMIUM_BLUE_ARGS"])
        self.mouse_movement_check.setChecked(settings["MOUSE_MOVEMENT_ENABLED"])
        self.concurrent_browsers_spin.setValue(1)
        self.scroll_duration_min_spin.setValue(settings["SCROLL_DURATION_MIN"])
        self.scroll_duration_max_spin.setValue(settings["SCROLL_DURATION_MAX"])
        self.form_fill_check.setChecked(settings["FORM_FILL_ENABLED"])
        self.impression_enabled_check.setChecked(settings["IMPRESSION_ENABLED"])
        self.next_page_selector_input.setText(settings["NEXT_PAGE_SELECTOR"])
        self.ad_click_enabled_check.setChecked(settings["AD_CLICK_ENABLED"])
        self.ad_selector_input.setText(settings["AD_SELECTOR"])
        self.ad_click_probability_spin.setValue(settings["AD_CLICK_PROBABILITY"])
        self.total_runs_spinbox.setValue(settings["TOTAL_RUNS"])
        self.run_group_size_spinbox.setValue(settings["RUN_GROUP_SIZE"])
        self.user_agent_generation_check.setChecked(settings["USER_AGENT_GENERATION_ENABLED"])
        self.user_agent_generation_count_spin.setValue(settings["USER_AGENT_GENERATION_COUNT"])

        self.toggle_proxy_fields(Qt.CheckState.Checked if settings["PROXY_ENABLED"] else Qt.CheckState.Unchecked)
        self.toggle_chromium_blue_fields(
            Qt.CheckState.Checked if settings["CHROMIUM_BLUE_ENABLED"] else Qt.CheckState.Unchecked)

        QMessageBox.information(self, "Settings Loaded", "Settings loaded from config/settings.py")

class ProxyValidator:
    """Validates proxy formats and checks proxy connectivity."""

    def validate_proxy_format(self, proxy: str) -> bool:
        """Validates if the proxy string is in the format IP:PORT."""
        pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$"
        return bool(re.match(pattern, proxy))

    def check_proxy(self, proxy: str, proxy_type: str) -> bool:
        """Checks if the proxy is reachable."""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    proxy={"server": f"{proxy_type}://{proxy}"},
                    headless=True
                )
                context = browser.new_context()
                page = context.new_page()
                page.goto("https://www.example.com", timeout=10000)
                browser.close()
                return True
        except PlaywrightError as e:
            logging.info(f"Proxy check failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during proxy check: {e}")
            return False

def check_requirements():
    """Checks if required libraries are installed."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
    except Exception as e:
        error_logger.error(f"Error while checking requirements: {e}")

def main():
    """Entry point for the GUI application."""
    check_requirements()
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget { font-size: 14px; }
        QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #555; border-radius: 8px; margin-top: 1ex; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; background-color: #f0f0f0; border-radius: 5px; }
        QTabBar::tab { background: #f0f0f0; color: #333; border: 1px solid #555; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
        QTabBar::tab:selected { background: #ddd; border-bottom-color: #ddd; }
        QTabWidget::pane { border-top: 2px solid #C2C7CB; }
        QTabWidget::tab-bar:top { top: 1px; }
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { border: 1px solid #aaa; border-radius: 4px; padding: 5px; background-color: #fff; selection-background-color: #a6c9e2; }
        QPushButton { background-color: #555; color: white; border: none; padding: 8px 16px; border-radius: 4px; }
        QPushButton:hover { background-color: #777; }
        QPushButton:pressed { background-color: #333; }
        QPushButton:disabled { background-color: #999; color: #ccc; }
        QCheckBox { spacing: 5px; }
        QCheckBox::indicator { width: 18px; height: 18px; }
        QListWidget { border: 1px solid #aaa; border-radius: 4px; }
        /* Style for the scroll area */
        QScrollArea {
            border: none; /* Remove border from the scroll area itself */
        }

    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()