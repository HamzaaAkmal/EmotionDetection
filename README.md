## 1. Introduction

Aladdin Traffic Bot is a Python-based application using PyQt6 for the graphical user interface (GUI) and Playwright for browser automation. It is designed to simulate human-like browsing behavior on target websites. The primary stated goal is to generate website traffic, scrolling, and potential impressions/clicks, ostensibly to influence metrics like AdSense RPM.

It incorporates various techniques to mimic human interaction and evade basic bot detection, including:

*   Proxy rotation
*   Browser fingerprint spoofing (User Agent, WebGL, Canvas, etc.)
*   AI-driven user agent and fingerprint generation (via Google Gemini)
*   Human-like scrolling patterns (including AI-generated)
*   Simulated mouse movements
*   Interaction with page elements (pagination, forms, potential ad clicks)
*   Evasion of common automation detection flags

### 1.1. Disclaimer (IMPORTANT!)

**Using this bot, or any similar tool, to generate artificial traffic, impressions, or clicks on advertisements (like Google AdSense) is a direct and serious violation of the respective ad network's Terms of Service and policies against Invalid Traffic.**

*   **High Risk of Account Suspension:** Attempting to inflate earnings or metrics using automated traffic is strictly prohibited and easily detectable by sophisticated anti-fraud systems. This will almost certainly lead to permanent suspension of your ad account(s) and forfeiture of any earnings.
*   **No Guarantees:** Despite the bot's features, bypassing advanced detection systems (like Google's) is extremely difficult and unlikely in the long run.
*   **Ethical Concerns:** Generating fake traffic undermines the advertising ecosystem.
*   **Use at Your Own Absolute Risk:** The developers/providers of this code are not responsible for any consequences resulting from its use, including but not limited to ad account suspension, financial loss, or legal issues.

**This documentation describes the technical functionality of the code provided. It does not endorse or recommend its use for violating AdSense or any other platform's policies.**

---

## 2. Core Features

*   **GUI:** User-friendly interface built with PyQt6 for configuration and monitoring.
*   **Multi-Instance Support:** Run multiple bot instances concurrently using threading and grouping.
*   **Proxy Management:**
    *   Load proxies from file (`config/proxies.txt`).
    *   Support for HTTP, HTTPS (treated as HTTP), SOCKS4, SOCKS5.
    *   Built-in proxy checker using Playwright.
    *   Optional proxy usage per bot instance, cycling through available proxies.
*   **Advanced Fingerprinting:**
    *   Spoof User-Agent, viewport, platform, vendor, languages, hardware concurrency, device memory, plugins, mimeTypes.
    *   Spoof screen resolution, color depth.
    *   Canvas fingerprinting noise injection.
    *   WebGL vendor/renderer spoofing.
    *   Timezone and locale spoofing.
    *   Configuration via JSON profiles (`config/fingerprint_profiles.json`).
    *   Option to use specific profiles, a random profile, or generate a new UA per bot.
*   **AI-Powered Generation (Google Gemini):**
    *   Generate realistic User-Agent strings (manually or automatically).
    *   Generate complete browser fingerprint profiles (manually).
    *   Generate human-like scrolling patterns.
*   **Evasion Techniques:**
    *   Hides `navigator.webdriver` flag via JavaScript.
    *   Removes common detection variables (e.g., `cdc_`).
    *   Uses Chromium launch flag (`--force-webrtc-ip-handling-policy`) to prevent WebRTC local IP leaks.
    *   Includes JS hooks as a secondary attempt to filter WebRTC candidates.
    *   Sends realistic `Sec-CH-UA` client hint headers.
*   **Human-Like Behavior Simulation:**
    *   Smooth scrolling animations using `requestAnimationFrame`.
    *   Randomized scrolling patterns.
    *   Optional AI-generated scroll patterns.
    *   Simulated mouse cursor movement towards "important" elements (links, buttons, keywords).
    *   Randomized delays between actions.
    *   "Scanning" vs. "Reading" behavioral states influencing actions.
    *   Probabilistic skipping of actions.
*   **Interaction Features:**
    *   **Impression Generation:** Navigate pagination links (Next Page, etc.).
    *   **Ad Clicking:** Probabilistically find and click elements identified as ads (via selectors/iframes). Handles new tabs/popups.
    *   **Form Filling:** Find and fill common form fields (text, email, select, textarea) with realistic data and simulated typos/corrections.
*   **Custom Browser Support:** Option to use a custom Chromium build (e.g., "Chromium Blue").
*   **Configuration:** Settings managed via `config/settings.py` and controllable through the GUI. Save/Load functionality.
*   **Logging:** Detailed activity and error logs (`logs/bot_activity.log`, `logs/errors.log`) and per-bot logs in the GUI.
*   **State Persistence:** Saves GUI settings and license status between sessions.

---

## 3. Project Structure

The bot automatically creates the following directory structure and essential files if they don't exist:

```
.
├── aladdin_bot_script.py   # Main script file (or similar name)
├── core/                   # Core bot logic classes (intended, currently integrated)
├── config/                 # Configuration files
│   ├── settings.py           # Main configuration (Editable Python dict)
│   ├── proxies.txt           # List of proxies (one per line)
│   └── fingerprint_profiles.json # Browser fingerprint profiles
├── logs/                   # Log files
│   ├── bot_activity.log    # General activity log
│   └── errors.log          # Error log
├── data/                   # Data files used by the bot
│   ├── user_agents.json        # Static list of user agents
│   ├── generated_user_agents.json # UAs generated by Gemini
│   └── important_words.json    # Keywords for mouse hovering
├── resources/              # Resource files (e.g., icons)
│   └── chromium_blue.png     # Bot icon
└── tests/                  # Placeholder for tests (intended)
```

---

## 4. Dependencies & Installation

### 4.1. Python Libraries

The bot requires Python 3.x and the following libraries:

*   **PyQt6:** For the graphical user interface.
*   **playwright:** For browser automation.
*   **google-generativeai:** For interacting with the Google Gemini API (User Agent/Fingerprint/Scroll generation).
*   **Pillow:** (Optional, used to create a dummy icon if missing).

Install them using pip:

```bash
pip install PyQt6 playwright google-generativeai Pillow
```

### 4.2. Playwright Browsers

Playwright needs browser binaries to function. After installing the library, you **must** install the browsers (Chromium is primarily used by this bot):

```bash
playwright install chromium
# Or install all default browsers:
# playwright install
```

The script includes a check on startup. If this fails, run the command above.

### 4.3. OS Dependencies (Linux)

On Linux systems, Playwright browsers might require additional operating system libraries. If you encounter issues launching browsers, run:

```bash
# For Debian/Ubuntu based systems:
sudo playwright install-deps chromium
# Or install all dependencies:
# sudo playwright install-deps
```

Refer to the official [Playwright documentation](https://playwright.dev/python/docs/library#install-system-dependencies) for dependencies on other Linux distributions.

---

## 5. Configuration (`config/settings.py`)

Most bot behavior is controlled via the `config/settings.py` file. This file contains a Python dictionary defining various parameters. You can edit this file directly or modify settings through the GUI's "Settings" tab and use the "Save Settings" button.

Default settings are defined in the script and used if the file is missing or a setting is absent.

### 5.1. Core & Control

*   `TOTAL_RUNS` (int): Total number of bot instances to launch over the entire run.
*   `CONCURRENT_BROWSERS` (int): [Currently less relevant due to grouping] Maximum number of browser instances allowed globally (legacy?).
*   `RUN_GROUP_SIZE` (int): Number of bot instances to launch and run concurrently within a single group. The bot waits for one group to finish before starting the next.
*   `LICENSE_KEY` (str): Stores the last entered license key (basic activation).
*   `MIN_DELAY` (float): Minimum delay (in seconds) between actions within a bot instance.
*   `MAX_DELAY` (float): Maximum delay (in seconds) between actions within a bot instance.

### 5.2. Proxy

*   `PROXY_ENABLED` (bool): If `True`, the bot attempts to use proxies loaded from the proxy file.
*   `PROXY_FILE` (str): Path to the text file containing proxies (default: `config/proxies.txt`).

### 5.3. Browser & Launch

*   `HEADLESS` (bool): If `True`, runs browsers without a visible window. If `False`, browser windows will appear.
*   `CHROMIUM_BLUE_ENABLED` (bool): If `True`, attempts to use a custom Chromium executable specified by `CHROMIUM_BLUE_PATH`.
*   `CHROMIUM_BLUE_PATH` (str): Filesystem path to the custom Chromium executable.
*   `CHROMIUM_BLUE_ARGS` (str): Space-separated extra command-line arguments to pass when launching the custom Chromium build.

### 5.4. Fingerprinting & Evasion

*   `FINGERPRINT_PROFILE_NAME` (str): Determines the fingerprinting strategy.
    *   Specific Profile Name (e.g., `"Default Realistic Chrome Win10"`): Uses the profile defined in `fingerprint_profiles.json`.
    *   `"Random"`: Selects a random profile from the JSON file for each bot instance.
    *   `"Generate & Use New UA"`: Generates a unique User Agent via Gemini for each bot instance and applies settings from a base profile (defaults to "Default Realistic Chrome Win10").
*   `DISABLE_AUTOMATION_FLAGS` (bool): If `True`, injects JavaScript to hide `navigator.webdriver` and other common automation flags.
*   `PREVENT_WEBRTC_IP_LEAK` (bool): If `True`, applies Chromium launch flag (`--force-webrtc-ip-handling-policy`) and JS hooks to prevent WebRTC from leaking local IP addresses. **Requires SOCKS5 proxy to also hide the public IP effectively via WebRTC.**
*   `USER_AGENT_SOURCE` (str): [Currently less used] Intended for future expansion (e.g., "Static", "Generated", "Combined"). Fingerprint selection logic handles this now.
*   `FINGERPRINT_FILE` (str): Path to the JSON file containing fingerprint profiles (default: `config/fingerprint_profiles.json`).
*   `VIEWPORT_MIN_WIDTH`, `VIEWPORT_MAX_WIDTH` (int): Range for random browser viewport width.
*   `VIEWPORT_MIN_HEIGHT`, `VIEWPORT_MAX_HEIGHT` (int): Range for random browser viewport height.

### 5.5. User Agent Generation (Gemini)

*   `GEMINI_API_KEY` (str): Your Google Gemini API key. **Required for all generation features.**
*   `USER_AGENT_GENERATION_ENABLED` (bool): If `True`, the bot will automatically generate new User Agents using Gemini when the existing pool (static + previously generated) is exhausted during a run.
*   `USER_AGENT_GENERATION_COUNT` (int): How many User Agents to generate automatically when triggered.
*   `USER_AGENTS_FILE` (str): Path to the JSON file for the static list of user agents (default: `data/user_agents.json`).
*   `GENERATED_USER_AGENTS_FILE` (str): Path to the JSON file where automatically and manually generated user agents are stored (default: `data/generated_user_agents.json`).

### 5.6. Behavior Simulation

*   `MOUSE_MOVEMENT_ENABLED` (bool): If `True`, simulates mouse cursor movements.
*   `SCROLL_DURATION_MIN`, `SCROLL_DURATION_MAX` (int): Range (in milliseconds) for the duration of individual smooth scroll animations during *random* scrolling.
*   `ENABLE_BEHAVIORAL_STATES` (bool): If `True`, the bot alternates between "Scanning" and "Reading" states, slightly influencing scroll speed and mouse movement frequency.
*   `SKIP_ACTION_PROBABILITY` (float): Probability (0.0 to 1.0) of randomly skipping a planned action (scroll, mouse move, click, etc.) in an interaction cycle.
*   `IMPORTANT_WORDS_FILE` (str): Path to the JSON file containing keywords used to guide simulated mouse hovering (default: `data/important_words.json`).

### 5.7. Interaction Features

*   `FORM_FILL_ENABLED` (bool): If `True`, the bot will attempt to find and fill forms on pages.
*   `IMPRESSION_ENABLED` (bool): If `True`, the bot will attempt to find and click pagination links (Next Page, etc.).
*   `NEXT_PAGE_SELECTORS` (List[str]): List of CSS selectors used to identify "next page" links/buttons.
*   `NEXT_PAGE_TEXT_FALLBACK` (List[str]): List of link/button text values to check if selectors fail (e.g., "Next", ">", "Next Page").
*   `AD_CLICK_ENABLED` (bool): If `True`, the bot will attempt to identify and click elements likely to be ads.
*   `AD_SELECTORS` (List[str]): List of CSS selectors used to identify potential ad elements (e.g., `.advertisement`, `ins.adsbygoogle`, `iframe[id*='google_ads']`).
*   `AD_CLICK_PROBABILITY` (float): Probability (0.0 to 1.0) that the bot will click an ad *if* one is found and `AD_CLICK_ENABLED` is true.

---

## 6. Data Files

*   `config/settings.py`: Main configuration file (see section 5).
*   `config/proxies.txt`: Stores proxies, one per line. Format: `HOST:PORT`, `USER:PASS@HOST:PORT`. Optionally specify type (http, https, socks4, socks5) after the address, separated by space (e.g., `1.2.3.4:8080 http`, `myuser:mypass@4.3.2.1:1080 socks5`). If type is omitted, `http` is assumed.
*   `config/fingerprint_profiles.json`: Stores detailed browser fingerprint profiles in JSON format. Includes definitions for navigator properties, screen, canvas, WebGL, timezone etc. Can be generated via Gemini or manually edited.
*   `data/user_agents.json`: A static list of fallback User-Agent strings.
*   `data/generated_user_agents.json`: Stores User-Agent strings generated via the Gemini API (manual or automatic).
*   `data/important_words.json`: A list of keywords (e.g., "Contact", "Price", "Download") that the bot uses to guide simulated mouse movements towards potentially relevant page elements.
*   `logs/bot_activity.log`: General log file recording major actions and events across all bots.
*   `logs/errors.log`: Detailed log file specifically for errors and exceptions.

---

## 7. Usage Guide (GUI)

### 7.1. Running the Bot

Execute the main Python script from your terminal:

```bash
python your_script_name.py
```

(Replace `your_script_name.py` with the actual filename).

### 7.2. Main Control Tab

*   **URLs:** Enter the target website URLs, one per line. Must start with `http://` or `https://`.
*   **Proxies:**
    *   **List:** Displays loaded proxies. Status indicators (colored circles) show validation results (Orange: Unchecked, Blue: Checking, Green: Valid, Red: Invalid).
    *   **Load:** Loads proxies from the file specified in `PROXY_FILE` (usually `config/proxies.txt`).
    *   **Check:** Initiates a connectivity check for all loaded proxies in the background. Buttons are disabled during the check.
    *   **Clear:** Removes all proxies from the list.
*   **Run Configuration:**
    *   **Total Bot Instances:** The total number of bot sessions to run across all groups.
    *   **Concurrent Instances per Group:** How many bots run simultaneously in each batch.
*   **Controls:**
    *   **Start Bot(s):** Begins the bot execution process based on the current configuration.
    *   **Stop All Bots:** Attempts to gracefully stop all currently running bot threads.
*   **Bot Logs:**
    *   A tabbed view showing real-time logs for each individual bot instance. Tabs are created as bots launch.
    *   Tabs for bots encountering critical errors are highlighted in red.

### 7.3. Settings Tab

This tab allows modification of most settings found in `config/settings.py`. Changes made here are used when "Start Bot(s)" is clicked.

*   **General:** Headless mode, Min/Max action delays.
*   **Proxy:** Enable/Disable proxy usage globally.
*   **Fingerprinting & Evasion:** Select fingerprint mode (Profile, Random, Generate), toggle automation flag hiding, toggle WebRTC leak prevention. Includes button to manually generate a new fingerprint profile via Gemini.
*   **User Agent Generation (Gemini):** Input Gemini API Key, enable/disable automatic UA generation, set auto-generation count. Includes controls for manually generating a specified number of UAs.
*   **Behavior Simulation:** Enable/Disable mouse movement, configure scroll animation duration, enable/disable behavioral states, set action skip probability.
*   **Interaction Features:** Enable/Disable form filling, impression/pagination, ad clicking. Configure selectors and text fallbacks for pagination, set ad click probability, configure ad selectors.
*   **Advanced / Custom Browser:** Enable custom Chromium, specify path and arguments.
*   **Save/Load Buttons:**
    *   **Load Settings:** Loads configuration from `config/settings.py`, overwriting current UI settings.
    *   **Save Settings:** Saves the current UI settings to `config/settings.py`, overwriting the file.

### 7.4. License Tab

*   **License Key:** Input field for the activation key.
*   **Activate License:** Attempts to validate and activate the entered key.
*   **Status:** Displays the current license status (Unknown, Active + Expiry, Expired, Invalid).

---

## 8. Fingerprint & User Agent Generation

The bot uses Google's Gemini API for advanced generation features, aiming for more unique and realistic browser profiles.

### 8.1. Gemini API Key

A valid Google Gemini API Key is **required** for:

*   Generating User Agents (manual or automatic).
*   Generating full Fingerprint Profiles (manual).
*   Generating AI-driven scroll patterns (optional behavior).

Enter the key in the "Settings" tab -> "User Agent Generation (Gemini)" section.

### 8.2. Fingerprint Profiles (`config/fingerprint_profiles.json`)

This JSON file defines detailed browser fingerprints.

*   It contains a list under the `"profiles"` key.
*   Each profile is an object with keys like `"name"`, `"description"`, `"navigator"`, `"screen"`, `"canvas"`, `"webgl"`, `"timezone"`.
*   The `"navigator"` object includes `user_agent`, `platform`, `vendor`, `languages`, `plugins`, `mimeTypes`, `hardwareConcurrency`, `deviceMemory`.
*   Defaults are created if the file is missing.
*   You can manually edit this file or use the "Generate New Profile (via Gemini)" button in Settings (requires API Key).

### 8.3. Fingerprint Modes

Selected via the "Fingerprint Mode" dropdown in Settings:

1.  **Specific Profile Name:** Uses the exact profile selected from the list (loaded from `fingerprint_profiles.json`).
2.  **`Random`:** Chooses a random profile from `fingerprint_profiles.json` for each new bot instance.
3.  **`Generate & Use New UA`:**
    *   Calls the Gemini API to generate a *single new User Agent string* for the bot instance.
    *   Saves this new UA to `data/generated_user_agents.json` if unique.
    *   Applies other fingerprint settings (WebGL, Canvas, etc.) from a *base profile* (usually the "Default Realistic Chrome Win10" profile, or another if specified/available).

### 8.4. Manual UA Generation

*   Go to "Settings" -> "User Agent Generation (Gemini)".
*   Enter your Gemini API Key.
*   Set the desired number of UAs to generate.
*   Click "Generate Now".
*   The bot calls Gemini, parses the results, filters out duplicates, and saves unique new UAs to `data/generated_user_agents.json`.
*   A status message indicates success or failure.

### 8.5. Manual Fingerprint Profile Generation

*   Go to "Settings" -> "Fingerprinting & Evasion".
*   Ensure your Gemini API Key is entered in the UA Gen section.
*   Click "Generate New Profile (via Gemini)".
*   The bot prompts Gemini for a complete, consistent profile in JSON format.
*   It validates the response structure.
*   If valid, it appends the new profile to `config/fingerprint_profiles.json` (checking for name uniqueness) and refreshes the "Fingerprint Mode" dropdown.
*   A status message indicates success or failure.

### 8.6. Automatic UA Generation

*   Requires `USER_AGENT_GENERATION_ENABLED = True` in settings and a valid `GEMINI_API_KEY`.
*   If the bot runs out of unique User Agents from `user_agents.json` and `generated_user_agents.json` during a run, it will automatically trigger a call to Gemini to generate `USER_AGENT_GENERATION_COUNT` new UAs.
*   These are saved to `generated_user_agents.json`, and the bot continues using the newly available pool.

---

## 9. Proxy Management

### 9.1. Loading Proxies (`config/proxies.txt`)

*   Add proxies to `config/proxies.txt`, one per line.
*   Supported formats:
    *   `HOST:PORT` (defaults to HTTP)
    *   `USER:PASS@HOST:PORT` (defaults to HTTP)
    *   `HOST:PORT TYPE` (e.g., `1.2.3.4:8080 http`, `1.2.3.4:1080 socks5`)
    *   `USER:PASS@HOST:PORT TYPE`
*   Valid types: `http`, `https` (treated as `http` for Playwright), `socks4`, `socks5`.
*   Use the "Load" button on the Main tab to load/reload proxies into the GUI list.

### 9.2. Checking Proxies

*   Click the "Check" button on the Main tab.
*   This launches background threads to test connectivity for each loaded proxy using Playwright against a test URL (like `api.ipify.org`).
*   The GUI list updates visually:
    *   Orange Circle: Unchecked
    *   Blue Circle: Checking
    *   Green Circle: Valid (connected successfully)
    *   Red Circle: Invalid (connection failed, timed out, or invalid format)
*   A summary message appears upon completion.

### 9.3. Proxy Usage

*   Enable proxy usage via the "Enable Proxy Usage" checkbox in Settings.
*   If enabled:
    *   If **valid** proxies exist (checked green), the bot cycles through only the valid ones, assigning one to each bot instance.
    *   If **no valid** proxies exist, the user is prompted whether to proceed using *any* loaded proxy (including unchecked or invalid). If yes, the bot cycles through all loaded proxies.
    *   If proxies are enabled but none are loaded, the bot will refuse to start.
*   **Note:** For effective WebRTC IP leak prevention (hiding public IP), **SOCKS5 proxies are strongly recommended** when `PREVENT_WEBRTC_IP_LEAK` is enabled. HTTP/HTTPS proxies typically won't mask the public IP through WebRTC, even with the flag enabled (though the flag still helps hide local IPs).

---

## 10. Logging

*   **GUI Logs:** Real-time logs for each bot instance appear in the tabs on the Main Control page. General status messages may also appear in the currently selected tab. Error tabs turn red.
*   **File Logs:**
    *   `logs/bot_activity.log`: Records high-level actions, bot start/stop events, and general information. Log level adjustable via `LOG_LEVEL` environment variable (default: INFO).
    *   `logs/errors.log`: Records detailed error messages, exceptions, and stack traces for debugging.

---

## 11. Troubleshooting

*   **Playwright Browser Issues:** If the bot fails to start with errors mentioning "Executable doesn't exist" or "missing dependencies":
    *   Run `playwright install chromium` in your terminal.
    *   On Linux, try `sudo playwright install-deps chromium`.
*   **Gemini API Errors:** If UA/Fingerprint/Scroll generation fails:
    *   Ensure a **valid** Google Gemini API Key is entered in Settings.
    *   Check your Gemini account for usage limits or billing issues.
    *   Check `logs/errors.log` for specific API error messages from Google.
    *   Network issues might prevent reaching the API.
*   **Proxy Errors:** If bots fail to connect:
    *   Use the "Check" button to validate proxies.
    *   Ensure proxies are in the correct format in `proxies.txt`.
    *   Verify the proxy provider allows connections from your IP and to the target sites.
    *   Remember SOCKS5 is best for privacy with WebRTC enabled.
*   **Bot Errors / Crashes:** Check the specific bot's log tab in the GUI and the `logs/errors.log` file for detailed error messages.

---

## 12. Final Disclaimer & Warning

**This software is provided for educational and research purposes only.**

**Automated interaction with websites, especially for the purpose of manipulating advertising metrics (like AdSense RPM) through artificial traffic, impressions, or clicks, is strictly prohibited by the terms of service of most platforms, including Google AdSense.**

**Engaging in such activities carries a very high risk of detection and will likely result in severe consequences, including but not limited to:**

*   **Permanent suspension of your advertising accounts (e.g., AdSense).**
*   **Forfeiture of any accrued earnings.**
*   **Potential legal action.**

**The creators and distributors of this software are not responsible for any misuse or any negative consequences arising from its use. You assume all risks and responsibilities associated with using this tool.**

**Do NOT use this bot on websites or ad accounts you are not explicitly authorized to test in this manner, and never use it in a way that violates any platform's terms of service.**
