#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cha-Cha Restaurant Voice Assistant - OpenAI Realtime API Version
A multilingual voice assistant powered by OpenAI's Realtime API for restaurant ordering
"""

import asyncio
import os
import json
import requests
import uuid
import sys
import traceback
import argparse
import time
import base64
import threading
import websocket
from datetime import datetime, timezone
from pathlib import Path
import pyaudio
import numpy as np
from scipy import signal
from dotenv import load_dotenv
import wave

# G1 Audio Streaming
try:
    audio_client_path = os.path.join(os.path.dirname(__file__), "..", "g1_audio")
    print(audio_client_path)
    sys.path.append(os.path.abspath(audio_client_path))
    import g1_audio_streaming

    G1_AUDIO_AVAILABLE = True
except ImportError:
    G1_AUDIO_AVAILABLE = False
    print("⚠️  G1 audio streaming not available. Using PyAudio fallback.")

# Supabase client (optional)
try:
    from supabase import create_client, Client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️  Supabase not available. Orders will be logged to console only.")

os.makedirs("recording", exist_ok=True)

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 24000  # OpenAI Realtime API uses 24kHz
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# OpenAI Realtime API configuration
OPENAI_WS_URL = "wss://api.openai.com/v1/realtime"
MODEL = "gpt-4o-realtime-preview-2025-06-03"

# Valid table numbers
VALID_TABLES = ["A1", "A2", "A3", "C1", "C2", "C3"]


class ChaChaVoiceAssistant:
    def __init__(self, debug=False, network_interface="eth0", interaction_timeout=120, force_audio_mode=None,
                 amplification_factor=1):
        # Load environment variables
        load_dotenv()

        self.debug = debug
        self.network_interface = network_interface
        self.interaction_timeout = interaction_timeout  # seconds (default 2 minutes)
        self.force_audio_mode = force_audio_mode  # "g1", "pyaudio", or None for auto

        # Validate and set amplification factor
        if amplification_factor < 0.1:
            print("⚠️  Warning: Amplification factor too low, setting to 0.1")
            self.amplification_factor = 0.1
        elif amplification_factor > 5.0:
            print("⚠️  Warning: Amplification factor too high, capping at 5.0 to prevent distortion")
            self.amplification_factor = 5.0
        else:
            self.amplification_factor = amplification_factor

        # Interaction tracking
        self.last_interaction_time = time.time()
        self.interaction_lock = asyncio.Lock()
        self.should_exit = False
        self.assistant_pcm = bytearray()
        self.customer_pcm = bytearray()

        self.flag_path = Path("/home/unitree/G1_deploy/l2_trigger_state.txt")

        # OpenAI API configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize Supabase client (optional)
        self.supabase_client = None
        if SUPABASE_AVAILABLE:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            if supabase_url and supabase_key:
                try:
                    self.supabase_client = create_client(supabase_url, supabase_key)
                    print("✅ Supabase client initialized")
                except Exception as e:
                    print(f"❌ Failed to initialize Supabase: {e}")

        # Initialize PyAudio for microphone
        self.pya = pyaudio.PyAudio()

        # Initialize audio output based on mode selection
        self.g1_audio = None
        self.use_g1_audio = False

        if self.force_audio_mode == "pyaudio":
            print("🔊 Forced PyAudio mode - skipping G1 audio initialization")

        elif self.force_audio_mode == "g1":
            print("🎵 Forced G1 audio mode - attempting G1 initialization...")
            if not G1_AUDIO_AVAILABLE:
                print("❌ G1 audio streaming library not available!")
                print("💡 Install g1_audio_streaming or use --audio-output pyaudio")
                raise RuntimeError("G1 audio forced but library not available")

            try:
                self.g1_audio = g1_audio_streaming.AudioClient()
                if self.g1_audio.init(self.network_interface):
                    self.use_g1_audio = True
                    print(f"✅ G1 audio streaming initialized via {self.network_interface}")
                    self.g1_audio.set_volume(100)  # Maximum volume
                    print("🔊 G1 volume set to 100 (maximum)")
                else:
                    print(f"❌ Failed to initialize G1 audio via {self.network_interface}")
                    raise RuntimeError("G1 audio initialization failed")
            except Exception as e:
                print(f"❌ G1 audio streaming error: {e}")
                raise RuntimeError(f"G1 audio forced but failed: {e}")

        else:  # Auto-detect mode (default)
            print("🔍 Auto-detecting audio output...")
            if G1_AUDIO_AVAILABLE:
                try:
                    self.g1_audio = g1_audio_streaming.AudioClient()
                    if self.g1_audio.init(self.network_interface):
                        self.use_g1_audio = True
                        print(f"✅ G1 audio streaming detected and initialized via {self.network_interface}")
                        self.g1_audio.set_volume(100)  # Maximum volume
                        print("🔊 G1 volume set to 100 (maximum)")
                    else:
                        print(f"⚠️  G1 audio available but failed to initialize via {self.network_interface}")
                except Exception as e:
                    print(f"⚠️  G1 audio streaming error: {e}")

        # Final audio output confirmation
        if self.use_g1_audio:
            print("🎵 Using G1 audio streaming for output")
        else:
            print("🔊 Using PyAudio for audio output")

        # Load menu data
        self.menu_data = self.load_menu_from_json_fallback("full_menu.json")  # Disabled live menu fetch for now

        # WebSocket connection variables
        self.websocket = None
        self.ws_thread = None
        self.ws_connected = False
        self.audio_in_queue = None
        self.out_queue = None
        self.audio_stream = None
        self.main_loop = None  # Store reference to main event loop
        self._pcm_queue = asyncio.Queue()

        # Enhanced microphone muting with audio duration tracking
        self.assistant_speaking = False
        self.mic_muted = False
        self.mic_mute_lock = None  # Will be initialized in run()

        # Audio duration tracking for better unmute timing
        self.audio_start_time = None
        self.total_audio_duration = 0.0  # Total duration of received audio in seconds
        self.unmute_task = None  # Task handle for delayed unmuting
        self.audio_buffer_delay = 0.5  # Extra buffer time in seconds before unmuting

        # Auto-amplification detection
        self.min_safe_amplification = float('inf')  # Track minimum safe amplification across session
        self.amplification_samples_count = 0  # Count of audio chunks analyzed

        # Audio processing counters for debug
        self.audio_chunk_count = 0

        # Create function definitions for order logging
        self.function_definitions = [
            {
                "type": "function",
                "name": "log_order_to_system",
                "description": "Log a completed order to the restaurant system with table number, items, quantities, predefined options, and special requests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_number": {
                            "type": "string",
                            "enum": VALID_TABLES,
                            "description": "Table number where the customer is seated"
                        },
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ordered menu item names in ENGLISH ONLY (name_en from menu), regardless of what language customer used"
                        },
                        "quantities": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of quantities corresponding to each item (same order as items array)"
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of PREDEFINED menu options for each item (ONLY 'Small', 'Large', or empty string '' for items without options). Must match length of items array."
                        },
                        "special_requests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of custom special requests for each item (e.g., 'extra spicy', 'no onions', or empty string '' for no special requests). Must match length of items array."
                        }
                    },
                    "required": ["table_number", "items", "quantities", "options", "special_requests"]
                }
            }
        ]

    def fetch_fresh_menu(self, fallback_file=None):
        """
        Fetch menu data directly from API and extract item names and prices

        Args:
            fallback_file (str, optional): Path to local JSON file to use if API fails

        Returns:
            list: Menu data in format [{"category": "Category Name(count)", "items": [{"name": "Item", "price": 12.95}]}]
        """
        url = "https://order.mealkeyway.com/merchant/4b303952355a315a58424d322b53386e6a6641396a773d3d/menu?productLine=SELF_DINE_IN&posPlatform=POS&onlineType=ONLINE_SELF_DINE_IN"

        try:
            print("🔄 Fetching fresh menu data from API...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract menu categories and items
            extracted_menu = []

            if 'menuCategories' in data:
                for category in data['menuCategories']:
                    category_info = {
                        "category": category['name']['en'],
                        "items": []
                    }

                    # Extract items from each category
                    if 'saleItems' in category:
                        for item in category['saleItems']:
                            # Get the English name, fallback to Chinese if English not available
                            item_name = item['name'].get('en', item['name'].get('zh-cn', 'Unknown'))
                            item_price = item.get('price', 0)

                            category_info['items'].append({
                                "name": item_name,
                                "price": item_price
                            })

                    # Add category count to category name
                    category_info['category'] += f"({len(category_info['items'])})"
                    extracted_menu.append(category_info)

            print(f"✅ Successfully loaded {len(extracted_menu)} menu categories from API")
            return extracted_menu

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching menu from API: {e}")
            if fallback_file:
                print("💡 Falling back to local menu file...")
                return self.load_menu_from_json_fallback(fallback_file)
            else:
                print("💡 No fallback file specified")
                return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing menu JSON: {e}")
            if fallback_file:
                return self.load_menu_from_json_fallback(fallback_file)
            return []
        except Exception as e:
            print(f"❌ Unexpected error loading menu: {e}")
            if fallback_file:
                return self.load_menu_from_json_fallback(fallback_file)
            return []

    def load_menu_from_json_fallback(self, json_path):
        """
        Fallback to load menu from local JSON file if API fails

        Args:
            json_path (str): Path to local JSON menu file

        Returns:
            list: Menu data in the same format as fetch_fresh_menu
        """
        try:
            print(f"📁 Loading fallback menu from {json_path}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                menu_data = json.load(f)
            print("✅ Fallback menu loaded successfully")
            return menu_data
        except FileNotFoundError:
            print(f"❌ Error: Fallback menu file '{json_path}' not found!")
            return []
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in '{json_path}'")
            return []
        except Exception as e:
            print(f"❌ Error loading fallback menu: {e}")
            return []

    def get_menu_context(self):
        """Create a formatted menu string for system instructions"""
        if not self.menu_data:
            return "Menu data not available."

        menu_text = "RESTAURANT MENU (English / Chinese):\n\n"
        for category in self.menu_data:
            # Show both language versions of category
            menu_text += f"{category['category_en']} / {category['category_cn']}:\n"
            for item in category['items']:
                # Show both language versions of item names
                item_display = f"  - {item['name_en']} / {item['name_cn']}"

                # Check if item has options (required selections)
                if 'options' in item and item['options']:
                    menu_text += f"{item_display} (REQUIRES SIZE SELECTION):\n"
                    for option in item['options']:
                        menu_text += f"    * {option['name']}: ${option['price']}\n"
                else:
                    menu_text += f"{item_display}: ${item['price']}\n"
            menu_text += "\n"

        return menu_text

    def resample_audio(self, audio_data, original_rate=24000, target_rate=16000):
        """Resample audio between different sample rates"""
        if original_rate == target_rate:
            return audio_data

        # Convert bytes to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_array = audio_data

        # Calculate resampling ratio
        num_samples = int(len(audio_array) * target_rate / original_rate)

        # Resample using scipy
        resampled = signal.resample(audio_array, num_samples)

        # Convert back to int16
        resampled_int16 = resampled.astype(np.int16)

        return resampled_int16

    def calculate_audio_duration(self, audio_bytes, sample_rate=24000, channels=1, bit_depth=16):
        """Calculate the duration of audio data in seconds"""
        bytes_per_sample = bit_depth // 8  # 16-bit = 2 bytes per sample
        total_samples = len(audio_bytes) // (bytes_per_sample * channels)
        duration = total_samples / sample_rate
        return duration

    def get_system_instructions(self):
        """Create comprehensive system instructions for the assistant"""
        menu_context = self.get_menu_context()

        return f"""You are Cha-Cha (English), 茶茶 (Chinese), or TéTé (Spanish) - a concise restaurant voice assistant. 

CORE BEHAVIOR:
- ONLY handle restaurant and menu questions
- For off-topic questions: "I'm not sure. Let me get a coworker to help you."
- Be friendly but BRIEF and DIRECT
- Match input language immediately
- Keep responses SHORT - avoid lengthy explanations

LANGUAGE SWITCHING:
- English: "Cha-Cha"
- Chinese: "茶茶" 
- Spanish: "TéTé"
- Switch languages INSTANTLY when detected
- No delays or context waiting
- IMPORTANT: Customers can use either English or Chinese dish/category names
- Always respond using the SAME language version the customer used
- Examples: If customer says "米饭", confirm "米饭"; if they say "Steamed Rice", confirm "Steamed Rice"

MENU ITEM HANDLING:
- Accept both English (name_en) and Chinese (name_cn) item names from customers
- When listing items or confirming orders, use the language the customer used
- CRITICAL: In tool calls, ALWAYS use English names (name_en) regardless of customer language
- Example: Customer orders "米饭" → Confirm "米饭" → Tool call uses "Steamed Rice"

ORDERING PROCESS:
- Help with menu questions directly
- Accept item names in either English or Chinese from customers
- Always respond using the same language the customer used
- If unclear pronunciation, confirm using customer's language: "Did you mean [item in customer's language]?"
- SIZE SELECTION: ONLY ask for size when item shows "(REQUIRES SIZE SELECTION)" in menu
- Items WITHOUT "(REQUIRES SIZE SELECTION)" do NOT need size - proceed directly
- For items that DO require size selection, ask: "What size - small or large?"
- Don't proceed until they pick a required option for items that need it
- SPECIAL REQUESTS: Accept custom requests like "extra spicy", "no onions", "less salt"
- Keep options and special requests separate:
  * Options: ONLY "Small" or "Large" (from menu, and only for items that have options)
  * Special requests: Everything else (extra ingredients, cooking preferences, etc.)
- Get table number: {', '.join(VALID_TABLES)}
- Keep responses brief during ordering: "Got it. Anything else?"
- IMPORTANT: When customer indicates they're done (says "that's it", "that's all", etc.), 
  repeat back the FULL order using the customer's language for confirmation:
  "Okay so that's 2 fried rice, 1 large hot and sour soup with less spicy. Did I get that right?"
- Wait for customer to confirm or make changes
- CRITICAL: In log_order_to_system function call, use ENGLISH item names only (name_en)
- ONLY call log_order_to_system function AFTER final confirmation
- If changes needed, update the order and confirm again

STYLE:
- Be helpful but concise
- No long explanations unless asked
- Quick responses, not chatty
- Professional efficiency

{menu_context}

Remember: Be helpful and accurate, but KEEP IT SHORT. Always confirm the complete order before submitting."""

    async def mute_microphone(self):
        """Mute the microphone during assistant speech"""
        async with self.mic_mute_lock:
            if not self.mic_muted:
                self.mic_muted = True
                self.assistant_speaking = True
                self.audio_start_time = time.time()
                self.total_audio_duration = 0.0
                print("🔇 MICROPHONE MUTED (assistant speaking)")

    async def schedule_delayed_unmute(self):
        """Schedule microphone unmuting after estimated audio completion"""
        if self.unmute_task:
            self.unmute_task.cancel()

        # Calculate delay: audio duration + buffer time
        delay = self.total_audio_duration + self.audio_buffer_delay

        if self.debug:
            print(
                f"🔇 Scheduling unmute in {delay:.2f}s (audio: {self.total_audio_duration:.2f}s + buffer: {self.audio_buffer_delay}s)")

        self.unmute_task = asyncio.create_task(self._delayed_unmute(delay))

    async def _delayed_unmute(self, delay):
        """Internal method to unmute after delay"""
        try:
            await asyncio.sleep(delay)
            await self.unmute_microphone()
        except asyncio.CancelledError:
            pass

    async def unmute_microphone(self):
        """Unmute the microphone when assistant stops speaking"""
        async with self.mic_mute_lock:
            if self.mic_muted:
                self.mic_muted = False
                self.assistant_speaking = False
                if self.unmute_task:
                    self.unmute_task.cancel()
                    self.unmute_task = None
                print("🎤 MICROPHONE UNMUTED (assistant finished)")

    async def update_interaction_time(self, interaction_type="unknown"):
        """Update the last interaction timestamp - called when meaningful interaction occurs"""
        async with self.interaction_lock:
            self.last_interaction_time = time.time()
            if self.debug:
                print(f"🔔 Interaction detected: {interaction_type} (timeout reset)")

    async def check_interaction_timeout(self):
        """Background task to monitor interaction timeout"""
        print(f"⏰ Auto-exit enabled: {self.interaction_timeout} seconds of inactivity")

        while not self.should_exit:
            try:
                # Check every 30 seconds
                await asyncio.sleep(30)

                async with self.interaction_lock:
                    time_since_last = time.time() - self.last_interaction_time
                    remaining_time = self.interaction_timeout - time_since_last

                if remaining_time <= 0:
                    print(f"\n⏰ No interaction for {self.interaction_timeout} seconds. Auto-exiting...")
                    print("👋 Cha-Cha going to sleep. Wake me up when you need help!")
                    self.should_exit = True
                    break
                elif remaining_time <= 60:  # Warn when < 1 minute left
                    if self.debug:
                        print(f"⏰ Auto-exit in {remaining_time:.0f} seconds (no recent interaction)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.debug:
                    print(f"❌ Timeout check error: {e}")

    def encode_audio_to_base64(self, audio_data):
        """Convert audio data to base64 string for OpenAI Realtime API"""
        if isinstance(audio_data, np.ndarray):
            # Convert numpy array to bytes
            audio_bytes = audio_data.astype(np.int16).tobytes()
        else:
            audio_bytes = audio_data

        return base64.b64encode(audio_bytes).decode('utf-8')

    def decode_audio_from_base64(self, base64_data):
        """Convert base64 string back to audio bytes"""
        audio_bytes = base64.b64decode(base64_data)
        return audio_bytes

    def amplify_audio(self, audio_bytes, amplification_factor=2.0):
        """Amplify audio volume with clipping protection and auto-amplification detection"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to float for processing
            audio_float = audio_array.astype(np.float32)

            # Calculate maximum safe amplification for this chunk (avoid division by zero)
            max_amplitude = np.max(np.abs(audio_float))
            if max_amplitude > 0:
                max_safe_amplification = 32767.0 / max_amplitude

                # Track minimum safe amplification across session
                if max_safe_amplification < self.min_safe_amplification:
                    self.min_safe_amplification = max_safe_amplification

                self.amplification_samples_count += 1

                # Debug info every 50 chunks to avoid spam
                if self.debug and self.amplification_samples_count % 50 == 1:
                    print(
                        f"🔍 Chunk max safe amp: {max_safe_amplification:.2f}x, Session min: {self.min_safe_amplification:.2f}x")

            # Apply amplification
            amplified = audio_float * amplification_factor

            # Prevent clipping by clamping to int16 range
            amplified = np.clip(amplified, -32768, 32767)

            # Convert back to int16
            amplified_int16 = amplified.astype(np.int16)

            # Convert back to bytes
            return amplified_int16.tobytes()

        except Exception as e:
            if self.debug:
                print(f"⚠️  Audio amplification error: {e}")
            # Return original audio if amplification fails
            return audio_bytes

    async def log_order(self, order_data):
        """Log simplified order to Supabase"""
        now_utc = datetime.now(timezone.utc)
        order_number = f"ORD-{now_utc.strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

        # Prepare simplified order record
        order_record = {
            "order_number": order_number,
            "table_number": order_data.get('table_number', ''),
            "items": order_data['items'],
            "quantities": order_data['quantities'],
            "options": order_data.get('options', []),  # Predefined menu options only
            "special_requests": order_data.get('special_requests', []),  # Custom requests
            "status": "new",
            "created_at": now_utc.isoformat(),
            "updated_at": now_utc.isoformat()
        }

        # Log to console with local time for readability
        local_time = datetime.now()
        print(f"\n🍽️  NEW ORDER: {order_number}")
        print(f"   Time: {local_time.strftime('%Y-%m-%d %H:%M:%S')} (Local)")
        print(f"   Table: {order_data.get('table_number', 'Unknown')}")
        print(f"   Items: {order_data['items']}")
        print(f"   Quantities: {order_data['quantities']}")
        print(f"   Options: {order_data.get('options', [])} (predefined sizes)")
        print(f"   Special Requests: {order_data.get('special_requests', [])} (custom requests)")
        print(
            f"   Total Items: {sum(order_data['quantities']) if order_data['quantities'] else len(order_data['items'])}")
        print("   Status: Order sent to kitchen\n")

        # Log to Supabase if available
        if self.supabase_client:
            try:
                result = self.supabase_client.table("restaurant_orders").insert(order_record).execute()
                print(f"✅ Order logged to database: {order_number}")
                return order_number
            except Exception as e:
                print(f"❌ Failed to log order to database: {e}")
                print("📋 Order logged to console only")

        return order_number

    async def connect_to_openai(self):
        """Connect to OpenAI Realtime API via WebSocket using websocket-client"""
        url = f"{OPENAI_WS_URL}?model={MODEL}"

        # Headers for websocket-client
        headers = [
            f"Authorization: Bearer {self.openai_api_key}",
            "OpenAI-Beta: realtime=v1"
        ]

        if self.debug:
            print(f"🔗 Connecting to: {url}")
            print(f"🔑 Using API key: {self.openai_api_key[:8]}...")

        try:
            # Create WebSocket connection with callbacks
            self.websocket = websocket.WebSocketApp(
                url,
                header=headers,
                on_open=self.on_ws_open,
                on_message=self.on_ws_message,
                on_error=self.on_ws_error,
                on_close=self.on_ws_close
            )

            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.websocket.run_forever, daemon=True)
            self.ws_thread.start()

            # Wait for connection to establish
            max_wait = 10  # seconds
            wait_time = 0
            while not self.ws_connected and wait_time < max_wait:
                await asyncio.sleep(0.1)
                wait_time += 0.1

            if self.ws_connected:
                print("✅ Connected to OpenAI Realtime API")
                return True
            else:
                print("❌ Failed to connect to OpenAI Realtime API: Connection timeout")
                return False

        except Exception as e:
            print(f"❌ Failed to connect to OpenAI Realtime API: {e}")
            if self.debug:
                print(f"🐛 Full error details: {type(e).__name__}: {str(e)}")
                print("💡 Check your OpenAI API key and internet connection")
            return False

    def on_ws_open(self, ws):
        """WebSocket connection opened"""
        self.ws_connected = True
        if self.debug:
            print("🔗 WebSocket connection opened")

    def on_ws_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            # Put the message in the queue for async processing
            if hasattr(self, 'message_queue') and self.main_loop:
                # Use thread-safe method to put message in async queue
                future = asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(message),
                    self.main_loop
                )
        except Exception as e:
            if self.debug:
                print(f"❌ Error queuing WebSocket message: {e}")

    def on_ws_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket error: {error}")
        self.ws_connected = False

    def on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        if self.debug:
            print(f"📡 WebSocket connection closed: {close_status_code} - {close_msg}")
        self.ws_connected = False

    async def send_to_openai(self, data):
        """Send data to OpenAI WebSocket"""
        if self.websocket and self.ws_connected:
            try:
                message = json.dumps(data)
                self.websocket.send(message)
                if self.debug:
                    print(f"📤 Sent to OpenAI: {data.get('type', 'unknown')}")
                return True
            except Exception as e:
                print(f"❌ Error sending to OpenAI: {e}")
                return False
        else:
            print("❌ WebSocket not connected")
            return False

    async def send_session_update(self):
        """Send session configuration to OpenAI"""
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.get_system_instructions(),
                "voice": "echo",  # Clear, direct voice for restaurant
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 1000
                },
                "tools": self.function_definitions,
                "tool_choice": "auto"
            }
        }

        await self.send_to_openai(session_update)
        if self.debug:
            print("📤 Sent session update to OpenAI")

    async def _pcm_worker(self):
        """Background thread: pull chunks from the queue and extend the buffer."""
        while True:
            data = await self._pcm_queue.get()   # blocks until data is available
            self.customer_pcm.extend(data)

    async def listen_audio(self):
        """Capture audio from microphone and send to OpenAI"""
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self._pcm_queue.put(data)

                # Check if microphone is muted (assistant is speaking)
                async with self.mic_mute_lock:
                    if self.mic_muted:
                        # Skip sending audio while assistant is speaking
                        # Small delay to avoid busy waiting
                        await asyncio.sleep(0.1)
                        continue

                # Convert audio to base64 and send to OpenAI
                audio_array = np.frombuffer(data, dtype=np.int16)
                base64_audio = self.encode_audio_to_base64(audio_array)

                append_event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_audio
                }

                await self.send_to_openai(append_event)

                if self.debug:
                    print(f"🎤 Sent audio chunk: {len(data)} bytes")

            except Exception as e:
                print(f"❌ Audio capture error: {e}")
                break

    def _save_assistant_audio(self, filename: str, pcm_bytes: bytes):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)         # 16-bit
            wf.setframerate(24000)     # 24 kHz
            wf.writeframes(pcm_bytes)
            self.assistant_pcm.clear()

    def _save_user_audio(self, filename: str, pcm_bytes: bytes):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)         # 16-bit
            wf.setframerate(24000)     # 24 kHz
            wf.writeframes(pcm_bytes)
            self.customer_pcm.clear()

    async def handle_openai_events(self):
        """Handle incoming events from OpenAI Realtime API"""
        try:
            while not self.should_exit:
                try:
                    # Get message from queue with timeout
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                    try:
                        event = json.loads(message)
                        event_type = event.get("type", "")

                        if self.debug:
                            print(f"📥 Received event: {event_type}")

                        # Handle different event types
                        if event_type == "session.created":
                            print("🎉 OpenAI session created")
                            await self.update_interaction_time("session_created")

                        elif event_type == "session.updated":
                            print("✅ OpenAI session updated")

                        elif event_type == "response.audio.delta":
                            # Stream audio output - mute microphone during speech
                            audio_delta = event.get("delta", "")
                            if audio_delta:
                                # Mute microphone when assistant starts speaking
                                if not self.assistant_speaking:
                                    await self.mute_microphone()

                                audio_bytes = self.decode_audio_from_base64(audio_delta)

                                # Track audio duration for better unmute timing
                                chunk_duration = self.calculate_audio_duration(audio_bytes)
                                self.total_audio_duration += chunk_duration

                                # Amplify the audio volume for better output
                                amplified_audio = self.amplify_audio(audio_bytes,
                                                                     amplification_factor=self.amplification_factor)

                                if self.debug:
                                    self.audio_chunk_count += 1
                                    # Only show amplification debug every 10 chunks to reduce spam
                                    if self.audio_chunk_count % 10 == 1:
                                        original_size = len(audio_bytes)
                                        amplified_size = len(amplified_audio)
                                        print(
                                            f"🔊 Audio amplified: {original_size} → {amplified_size} bytes ({self.amplification_factor}x volume)")
                                        print(f"🕒 Total audio duration: {self.total_audio_duration:.2f}s")

                                await self.audio_in_queue.put(amplified_audio)
                                self.assistant_pcm.extend(audio_bytes)
                                await self.update_interaction_time("audio_response")

                        elif event_type == "response.text.delta":
                            # Handle text responses (for debugging)
                            text_delta = event.get("delta", "")
                            if text_delta:
                                # Try to detect language of response for debugging
                                lang_indicator = ""
                                if any(char in text_delta for char in "你好茶茶我想点菜"):
                                    lang_indicator = " [🇨🇳 Chinese]"
                                elif any(word in text_delta.lower() for word in
                                         ["hola", "tété", "quiero", "cuánto", "español"]):
                                    lang_indicator = " [🇪🇸 Spanish]"
                                elif any(word in text_delta.lower() for word in ["cha-cha", "hello", "hi", "english"]):
                                    lang_indicator = " [🇺🇸 English]"

                                print(f"🤖 Cha-Cha{lang_indicator}: {text_delta}", end="", flush=True)
                                await self.update_interaction_time("text_response")

                        elif event_type == "response.text.done":
                            full_response = event.get("text", "")
                            with open("response.txt", "w", encoding="utf-8") as f:
                                f.write(full_response)
                            print(f"✅ Saved response to response.txt:\n\n{full_response}")

                        elif event_type == "response.function_call_arguments.delta":
                            # Function call in progress
                            if self.debug:
                                print(f"🔧 Function call delta: {event.get('delta', '')}")

                        elif event_type == "response.audio.done":
                            # Assistant finished sending audio - schedule delayed unmute
                            if self.assistant_speaking:
                                await self.schedule_delayed_unmute()
                            if self.debug:
                                print("🎵 Assistant finished sending audio")

                            now = datetime.now().strftime("%H%M%S")
                            date = datetime.now().strftime("%Y%m%d")

                            os.makedirs(date, exist_ok=True)

                            fname = rf"{date}\call_assistant_{now}.wav"
                            await asyncio.to_thread(self._save_assistant_audio, fname, self.assistant_pcm)
                            print(f"✅ Saved {fname}")
                            
                            fname = rf"{date}\call_customer_{now}.wav"
                            await asyncio.to_thread(self._save_user_audio, fname, self.customer_pcm)
                            print(f"✅ Saved {fname}")

                        elif event_type == "response.done":
                            # Response completed - ensure unmute is scheduled if not already
                            if self.assistant_speaking and not self.unmute_task:
                                await self.schedule_delayed_unmute()

                            response = event.get("response", {})
                            output = response.get("output", [])

                            for item in output:
                                if item.get("type") == "function_call":
                                    await self.handle_function_call(item)
                                    await self.update_interaction_time("function_call")

                            # Handle usage metadata if available
                            usage = response.get("usage", {})
                            if usage and self.debug:
                                total_tokens = usage.get("total_tokens", 0)
                                input_tokens = usage.get("input_tokens", 0)
                                output_tokens = usage.get("output_tokens", 0)
                                print(f"🔢 Used {total_tokens} tokens (input: {input_tokens}, output: {output_tokens})")

                        elif event_type == "input_audio_buffer.speech_started":
                            # User started speaking - immediately unmute microphone and cancel any scheduled unmute
                            if self.assistant_speaking:
                                await self.unmute_microphone()
                                if self.debug:
                                    print("🎤 User interrupted - microphone unmuted immediately")
                            if self.debug:
                                print("🎤 Speech detected")

                        elif event_type == "input_audio_buffer.speech_stopped":
                            if self.debug:
                                print("🤐 Speech ended")

                        elif event_type == "error":
                            error_message = event.get("error", {}).get("message", "Unknown error")
                            print(f"❌ OpenAI API error: {error_message}")

                    except json.JSONDecodeError as e:
                        print(f"❌ Failed to parse OpenAI event: {e}")
                    except Exception as e:
                        print(f"❌ Error handling OpenAI event: {e}")

                except asyncio.TimeoutError:
                    # Timeout is normal, just continue
                    continue
                except Exception as e:
                    print(f"❌ Error in message queue: {e}")
                    break

        except Exception as e:
            print(f"❌ Error in OpenAI event handler: {e}")

    async def handle_function_call(self, function_call_item):
        """Handle function calls from OpenAI"""
        function_name = function_call_item.get("name")
        call_id = function_call_item.get("call_id")
        arguments_str = function_call_item.get("arguments", "{}")

        print(f"\n🔧 FUNCTION CALL RECEIVED")
        print(f"   Function: {function_name}")
        print(f"   Call ID: {call_id}")
        print(f"   Arguments: {arguments_str}")

        try:
            arguments = json.loads(arguments_str)

            if function_name == "log_order_to_system":
                order_data = {
                    'table_number': arguments.get('table_number'),
                    'items': arguments.get('items', []),
                    'quantities': arguments.get('quantities', []),
                    'options': arguments.get('options', []),  # Predefined menu options only
                    'special_requests': arguments.get('special_requests', [])  # Custom requests
                }

                print(f"   📋 Processing order: {order_data}")
                order_number = await self.log_order(order_data)

                # Send function response back to OpenAI
                function_output = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({
                            "success": True,
                            "order_number": order_number,
                            "message": f"Order {order_number} successfully logged to kitchen system"
                        })
                    }
                }

                await self.send_to_openai(function_output)

                # Create a response to continue the conversation
                response_create = {
                    "type": "response.create"
                }
                await self.send_to_openai(response_create)

                print(f"   ✅ Function response sent: Order logged successfully")

            else:
                print(f"   ⚠️  Unknown function call: {function_name}")

                # Send error response
                error_output = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({
                            "success": False,
                            "error": f"Unknown function: {function_name}"
                        })
                    }
                }
                await self.send_to_openai(error_output)

        except Exception as e:
            print(f"   ❌ Error processing function call: {e}")

            # Send error response
            error_output = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({
                        "success": False,
                        "error": f"Error processing function call: {str(e)}"
                    })
                }
            }
            await self.send_to_openai(error_output)

        print("   🔧 Function call processing complete\n")

    async def play_audio(self):
        """Play audio responses through G1 speakers or PyAudio fallback"""
        if self.use_g1_audio:
            # Use G1 audio streaming
            print("🎵 Starting G1 audio streaming...")

            # Start continuous streaming for low latency
            if not self.g1_audio.start_continuous():
                print("❌ Failed to start G1 continuous streaming")
                return

            try:
                while True:
                    # Get audio data from OpenAI (24kHz, int16)
                    bytestream = await self.audio_in_queue.get()

                    # Resample from 24kHz to 16kHz for G1
                    audio_array = np.frombuffer(bytestream, dtype=np.int16)
                    resampled_audio = self.resample_audio(audio_array, 24000, 16000)

                    # Stream to G1
                    start_time = time.time() if self.debug else None
                    success = self.g1_audio.add_chunk(resampled_audio)

                    if self.debug and start_time:
                        stream_time = (time.time() - start_time) * 1000
                        print(
                            f"🎵 G1 audio chunk: {'✅' if success else '❌'} ({stream_time:.1f}ms, {len(resampled_audio)} samples)")

                    if not success and self.debug:
                        print("⚠️  G1 audio streaming failed, continuing...")

            except Exception as e:
                print(f"❌ G1 audio streaming error: {e}")
                # Ensure microphone is unmuted on audio error
                if self.assistant_speaking:
                    await self.unmute_microphone()
            finally:
                # Clean up continuous stream and unmute microphone
                self.g1_audio.stop_stream()
                if self.assistant_speaking:
                    await self.unmute_microphone()
                print("🔇 G1 audio streaming stopped")
        else:
            # Use PyAudio (computer speakers/headphones)
            print("🔊 Starting PyAudio output...")
            stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )

            try:
                while True:
                    bytestream = await self.audio_in_queue.get()
                    await asyncio.to_thread(stream.write, bytestream)

                    if self.debug:
                        print(f"🔊 PyAudio chunk: ✅ ({len(bytestream)} bytes)")

            except Exception as e:
                print(f"❌ PyAudio playback error: {e}")
                # Ensure microphone is unmuted on audio error
                if self.assistant_speaking:
                    await self.unmute_microphone()
            finally:
                stream.close()
                # Ensure microphone is unmuted when audio stops
                if self.assistant_speaking:
                    await self.unmute_microphone()
                print("🔇 PyAudio output stopped")

    async def send_text(self):
        """Handle text input for testing (type 'q' to quit)"""
        while True:
            try:
                text = await asyncio.to_thread(input, "💬 Type message (or 'q' to quit): ")
                if text.lower() == "q":
                    break

                # Track user interaction (manual text input)
                await self.update_interaction_time("user_text_input")

                # Send text as conversation item to OpenAI
                text_item = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": text
                            }
                        ]
                    }
                }

                await self.send_to_openai(text_item)

                # Create response
                response_create = {
                    "type": "response.create"
                }
                await self.send_to_openai(response_create)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Send text error: {e}")
                break

    async def remote_poll(self):
        """在主线程里循环调用，非阻塞读取遥控器事件"""

        while True:
            controller = None
            with self.flag_path.open('r') as f:
                lines = f.read().splitlines()
                if lines:
                    controller = lines[-1].strip()

            if controller == "None":
                continue
            if controller == "L2_pressed":
                try:
                    text = "Say: 'Here is your bill'"

                    # Track user interaction (manual text input)
                    await self.update_interaction_time("user_text_input")

                    # Send text as conversation item to OpenAI
                    text_item = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": text
                                }
                            ]
                        }
                    }

                    await self.send_to_openai(text_item)

                    # Create response
                    response_create = {
                        "type": "response.create"
                    }
                    await self.send_to_openai(response_create)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Send text error: {e}")
                    break
                finally:
                    print("Sent text successfully")
                    self.flag_path.write_text("None\n")

            await asyncio.sleep(0.02)

    async def run(self):
        """Main execution loop"""
        try:
            # Store reference to main event loop and initialize locks
            self.main_loop = asyncio.get_event_loop()
            self.mic_mute_lock = asyncio.Lock()

            print("🎤 Starting Cha-Cha voice assistant...")
            print("🎧 Make sure you're using headphones to prevent audio feedback!")
            print("💬 Speak naturally or type 'q' to quit")
            print("🌍 Supported languages: English, Chinese (Mandarin), Spanish")
            print(f"🍽️  Valid table numbers: {', '.join(VALID_TABLES)}")
            print("🔊 VAD Settings: Server-side voice activity detection enabled")
            print("🔇 Smart microphone muting with audio duration estimation enabled")
            print(f"🔊 Audio boost: {self.amplification_factor}x volume amplification active")
            print("🔍 Auto-amplification analysis enabled (optimal settings will be shown on exit)")

            # Audio output status
            if self.force_audio_mode:
                audio_mode_text = f"🎵 Audio Output: {self.force_audio_mode.upper()} (forced)"
            else:
                audio_mode_text = "🎵 Audio Output: AUTO-DETECT"

            if self.use_g1_audio:
                print(f"{audio_mode_text} → G1 Streaming via {self.network_interface}")
            else:
                print(f"{audio_mode_text} → PyAudio")

            print(f"⏰ Auto-exit: {self.interaction_timeout} seconds of inactivity")
            print(f"⏱️  Microphone unmute buffer: {self.audio_buffer_delay}s after audio completion")
            print("-" * 60)

            # Connect to OpenAI Realtime API
            if not await self.connect_to_openai():
                raise RuntimeError("Failed to connect to OpenAI Realtime API")

            # Initialize audio queues and message queue
            self.audio_in_queue = asyncio.Queue()
            self.out_queue = asyncio.Queue(maxsize=10)
            self.message_queue = asyncio.Queue()

            async with asyncio.TaskGroup() as tg:
                # Send initial session configuration
                await self.send_session_update()

                # Create tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.listen_audio())
                tg.create_task(self.handle_openai_events())
                tg.create_task(self.play_audio())
                tg.create_task(self._pcm_worker())
                tg.create_task(self.remote_poll())

                # Add timeout monitoring task
                timeout_task = tg.create_task(self.check_interaction_timeout())

                # Wait for either user quit or timeout
                done, pending = await asyncio.wait(
                    [send_text_task, timeout_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Check what caused the exit
                if timeout_task in done:
                    print("\n⏰ Session timed out due to inactivity")
                    raise asyncio.CancelledError("Interaction timeout reached")
                else:
                    print("\n👤 User requested exit")
                    self.should_exit = True
                    raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            traceback.print_exc()
        finally:
            # Cancel any pending unmute task
            if hasattr(self, 'unmute_task') and self.unmute_task:
                self.unmute_task.cancel()

            # Ensure microphone is unmuted on exit
            if hasattr(self, 'mic_mute_lock') and self.mic_mute_lock:
                try:
                    async with self.mic_mute_lock:
                        if self.mic_muted:
                            self.mic_muted = False
                            self.assistant_speaking = False
                            print("🎤 MICROPHONE UNMUTED (cleanup)")
                except:
                    pass

            # 1. Close WebSocket connection first (stop receiving audio from OpenAI)
            if self.websocket:
                try:
                    self.websocket.close()
                    print("📡 OpenAI WebSocket connection closed")
                except:
                    pass

            # 2. Close microphone input stream
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.close()
                print("🎤 Microphone input closed")

            # 3. Stop audio output systems
            if self.use_g1_audio and self.g1_audio:
                try:
                    self.g1_audio.stop_stream()
                    print("🔇 G1 audio cleanup complete")
                except:
                    pass

            # 4. Terminate PyAudio last
            self.pya.terminate()
            print("🔊 PyAudio terminated")

            # Report optimal amplification factor
            if self.amplification_samples_count > 0 and self.min_safe_amplification != float('inf'):
                print(f"\n🔍 OPTIMAL AMPLIFICATION ANALYSIS:")
                print(f"   📊 Analyzed {self.amplification_samples_count} audio chunks")
                print(f"   🎯 Maximum safe amplification: {self.min_safe_amplification:.2f}x")
                print(f"   ⚙️  Current amplification: {self.amplification_factor:.2f}x")

                if self.amplification_factor > self.min_safe_amplification:
                    print(f"   ⚠️  Current setting may cause clipping!")
                    print(f"   💡 Recommended: --amplification {self.min_safe_amplification:.2f}")
                elif self.amplification_factor < self.min_safe_amplification * 0.8:
                    print(f"   📈 You could safely increase volume!")
                    print(f"   💡 Suggested: --amplification {self.min_safe_amplification * 0.9:.2f}")
                else:
                    print(f"   ✅ Current amplification is well-tuned!")
            else:
                print(f"\n🔍 No audio analysis data available (no chunks processed)")

            # Final message
            if not self.should_exit:
                print("👋 Cha-Cha shutting down. Have a great day!")
            else:
                print("👋 Goodbye! Come back anytime for more help.")

def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Cha-Cha Restaurant Voice Assistant - OpenAI Realtime API")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    parser.add_argument("--interface", default="eth0", help="Network interface for G1 audio (default: eth0)")
    parser.add_argument("--timeout", type=int, default=120, help="Auto-exit timeout in seconds (default: 120)")
    parser.add_argument("--audio-output", choices=["auto", "g1", "pyaudio"], default="auto",
                        help="Force audio output mode: auto (detect), g1 (force G1), pyaudio (force PyAudio)")
    parser.add_argument("--amplification", type=float, default=1,
                        help="Audio amplification factor for volume boost (default: 1.75, range: 0.1-5.0, 1.0=no boost)")
    args = parser.parse_args()

    # Convert audio output argument
    force_audio_mode = None if args.audio_output == "auto" else args.audio_output

    try:
        assistant = ChaChaVoiceAssistant(
            debug=args.debug,
            network_interface=args.interface,
            interaction_timeout=args.timeout,
            force_audio_mode=force_audio_mode,
            amplification_factor=args.amplification
        )

        # Debug information
        if args.debug:
            print("🐛 Debug mode enabled - detailed logging will be shown")

        # Audio mode information
        print(f"🎵 Audio mode: {args.audio_output}")
        print(f"🔊 Audio amplification: {args.amplification}x")
        if assistant.use_g1_audio:
            print(f"🎵 Using G1 audio streaming via {args.interface}")
        else:
            print("🔊 Using PyAudio mode")

        asyncio.run(assistant.run())

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()