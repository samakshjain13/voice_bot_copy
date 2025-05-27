import sounddevice as sd
import numpy as np
import queue
import requests
import json
import uuid
import torch
import os
import threading
import webrtcvad
import re
import datetime
import time
import simpleaudio as sa
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.playback import play
from TTS.api import TTS
from shutil import copyfile

def backup_conversation():
    
    if os.path.exists(CONVERSATION_FILE):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"conversation_history_{timestamp}.json"
        copyfile(CONVERSATION_FILE, backup_name)
    else:
        print("⚠️ No conversation history to back up.")    

def numpy_float32_to_pcm16(audio_np: np.ndarray) -> bytes:
    # Convert mono float32 audio (-1.0 to 1.0) to 16-bit PCM
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    return audio_int16.tobytes()

# === CONFIG ===
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5  # seconds
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
MAX_BUFFER_SECONDS = 5

vad = webrtcvad.Vad()
vad.set_mode(3)  # Aggressiveness mode (0-3). 3 = most aggressive, 0 = least.

# === Initialize Faster-Whisper ===
model = WhisperModel("tiny", device="cpu", compute_type="int8")  # Use "tiny" for lower latency

# === Audio queue ===
audio_queue = queue.Queue()

tts_playback_thread = None
tts_interrupt_flag = threading.Event()

def audio_callback(indata, frames, time, status):
    if status:
        print("Stream status:", status)

    # Check for early voice activity directly on raw float32
    flat = indata[:, 0]
    frame_duration_ms = 30
    frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)
    num_frames = 3  # total 90ms VAD check

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        if end <= len(flat):
            pcm_bytes = numpy_float32_to_pcm16(flat[start:end])
            if is_speech(pcm_bytes):
                if tts_playback_thread and tts_playback_thread.is_alive():
                    tts_interrupt_flag.set()
                    break   
    
    audio_queue.put(indata.copy())

# === XTTS setup ===
tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=False)
tts.to("cpu") 

# === Groq API setup ===
GROQ_API_KEY = "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"  
GROQ_MODEL = "llama3-70b-8192"

CONVERSATION_FILE = "conversation_history.json"

# === ERP System Prompt Templates ===
BASE_PROMPT = """You are an expert ERP sales representative trained to communicate with potential customers in industries like {industry_name}.
You speak clearly, confidently, and persuasively — like a professional sales agent on a call.

Your goal is to:
1. Greet the customer warmly.
2. Quickly explain the benefits of the ERP system tailored for {industry_name}.
3. Overcome objections or hesitation with facts and friendly reassurance.
4. Guide the user toward booking a product demo or scheduling a callback.

Tone: Conversational, helpful, assertive — but never pushy.
Avoid jargon. Speak in benefits like:
- "{benefit_1}"
- "{benefit_2}"
- "{benefit_3}"

NEVER say you're a bot. Respond like a human agent.

If the user says they are busy or uninterested, offer to reschedule politely or send more info via WhatsApp or email.
"""

def get_prompt_for_erp(erp_type):
    if erp_type == "real_estate":
        return BASE_PROMPT.format(
            industry_name="real estate",
            benefit_1="Track projects, sales, and payments in real time",
            benefit_2="Automate broker commissions and unit-wise pricing",
            benefit_3="Ensure RERA compliance and streamline documentation"
        )
    elif erp_type == "construction":
        return BASE_PROMPT.format(
            industry_name="construction",
            benefit_1="Manage contractors, procurement, and billing in one place",
            benefit_2="Track site-wise expenses and material usage live",
            benefit_3="Reduce manual errors and project delays by 60%"
        )
    elif erp_type == "mall":
        return BASE_PROMPT.format(
            industry_name="mall management",
            benefit_1="Handle tenant billing and agreements efficiently",
            benefit_2="Automate electricity and maintenance charges",
            benefit_3="Generate real-time reports for occupancy and revenue"
        )
    else:
        return BASE_PROMPT.format(
            industry_name="your industry",
            benefit_1="Streamline operations",
            benefit_2="Automate billing and reporting",
            benefit_3="Track everything in real-time"
        )

def load_conversation_history():
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as f:
            return json.load(f)
    else:
        return [{"role": "system", "content": get_prompt_for_erp("real_estate")}]

def save_conversation_history(history):
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(history, f, indent=2)

# Load at start
conversation_history = load_conversation_history()

def get_llama_response(user_input, erp_type="real_estate"):
    global conversation_history

    conversation_history.append({"role": "user", "content": user_input})
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL,
        "messages": conversation_history  # full chat memory
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        assistant_message = response.json()["choices"][0]["message"]["content"].strip()
        # Save assistant's reply into conversation history
        conversation_history.append({"role": "assistant", "content": assistant_message})
        save_conversation_history(conversation_history)
        return assistant_message
    else:
        print("Groq error:", response.text)
        return "Sorry, I had trouble generating a response."

def wait_for_file(file_path, timeout=5):
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            raise TimeoutError("TTS output file did not appear.")
        time.sleep(0.1)
    return True

def speak(text):
    global tts_playback_thread
    tts_interrupt_flag.clear()  # Reset flag

    if tts_playback_thread and tts_playback_thread.is_alive():
        tts_interrupt_flag.set()
        tts_playback_thread.join()

    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"tts_out_{uuid.uuid4().hex}.wav")
    tts.tts_to_file(text=text, file_path=output_path)

    wait_for_file(output_path)
    audio = AudioSegment.from_wav(output_path)

    def play_audio():
        chunk_ms = 100
        for i in range(0, len(audio), chunk_ms):
            if tts_interrupt_flag.is_set():
                print("⏹️ Playback interrupted by user speech.")
                return
            
            chunk = audio[i:i + chunk_ms]
            play_obj = sa.play_buffer(
                chunk.raw_data,
                num_channels=chunk.channels,
                bytes_per_sample=chunk.sample_width,
                sample_rate=chunk.frame_rate
            )
            play_obj.wait_done()    

    tts_playback_thread = threading.Thread(target=play_audio)
    tts_playback_thread.start() 
    tts_playback_thread.join()

def remove_stutters(text):
    # Remove repeated characters like "I-I-I want"
    text = re.sub(r'\b(\w+)(-\1)+\b', r'\1', text)
    
    # Remove repeated words like "I I I want"
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    
    return text

def remove_fillers(text):
    fillers = r'\b(um+|uh+|like|you know|i mean|so|well|hmm+)\b'
    return re.sub(fillers, '', text, flags=re.IGNORECASE).strip()

def remove_pauses(text):
    text = re.sub(r'\.{2,}', ' ', text)             # Replace "..." or "......"
    text = re.sub(r'\[pause[^\]]*\]', ' ', text)    # Remove things like "[pause 1s]"
    return text.strip()

def clean_transcription(text):
    text = remove_stutters(text)
    text = remove_fillers(text)
    text = remove_pauses(text)
    text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
    return text.strip()


def handle_conversation(user_text, erp_type="real_estate"):
    global conversation_history

    print(f"\n🗣️  User: {user_text}")

        # 🔁 Reset conversation if user asks
    if "reset conversation" in user_text.lower():
        conversation_history = [{"role": "system", "content": get_prompt_for_erp(erp_type)}]
        save_conversation_history(conversation_history)
        print("🤖 Bot: Conversation reset.")
        speak("Conversation reset. How can I assist you now?")
        return
    
    bot_response = get_llama_response(user_text, erp_type)
    print(f"🤖 Bot: {bot_response}")
    speak(bot_response)

def is_speech(pcm_bytes: bytes, sample_rate: int = 16000) -> bool:
    frame_duration_ms = 30  # must be 10, 20, or 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)  # e.g., 480 for 30ms at 16kHz

    if len(pcm_bytes) != frame_size * 2:  # 2 bytes per sample
        return False  # Invalid frame size

    try:
        return vad.is_speech(pcm_bytes, sample_rate)
    except webrtcvad.Error:
        return False

# === Transcription loop ===
def transcribe_stream():
    buffer = np.zeros((0, 1), dtype=np.float32)
    print("🎙️  Listening... Press Ctrl+C to stop.")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        callback=audio_callback, blocksize=BLOCK_SIZE):
        try:
            while True:
                chunk = audio_queue.get()

                # Flatten chunk from shape (N, 1) to (N,)
                flat_chunk = chunk[:, 0]

                # Prepare 30ms frame (needed by VAD)
                frame_duration_ms = 30
                frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)
                if len(flat_chunk) >= frame_size:
                    vad_frame = flat_chunk[:frame_size]
                    pcm_bytes = numpy_float32_to_pcm16(vad_frame)
                    # Only add to buffer if VAD detects speech
                    if is_speech(pcm_bytes):
                        # Interrupt TTS if speaking
                        if tts_playback_thread and tts_playback_thread.is_alive():
                            tts_interrupt_flag.set()
                            tts_playback_thread.join()  # Wait until TTS playback stops

                        buffer = np.concatenate((buffer, chunk), axis=0)
                    else:
                        # Transcribe when enough audio is collected
                        if len(buffer) >= SAMPLE_RATE * 2:
                            samples = buffer[:, 0].astype(np.float32)
                            segments, _ = model.transcribe(samples, beam_size=1, vad_filter=True)

                            full_text = ""
                            low_confidence = False
                            confidence_threshold = -1.5  # Adjust as needed; closer to 0 is better

                            for seg in segments:
                                if seg.avg_logprob < confidence_threshold:
                                    low_confidence = True
                                    break  # No need to process further
                                full_text += seg.text.strip() + " "

                            if low_confidence or not full_text.strip():
                                print("🤖 Bot: I couldn't understand you clearly. Could you please repeat that?")
                                speak("I couldn't understand you clearly. Could you please repeat that?")
                            else:
                                cleaned_text = clean_transcription(full_text)
                                handle_conversation(cleaned_text)

                            buffer = np.zeros((0, 1), dtype=np.float32)  # reset

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
            backup_conversation()
if __name__ == "__main__":
    transcribe_stream()

