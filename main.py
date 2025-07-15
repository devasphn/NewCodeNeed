import torch
import transformers
import faster_whisper
from TTS.api import TTS
import gradio as gr
import numpy as np
import webrtcvad
import collections
import time
import os
from threading import Thread

# --- 1. Global State and Configuration ---
class AgentState:
    def __init__(self):
        self.agent_running = False
        self.agent_muted = False
        self.transcription_queue = collections.deque()
        self.response_queue = collections.deque()

state = AgentState()

# --- 2. Core AI Models Initialization ---
class AIModels:
    def __init__(self):
        print("--- Initializing AI Models ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using device: {self.device.upper()}")

        # STT Model (changed to large-v2 as requested)
        stt_model_name = "large-v2"
        print(f"Loading STT model: faster-whisper-{stt_model_name}...")
        self.stt_model = faster_whisper.WhisperModel(
            stt_model_name,
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8"
        )
        print("STT model loaded.")

        # LLM Model
        llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"Loading LLM: {llm_model_name}...")
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model=llm_model_name,
            model_kwargs={"torch_dtype": self.torch_dtype, "load_in_4bit": True},
            device_map=self.device,
        )
        print("LLM loaded.")

        # TTS Model
        tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print(f"Loading TTS model: {tts_model_name}...")
        self.tts_model = TTS(tts_model_name).to(self.device)
        print("TTS model loaded.")
        print("\n--- All Models Initialized ---")

models = AIModels()

# --- 3. Audio Processing and VAD ---
class AudioProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.chunk_duration_ms = 30
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.padding_duration_ms = 500
        self.num_padding_chunks = int(self.padding_duration_ms / self.chunk_duration_ms)
        self.ring_buffer = collections.deque(maxlen=self.num_padding_chunks)
        self.voiced_frames = []
        self.triggered = False

    def process_audio_chunk(self, audio_chunk):
        # The audio from gr.Audio is float32, convert to int16 for VAD
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        is_speech = self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)

        if not self.triggered:
            self.ring_buffer.append((audio_chunk, is_speech))
            num_voiced = sum(1 for _, speech in self.ring_buffer if speech)
            if num_voiced > 0.8 * self.ring_buffer.maxlen:
                self.triggered = True
                print("Voice detected, recording...")
                self.voiced_frames.extend([f for f, _ in self.ring_buffer])
                self.ring_buffer.clear()
        else:
            self.voiced_frames.append(audio_chunk)
            self.ring_buffer.append((audio_chunk, is_speech))
            num_unvoiced = sum(1 for _, speech in self.ring_buffer if not speech)
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                print("End of speech detected.")
                audio_data = np.concatenate(self.voiced_frames)
                self.reset()
                return audio_data
        return None

    def reset(self):
        self.triggered = False
        self.ring_buffer.clear()
        self.voiced_frames = []

audio_processor = AudioProcessor()

# --- 4. Core Agent Logic ---
def transcribe_audio(audio_data):
    if audio_data is None or len(audio_data) == 0:
        return ""
    print("Transcribing...")
    segments, _ = models.stt_model.transcribe(audio_data, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    print(f"User: {transcription}")
    return transcription

def generate_response(text):
    print("Generating response...")
    messages = [
        {"role": "system", "content": "You are a friendly and helpful conversational AI. Keep your responses concise and to the point."},
        {"role": "user", "content": text},
    ]
    terminators = [
        models.llm_pipeline.tokenizer.eos_token_id,
        models.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = models.llm_pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    assistant_response = outputs[0]["generated_text"][-1]['content']
    print(f"Agent: {assistant_response}")
    return assistant_response

def text_to_speech(text):
    output_file = "output.wav"
    print("Speaking...")
    models.tts_model.tts_to_file(
        text=text,
        speaker=models.tts_model.speakers[0],
        language=models.tts_model.languages[0],
        file_path=output_file
    )
    return output_file

def agent_pipeline(stream, new_chunk):
    global audio_processor
    sr, y = new_chunk
    if y is None:
        return stream, "", "", None

    # Resample to 16kHz if necessary
    if sr != 16000:
        y = gr.processing_utils.resample(y, sr, 16000)
    
    y = y.astype(np.float32) / 32767.0 # Normalize

    if not state.agent_muted:
        audio_segment = audio_processor.process_audio_chunk(y)
        if audio_segment is not None:
            # Add tasks to queues instead of blocking
            state.transcription_queue.append(audio_segment)

    # This function is called rapidly, so we just return the current state
    # The actual processing happens in the background thread
    return stream, "Listening...", "Thinking...", None

def background_processing_loop():
    while True:
        if state.agent_running:
            if state.transcription_queue:
                audio_segment = state.transcription_queue.popleft()
                user_text = transcribe_audio(audio_segment)
                if user_text:
                    llm_response = generate_response(user_text)
                    state.response_queue.append((user_text, llm_response))
            
            # Prevent busy-waiting
            time.sleep(0.1)
        else:
            # Sleep longer when the agent is not active
            time.sleep(1)


# --- 5. Gradio UI ---
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="Real-Time AI Agent") as app:
        gr.Markdown("# Real-Time Conversational AI Agent")
        gr.Markdown("Click 'Start Agent' to begin. Speak into your microphone. The agent will listen, think, and respond.")

        with gr.Row():
            start_button = gr.Button("Start Agent", variant="primary")
            stop_button = gr.Button("Stop Agent")
            mute_button = gr.Button("Mute")

        with gr.Row():
            with gr.Column():
                gr.Label("Your Speech (Transcription)")
                user_text_output = gr.Textbox(lines=5, interactive=False)
            with gr.Column():
                gr.Label("Agent's Response")
                agent_text_output = gr.Textbox(lines=5, interactive=False)
        
        with gr.Row():
            agent_audio_output = gr.Audio(label="Agent's Voice", autoplay=True, interactive=False)

        # Hidden component for streaming audio input
        audio_stream = gr.Audio(sources=["microphone"], streaming=True, visible=False)

        # --- UI Control Logic ---
        def start_agent():
            state.agent_running = True
            state.agent_muted = False
            audio_processor.reset()
            return {
                start_button: gr.Button(interactive=False),
                stop_button: gr.Button(interactive=True),
                mute_button: gr.Button("Mute", interactive=True),
                user_text_output: "",
                agent_text_output: "Agent started. Listening...",
            }

        def stop_agent():
            state.agent_running = False
            audio_processor.reset()
            return {
                start_button: gr.Button(interactive=True),
                stop_button: gr.Button(interactive=False),
                mute_button: gr.Button("Mute", interactive=False),
                agent_text_output: "Agent stopped.",
            }

        def toggle_mute():
            state.agent_muted = not state.agent_muted
            mute_text = "Unmute" if state.agent_muted else "Mute"
            return gr.Button(mute_text)
        
        def update_ui_components():
            """This function will be called periodically to update the UI."""
            if state.response_queue:
                user_text, agent_text = state.response_queue.popleft()
                audio_file = text_to_speech(agent_text)
                return user_text, agent_text, audio_file
            return gr.Skip(), gr.Skip(), gr.Skip()

        # --- Event Listeners ---
        start_button.click(start_agent, outputs=[start_button, stop_button, mute_button, user_text_output, agent_text_output])
        stop_button.click(stop_agent, outputs=[start_button, stop_button, mute_button, agent_text_output])
        mute_button.click(toggle_mute, outputs=mute_button)
        
        # This is the core listener for the audio stream
        audio_stream.stream(agent_pipeline, inputs=[audio_stream, audio_stream], outputs=[audio_stream, user_text_output, agent_text_output, agent_audio_output], show_progress="hidden")

        # This dependency will poll the backend for completed responses
        app.load(update_ui_components, None, [user_text_output, agent_text_output, agent_audio_output], every=0.5)

    return app

if __name__ == "__main__":
    # Start the background processing thread
    processing_thread = Thread(target=background_processing_loop, daemon=True)
    processing_thread.start()

    # Launch the Gradio UI
    web_ui = create_ui()
    web_ui.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    # Clean up the output file on exit
    if os.path.exists("output.wav"):
        os.remove("output.wav")
