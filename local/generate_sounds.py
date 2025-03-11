import asyncio
import edge_tts
import os

# Ensure sounds directory exists
os.makedirs("sounds", exist_ok=True)

# Define messages to generate
MESSAGES = {
    "goodbye": {
        "english": {
            "text": "Goodbye!",
            "voice": "en-US-JennyNeural",
            "file": "sounds/goodbye_en.mp3"
        },
        "chinese": {
            "text": "再见！",
            "voice": "zh-CN-XiaoxiaoNeural",
            "file": "sounds/goodbye_zh.mp3"
        }
    },
    "not_understood": {
        "english": {
            "text": "I didn't catch that.",
            "voice": "en-US-JennyNeural",
            "file": "sounds/not_understood_en.mp3"
        },
        "chinese": {
            "text": "我没听清楚。",
            "voice": "zh-CN-XiaoxiaoNeural",
            "file": "sounds/not_understood_zh.mp3"
        }
    },
    # "wake": {
    #     "english": {
    #         "text": "Hi there!",
    #         "voice": "en-US-JennyNeural",
    #         "file": "sounds/wake.mp3"
    #     }
    # }
}

async def list_voices():
    voices = await edge_tts.list_voices()
    print("\nAvailable voices:")
    print("=" * 50)
    for voice in voices:
        if voice["Locale"].startswith(("en-", "zh-")) and voice["Gender"] == "Female":
            name = voice["ShortName"]
            style_list = voice.get("StyleList", [])
            styles = f"Styles: {', '.join(style_list)}" if style_list else "No additional styles"
            print(f"Name: {name}")
            print(f"Locale: {voice['Locale']}")
            print(styles)
            print("-" * 30)

async def generate_audio_files():
    # First list available voices
    await list_voices()

    print("\nGenerating audio files...")
    for message_type, languages in MESSAGES.items():
        for lang, config in languages.items():
            try:
                print(f"\nGenerating {message_type} in {lang}")
                print(f"Text: {config['text']}")
                print(f"Voice: {config['voice']}")
                print(f"Output: {config['file']}")

                communicate = edge_tts.Communicate(config['text'], config['voice'])
                await communicate.save(config['file'])
                print(f"✅ Generated {config['file']}")
            except Exception as e:
                print(f"❌ Error generating {message_type} in {lang}: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_audio_files())
    print("\n✨ All audio files generated!")
