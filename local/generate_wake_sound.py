import asyncio
import edge_tts

async def list_voices():
    voices = await edge_tts.list_voices()
    print("\nAvailable English female voices:")
    print("=" * 50)
    for voice in voices:
        # Filter for English female voices
        if voice["Locale"].startswith("en-") and voice["Gender"] == "Female":
            name = voice["ShortName"]
            style_list = voice.get("StyleList", [])
            styles = f"Styles: {', '.join(style_list)}" if style_list else "No additional styles"
            print(f"Name: {name}")
            print(f"Locale: {voice['Locale']}")
            print(styles)
            print("-" * 30)

async def generate_greeting():
    # Use a young-sounding female voice
    voice = "en-US-JennyNeural"  # Jenny has a younger-sounding voice
    text = "Hi there!"

    # First list available voices
    await list_voices()

    print(f"\nGenerating greeting with voice: {voice}")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("sounds/wake.mp3")
    print("Greeting sound saved to sounds/wake.mp3")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_greeting())
