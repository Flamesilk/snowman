# Chinese Language Support

This guide explains how to use the Chinese language support in the Simple Local Voice Assistant.

## Setup

1. Make sure you have installed all the required dependencies by running the setup script:
   ```
   python setup.py
   ```

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

## Running the Assistant in Chinese

To start the assistant with Chinese language support, use the `--language` flag:

```
python simple_local_assistant.py --language chinese
```

## Features

When running in Chinese mode:

1. **Speech Recognition**: The assistant uses OpenAI's Whisper (base model) for Chinese speech recognition, providing excellent accuracy.
2. **Text-to-Speech**: The assistant will use a Chinese voice from Microsoft Edge TTS.
3. **Conversation End Phrases**: You can end the conversation using Chinese phrases like "再见" (goodbye), "谢谢" (thank you), or "结束对话" (end conversation).

## How It Works

The speech recognition uses OpenAI's Whisper model:

- The base model (~140MB) is used by default for a good balance of accuracy and performance
- The model runs entirely locally on your device - no internet connection required for speech recognition
- If you have a GPU, the model will automatically use it for faster processing
- The same model is used for both English and Chinese, with language-specific optimizations

## Customizing the Chinese Voice

You can customize the Chinese voice by setting the `CHINESE_EDGE_TTS_VOICE` variable in your `.env` file:

```
CHINESE_EDGE_TTS_VOICE=zh-CN-YunxiNeural
```

Available Chinese voices include:
- `zh-CN-YunxiNeural` (male)
- `zh-CN-XiaoxiaoNeural` (female)
- `zh-CN-YunyangNeural` (male)
- `zh-CN-XiaochenNeural` (female)
- `zh-CN-YunjianNeural` (male)

## Limitations

- Wake word detection still uses English wake words (like "computer" or "alexa").
- The Gemini model works with Chinese text, but its responses may be more optimized for English.

## Troubleshooting

If you encounter issues with Chinese language support:

1. **Speech Recognition Issues**:
   - For better accuracy, try speaking clearly and in a quiet environment
   - If you need better accuracy, you can upgrade to a larger Whisper model by changing the `WHISPER_MODEL_SIZE` parameter in the code

2. **Text-to-Speech Issues**: If the Chinese voice doesn't sound right, try changing to a different voice using the `.env` file.

3. **Model Loading Errors**: The first time you run the assistant, the Whisper model will be downloaded automatically. This may take some time depending on your internet connection.

For any other issues, please refer to the main README file or open an issue on the project repository.
