from googletrans import Translator

translator = Translator()

def detect_language(text):
    try:
        return translator.detect(text).lang
    except Exception as e:
        print(f"❌ Language Detection Error: {e}")
        return "en"

def translate_text(text, target_lang):
    try:
        if target_lang == "zh":
            target_lang = "zh-CN"

        translated = translator.translate(text, dest=target_lang)
        print(f"✅ Google Translate ({target_lang}): {translated.text}")  # Debugging
        return translated.text
    except Exception as e:
        print(f"❌ Translation Error: {e} - Failed text: {text}")
        return text  # Fallback to original text