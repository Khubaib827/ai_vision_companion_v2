import os
import tempfile
from gtts import gTTS

class VoiceAssistant:
    def __init__(self):
        pass

    def text_to_speech(self, text, lang='en'):
        if not text or not text.strip():
            return None
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(fp.name)
            return fp.name
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def generate_instructions(self, hazards, ocr_results):
        eng_instructions = []
        urdu_instructions = []
        
        if hazards:
            for hzd in hazards:
                cls = hzd["class"].capitalize()
                pos = hzd["position"]
                dist = hzd["distance"]
                eng_instructions.append(f"Warning! {cls} {pos}, approximately {dist} meters away.")
                
                urdu_map = {
                    "car": "گاڑی", "bus": "بس", "truck": "ٹرک", "person": "شخص",
                    "chair": "کرسی", "table": "میز", "door": "دروازہ", "bicycle": "سائیکل", "motorcycle": "موٹر سائیکل"
                }
                urdu_cls = urdu_map.get(hzd["class"], hzd["class"])
                urdu_instructions.append(f"خبردار! {urdu_cls} {pos}، فاصلہ تقریباً {dist} میٹر ہے۔")
        else:
            eng_instructions.append("Path ahead looks clear.")
            urdu_instructions.append("سامنے کا راستہ صاف ہے۔")
            
        if ocr_results:
            for txt in ocr_results[:2]:
                eng_instructions.append(f"Signboard text detected: {txt['text']}.")
                urdu_instructions.append(f"بورڈ پر لکھا ہے: {txt['text']}")
                
        full_eng = " ".join(eng_instructions)
        full_urdu = " ".join(urdu_instructions)
        return full_eng, full_urdu
