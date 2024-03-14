from transformers import pipeline, AutoTokenizer
import fasttext

class NERecognition:
    def __init__(self):
        self.model_en_us = pipeline("text2text-generation", model="cnmoro/t5-small-named-entity-recognition")
        self.model_pt_br = pipeline("text2text-generation", model="cnmoro/ptt5-small-named-entity-recognition")
        self.lang_detector = fasttext.load_model('lid.176.ftz')
        self.tokenizer = AutoTokenizer.from_pretrained("cnmoro/t5-small-named-entity-recognition")

    def recognize(self, text):
        all_parts_of_text = self._split_into_chunks_of_390_tokens(text)

        named_entities = []

        for parts in all_parts_of_text:
            language = self._detect_language(parts)
            if language == 'en':
                self.model = self.model_en_us
            else:
                self.model = self.model_pt_br

            result = self.model(parts, 
                                repetition_penalty = 1.5,
                                temperature = 1,
                                top_p = 0.9,
                                top_k = 50,
                                num_beams = 6,
                                do_sample = True,
                                max_new_tokens = 128,
                                )[0]["generated_text"]
            
            named_entities.extend(result.split(";"))

        named_entities = [entity.strip() for entity in named_entities if entity.strip() != ""]
        
        return list(set(named_entities))
            
    def _split_into_chunks_of_390_tokens(self, text):
        all_tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = []
        for i in range(0, len(all_tokens), 390):
            tokens.append(all_tokens[i:i+390])

        for i in range(len(tokens)):
            tokens[i] = self.tokenizer.decode(tokens[i], skip_specialtokens=True)
        
        return tokens
    
    def _detect_language(self, text):
        try:
            detected_lang = self.lang_detector.predict(text.replace('\n', ' '), k=1)[0][0]
            return 'en' if (str(detected_lang) == '__label__en' or str(detected_lang) == 'english') else 'pt'
        except:
            return 'pt'
    