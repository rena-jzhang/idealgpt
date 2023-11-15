from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration



class Blip2Lavis():
    def __init__(self, name="blip2_opt", model_type="pretrain_opt6.7b", device="cuda"):
        self.model_type = model_type
        # self.blip2, self.blip2_vis_processors, _ = load_model_and_preprocess(
        #     name=name, model_type=model_type, is_eval=True, device=device, torch_dtype=torch.float16)

        self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", 
                                                                    torch_dtype=torch.float16, 
                                                                    device_map="auto")
        self.blip2.eval()
        self.device = device

    def ask(self, img_path, question, length_penalty=1.0, max_length=30):
        raw_image = Image.open(img_path).convert('RGB')
        inputs = self.blip2_processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
        out = self.blip2.generate(**inputs, max_length = max_length)
        answer = self.blip2_processor.decode(out[0], skip_special_tokens=True)

        # image = self.blip2_vis_processors(raw_image).unsqueeze(0).to(self.device, torch.float16)
        # if 't5' in self.model_type:
        #     answer = self.blip2.predict_answers({"image": image, "text_input": question}, length_penalty=length_penalty, max_length=max_length)
        # else:
        #     answer = self.blip2.generate({"image": image, "prompt": question}, length_penalty=length_penalty, max_length=max_length)
        # answer = answer[0]
        print("\nAnswer: " + answer)

        return answer

    def caption(self, img_path, prompt='a photo of'):
        # TODO: Multiple captions
        raw_image = Image.open(img_path).convert('RGB')
        # image = self.blip2_vis_processors(raw_image).unsqueeze(0).to(self.device, torch.float16)
        # caption = self.blip2.generate({"image": image})

        # caption = self.blip2.generate({"image": image, "prompt": prompt})
        inputs = self.blip2_processor(raw_image, prompt, return_tensors="pt").to("cuda", torch.float16)
        out = self.blip2.generate(**inputs)
        caption = self.blip2_processor.decode(out[0], skip_special_tokens=True)
        caption = caption.replace('\n', ' ').strip()  # trim caption
        print("\nCaption: " + caption)
        return caption
