import argparse

from transformers import pipeline
import torch
from tqdm import tqdm

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--src_lang', type=str)
    arg_parser.add_argument('--tgt_lang', type=str)
    arg_parser.add_argument('--input', type=str)
    arg_parser.add_argument('--output', type=str)
    
    args = arg_parser.parse_args()
    return args

class Translator:
    def __init__(self, source_language, target_language):
        self.source_language = source_language
        self.target_language = target_language
        self.LANGMAP = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'ml': 'Malayalam',
            'id': 'Indonesian'
        }
        self.EXAMPLEMAP = {
            'en': 'The reward of goodness shall be nothing but goodness',
            'hi': 'अच्छाई का पुरस्कार अच्छाई के अलावा कुछ नहीं होग',
            'ml': 'നന്മയുടെ പ്രതിഫലം നന്മയല്ലാതെ മറ്റൊന്നുമല്ലा',
            'ta': 'நன்மையின் வெகுமதி நன்மையைத் தவிர வேறில்லை',
            'id': 'Pahala kebaikan tak lain hanyalah kebaikan'
        }
        # self.timeout = timeout
        self.model = pipeline('text-generation', 
            'meta-llama/Meta-Llama-3-8B-Instruct',
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
    def transform(self, list_of_sentences):
        list_of_chat = [   
            [
                {"role": "system", "content": f"Translate the following to {self.LANGMAP[self.target_language]}: "},
                {"role": "user", "content": self.EXAMPLEMAP[self.source_language] },
                {"role": "assistant", "content": self.EXAMPLEMAP[self.target_language] },
                {"role": "user", "content": l.strip() }
            ] for l in list_of_sentences
        ]
        return list_of_chat
    
    def translate(self, list_of_sentences):
        chat_template = self.transform(list_of_sentences)
        output = self.model(chat_template)
        clean_output = []
        for i in range(len(output)):
            clean_output.append(output[i][0]['generated_text'][len(list_of_sentences[i]):].split('\n')[0])
        return clean_output

def main(args):
    translator = Translator(
        source_language=args.src,
        target_language=args.tgt,
    )
    
    print(f"reading input file {args.input}")
    # read input file
    with open(args.input) as f:
        raw_lines = [line.strip() for line in f.readlines()]
    
    print(f"translating from {args.src} to {args.tgt}")
    translated_lines = translator.translate(raw_lines)
    
    print(f"writing output file {args.output}")
    with open(args.output, 'w+') as f:
        for line in translated_lines:
            f.write(f"{line}\n")
            
if __name__ == "__main__":
    args = init()
    
    main(args)