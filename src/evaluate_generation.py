import argparse
import pandas as pd
import numpy as np

import evaluate

import openai
import json

GPT_EVAL_DESCRIPTION = """#Evaluation Task for Code-Switched Sentence Generation
You are provided with pairs of sentences. The first sentence in each pair is the original monolingual sentence. The second sentence is a generated code-switched sentence. Your task is to evaluate the generated sentence based on two criteria: Accuracy and Fluency. You will score each criterion on a scale from 1 to 3, where 1 is the lowest and 3 is the highest.
When evaluating the generated sentences, focus on the content and meaning. Ignore any extra formatting, alignment artifacts, or additional explanatory text. Judge the sentence to determine its accuracy and fluency.

#Evaluation Criteria
##Accuracy
Accuracy measures how well the generated sentence preserves the meaning and information of the original sentence and whether the code-switched terms are used correctly.
Scores:
1. Low Accuracy
Significant deviations from the original meaning.
Key information is missing or altered.
Code-switched terms are incorrect or inappropriate.
2. Moderate Accuracy
Minor deviations from the original meaning.
Most key information is present but may have slight errors.
Most code-switched terms are appropriate but with minor mistakes.
3. High Accuracy
Preserves the original meaning fully.
All key information is present and correct.
Code-switched terms are accurate and appropriately used.
##Fluency
Fluency measures how natural and easy to understand the generated sentence is, considering grammar, syntax, and the smooth integration of code-switching.
Scores:
1. Low Fluency
The sentence is difficult to understand or awkward.
Poor grammar or syntax in either language.
Code-switching disrupts the flow of the sentence.
2. Moderate Fluency
The sentence is understandable but may have awkward or unnatural phrasing.
Acceptable grammar and syntax in both languages.
Code-switching is somewhat smooth but not perfectly integrated.
3. High Fluency
The sentence is natural and easy to understand.
Good grammar and syntax in both languages.
Code-switching is smooth and seamless, enhancing the sentence flow.
"""

class GPTEvaluator:
    def __init__(self, openai_key, openai_org):
        self.openai_key = openai_key
        self.openai_org = openai_org
        self.openai = openai.OpenAI(self.openai_key, self.openai_org)
        example_input_1 = """{
            "original_l1": "Enjoy 0% instalment for up to 12 months when ordering an Air Asia plane ticket with BNI Credit Card!",
            "original_l2": "Nikmati cicilan 0% hingga 12 bulan untuk pemesanan tiket pesawat Air Asia dengan kartu kredit BNI!",
            "generated": "Enjoy 0% cicilan for up to 12 bulan with Air Asia's tiket pemesanan hingga 12 bulan, and get ready to soar with BNI kartu kredit, dengan 0% interest, untuk your next adventure!"
        }
        """
        example_output_1 = """{
            "accuracy": 1,
            "fluency": 1
        }
        """
        example_input_2 = """{
            "original_l1": "Enjoy 0% instalment for up to 12 months when ordering an Air Asia plane ticket with BNI Credit Card!",
            "original_l2": "Nikmati cicilan 0% hingga 12 bulan untuk pemesanan tiket pesawat Air Asia dengan kartu kredit BNI!",
            "generated": "Enjoy 0% instalment untuk 12 bulan when ordering an Air Asia tiket pesawat dengan BNI Credit Card!"
        }
        """
        example_output_2 = """{
            "accuracy": 3,
            "fluency": 2
        }
        """
        self.chat_template = [
            {"role": "system", "content": GPT_EVAL_DESCRIPTION},
            {"role": "user", "content": example_input_2},
            {"role": "assistant", "content": example_output_2},
            {"role": "user", "content": example_input_1},
            {"role": "assistant", "content": example_output_1},
        ]
    def transform(self, df):
        list_of_dict = [
            {
                "original_l1": row["src"],
                "original_l2": row["tgt"],
                "generated": row["generated"]
            } for i, row in df.iterrows()
        ]

        list_of_chat = [   
            self.chat_template + [
                {"role": "user", "content": json.dumps(l)}
            ] for l in list_of_dict
        ]
        return list_of_chat
    
    def preprocess_batch(self, input_list):
        batch_format = []
        for custom_id, mes in enumerate(input_list):
            batch_format.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions", 
                    "body": {
                        "model": "gpt-4o-mini", 
                        "messages": mes,
                        "max_tokens": 20
                    }
                }
            )
        return batch_format
    
    def post_process(self, output):
        json_output = []
        for i in range(0, output):
            json_output.append(json.loads(output[i]))
        return json_output

    def evaluate(self, list_of_sentences):
        list_of_chat = self.transform(list_of_sentences)
        output = []
        for chat in list_of_chat:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=chat,
                max_tokens=20
            )
            output.append(response.choices[0].message.content)
        return self.post_process(output)
    
    def evaluate_batch(self, list_of_sentences):
        list_of_chat = self.transform(list_of_sentences)
        batch_list = self.preprocess_batch(list_of_chat)
        with open('/tmp/batch_input.jsonl', 'w+') as f:
            for chat in batch_list:
                f.write(json.dumps(chat) + '\n')
        
        batch_input_file = self.openai.files.create(
            file=open("/tmp/batch_input.jsonl", "rb"),
            purpose="batch"
        )
        response = self.openai.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Evaluation of generations"
            }
        )

        return response
    
def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--generation_file', type=str)
    arg_parser.add_argument('--lang', type=str)
    arg_parser.add_argument('--output_file', type=str)
    
    arg_parser.add_argument('--openai_key', type=str)
    arg_parser.add_argument('--openai_org', type=str)
    arg_parser.add_argument('--batch', action='store_true')        
    args = arg_parser.parse_args()
    return args

def main(args):
    print(f"Evaluating {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    assert 'src' in df.columns
    assert 'tgt' in df.columns
    assert 'generated' in df.columns

    comet = evaluate.load('comet')
    comet_1 = comet.compute(predictions=df['generated'], sources=df['src'], references=df['tgt'])
    comet_2 = comet.compute(predictions=df['generated'], sources=df['tgt'], references=df['src'])
    df['comet'] = (np.array(comet_1['scores']) + np.array(comet_2['scores'])) / 2
    bertscore = evaluate.load('bertscore')
    bertscore_1 = bertscore.compute(predictions=df['generated'], references=df['src'], lang=args.lang)
    bertscore_2 = bertscore.compute(predictions=df['generated'], references=df['tgt'], lang=args.lang)
    df['bertscore'] = (np.array(bertscore_1['f1']) + np.array(bertscore_2['f1'])) / 2

    evaluator = GPTEvaluator(args.openai_key, args.openai_org)


    if args.batch:
        print("Submitting batch evaluation request...")
        response = evaluator.evaluate_batch(df)
        print(response)
    else:
        print("Evaluating one by one...")
        output = evaluator.evaluate(df)
        df['GPT_a'] = [o['accuracy'] for o in output]
        df['GPT_f'] = [o['fluency'] for o in output]

    df.to_csv(args.output_file, index=False)
    print(f"Output saved to {args.output_file}")

        
if __name__ == "__main__":
    args = init()
    main(args)
