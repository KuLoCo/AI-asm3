import os
import argparse
import re

from src.predict import predict_file

def map_punctuation(text):
    stripped = text.strip()
    if re.fullmatch(r"\.+", stripped):
        if len(stripped) == 1:
            return ["no emotion"]
        else:
            return ["confusion", "disapproval"]
    if re.fullmatch(r"\?+", stripped):
        if len(stripped) == 1:
            return ["confusion", "curiosity"]
        else:
            return ["surprise", "confusion"]
    if re.fullmatch(r"!+", stripped):
        if len(stripped) == 1:
            return ["surprise"]
        else:
            return ["surprise", "excitement"]
    return None

def main():
    root = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Batch-predict emotions from a text file and save results."
    )
    parser.add_argument(
        '--model', '-m',
        default=os.path.join(root, "results", "logistic_regression.joblib"),
        help="Path to your trained model (.joblib)."
    )
    parser.add_argument(
        '--data-dir', '-d',
        default=os.path.join(root, "data", "goemotions", "data"),
        help="Path to GoEmotions data folder."
    )
    parser.add_argument(
        '--input-file', '-i',
        default=os.path.join(root, "User", "1.txt"),
        help="Path to text file with one sentence per line to analyze."
    )
    parser.add_argument(
        '--output-file', '-o',
        help="Path to write the results. Defaults to <input_file>_results.txt"
    )
    args = parser.parse_args()

    if args.output_file:
        out_path = args.output_file
    else:
        base, _ = os.path.splitext(args.input_file)
        out_path = base + "_results.txt"

    if not os.path.isfile(args.input_file):
        parser.error(f"Input file not found: {args.input_file}")
    if not os.path.isfile(args.model):
        parser.error(f"Model file not found: {args.model}")

    raw_results = predict_file(args.input_file, args.data_dir, args.model)

    with open(out_path, 'w', encoding='utf-8') as fout:
        for text, labels in raw_results:
            fout.write(f"Text: {text}\n")
            punct_labels = map_punctuation(text)
            if punct_labels is not None:
                fout.write("Emotional analysis: " + ", ".join(punct_labels) + "\n\n")
                continue

            if not labels:
                fout.write("Emotional analysis: no emotion\n\n")
            else:
                fout.write("Emotional analysis: " + ", ".join(labels) + "\n\n")

    print(f"Analysis complete. Results saved to:\n  {out_path}\n")
    for text, labels in raw_results:
        print(f"Text: {text}")
        punct_labels = map_punctuation(text)
        if punct_labels is not None:
            print("Emotional analysis:", ", ".join(punct_labels))
        elif not labels:
            print("Emotional analysis: no emotion")
        else:
            print("Emotional analysis:", ", ".join(labels))
        print()

if __name__ == '__main__':
    main()