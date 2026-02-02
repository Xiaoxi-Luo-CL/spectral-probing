'''Reads ZSRE dataset in JSON format and converts it to a binary classification CSV,
in order to enlarge the mkqa dataset.
Xiaoxi Luo, 2026/2/1'''
import json
import csv
import os


def process_zsre_to_csv(input_path, output_path):
    """
    Reads ZSRE dataset in JSON format and converts it to a binary classification CSV.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output CSV file.
    """

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    try:
        # Read JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        csv_rows = []

        # Process each entry in the dataset
        for entry in data:
            question = entry.get('src')

            # Extract answers
            # Assuming 'answers' is a list and taking the first one as ground truth (Label 1)
            true_answer = entry.get('answers')[
                0] if entry.get('answers') else ""

            # Assuming 'alt' is the incorrect/alternative answer (Label 0)
            false_answer = entry.get('alt')

            # Construct the positive sample (Label 1)
            if question and true_answer:
                csv_rows.append([question, true_answer, "1"])

            # Construct the negative sample (Label 0)
            if question and false_answer:
                csv_rows.append([question, false_answer, "0"])

        # split to train/dev/test set (80/10/10)
        test_index = int(0.8 * len(csv_rows))
        dev_index = int(0.9 * len(csv_rows))
        train_rows = csv_rows[:test_index]
        dev_rows = csv_rows[test_index:dev_index]
        test_rows = csv_rows[dev_index:]

        # quoting=csv.QUOTE_ALL ensures all fields are wrapped in double quotes as requested
        with open('zsre_train.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["text0", "text1", "label"])
            writer.writerows(train_rows)

        with open('zsre_test.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["text0", "text1", "label"])
            writer.writerows(test_rows)

        with open('zsre_dev.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["text0", "text1", "label"])
            writer.writerows(dev_rows)

        print(
            f"Successfully converted {len(data)} JSON entries into {len(csv_rows)} CSV rows.")
        print(f"Output saved to: {output_path}")

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the input file format.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Run conversion
process_zsre_to_csv("zsre_mend_eval.json", "zsre_qa.csv")
