import argparse
import json

def extract_hyp(input_file, output_file):
    """
    Extract the `hyp` field from each entry in a JSON file and write it to an output file.
    Escape characters will be rendered as-is.
    """
    try:
        # Read the input file
        with open(input_file, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        
        # Extract the `hyp` field and concatenate into a single text block
        hyp_list = [entry["hyp"] for entry in data if "hyp" in entry]
        result_text = "\n".join(hyp_list)

        # Write to the output file
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write(result_text)

        print(f"Successfully extracted {len(hyp_list)} 'hyp' entries.")
        print(f"Output saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {input_file}.")
    except KeyError as e:
        print(f"Error: Missing expected key in JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    # Use argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description="Extract 'hyp' fields from a JSON file and write them to an output file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output text file")
    
    args = parser.parse_args()

    # Call the processing function
    extract_hyp(args.input, args.output)

if __name__ == "__main__":
    main()