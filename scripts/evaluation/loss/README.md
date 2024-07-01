# Evaluation by Loss & PPL on C4 dataset

Here is a simple framework for evaluating the compressed models by their losses & PPLs on the pretraining datasets.

For now, we provide a sample dataset file [c4_valid_part_of_0.json](..%2F..%2F..%2Fdata%2Fc4_valid_part_of_0.json) with only a part of the C4 validation data, which is enough to give an approximate estimation of the validation loss on C4.

For more accurate estimations, you should download the full validation data from [C4 dataset](https://huggingface.co/datasets/allenai/c4):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/c4-val*"
```

To convert the downloaded data into json format, you can use the following script:

```python
import gzip
import json
import glob

def extract_json_gz(file_path):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line}: {e}")
    return data

def main():
    file_pattern = './en/c4-validation*.json.gz' 
    json_gz_files = glob.glob(file_pattern)
    
    all_data = []
    for file_path in json_gz_files:
        data = extract_json_gz(file_path)
        all_data.extend(data)
    
    # Transform data to the required format
    transformed_data = [
        {
            "text": item.get("text", ""),
        }
        for item in all_data
    ]
    
    output_file = 'c4_valid_full.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)
    print(f"Saved transformed data to {output_file}")

if __name__ == "__main__":
    main()
```

Finally, you should move the converted `c4_valid_full.json` file to `./data`, and add the following lines to `./data/dataset_info.json`:

```json
"c4_valid_full": {
  "file_name": "c4_valid_full.json",
  "columns": {
    "prompt": "text"
  }
}
```
