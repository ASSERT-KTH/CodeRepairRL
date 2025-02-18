import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Literal

import gdown
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class PrimeVul(Dataset):
    """PrimeVul dataset for vulnerability detection.
    
    Downloads and processes the PrimeVul dataset from Google Drive.
    """
    
    URL = "https://drive.google.com/drive/folders/19iLaNDS0z99N8kB_jBRTmDLehwZBolMY"
    SPLITS = ("train", "valid", "test")
    TYPES = ("paired", "unpaired")
    
    def __init__(self, split:Literal["train", "valid", "test"], dataset_type:Literal["paired", "unpaired"]="paired", root:str="data"):
        """Initialize PrimeVul dataset.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            dataset_type: Type of dataset ('paired', 'unpaired'), 'paired' being adjacent vuln/fix pairs
            root: Root directory for data
        """
        self.root = Path(root) / "PrimeVul"
        self.split = split
        self.type = dataset_type
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        
        if not self._check_exists():
            self.prepare_data()
            
        self.data = self.load()
        
    def _check_exists(self) -> bool:
        return (self.processed_dir / f"{self.split}_{self.type}.json").exists()
    
    # TODO: There is also a file_contents directory which contains the entire files instead of just the vulnerable snippets.
    #       We would need to create a mapping from hash to file since we need to search through all the folders for the file
    #       with the same hash.
    def prepare_data(self):
        """Download, extract and process the dataset."""
        # Download if needed
        if not self.raw_dir.exists():
            logger.info("Downloading PrimeVul dataset...")
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            gdown.download_folder(PrimeVul.URL, output=str(self.raw_dir), quiet=False)
        
        # Process all splits
        logger.info("Processing dataset...")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        for split in PrimeVul.SPLITS:
            for type in PrimeVul.TYPES:
                file_name = f"primevul_{split}" + (f"_{type}.jsonl" if type == 'paired' else ".jsonl")
                
                data = [json.loads(line) for line in open(self.raw_dir / file_name)]
                
                # Standardize CWE labels
                data = [x if "cwe" in x else x | {"cwe": "Other"} for x in data]
                data = [x | {"cwe": "Other"} if x["cwe"] == "" else x for x in data]
                data = [x | {"cwe": "Other"} if isinstance(x["cwe"], list) and len(x["cwe"]) == 0 else x for x in data]
                data = [x | {"cwe": x["cwe"][0]} if isinstance(x["cwe"], list) and len(x["cwe"]) == 1 else x for x in data]
                data = [x | {"cwe": "Other"} if isinstance(x["cwe"], str) and not x["cwe"].startswith("CWE-") else x for x in data]
                data = [x | {"cwe": [x["cwe"]]} if isinstance(x["cwe"], str) else x for x in data]
                
                # Save processed data
                json.dump(data, open(self.processed_dir / f"{split}_{type}.json", "w"), indent=4)
            
    def load(self) -> List[Dict[str, Any]]:
        """Load the processed data for the current split."""
        file_path = self.processed_dir / f"{self.split}_{self.type}.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed file {file_path} not found. "
                "This should not happen as _check_exists should have caught this."
            )
        return json.load(open(file_path))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]  # maybe just return a tuple (func, is_vulnerable) and keep that format the same for the other datasets
    
    @property
    def num_classes(self) -> int:
        return len(set(cwe for item in self.data for cwe in item["cwe"]))
    
    @property
    def classes(self) -> List[str]:
        return sorted(list(set(cwe for item in self.data for cwe in item["cwe"])))
    

if __name__ == "__main__":
    dataset = PrimeVul("train")
    print(dataset.num_classes)
    print(dataset.classes)

    if input("Convert to HF dataset? (y/n)") == "n": exit()

    from datasets import Dataset as HFDataset, DatasetDict

    def convert(data):
        return [{
            "is_vulnerable": x["target"] == 0,
            "func": x["func"],
            "cwe": x["cwe"],
            "project": x["project"],
            "commit_id": str(x["commit_id"]),
            "hash": str(x["hash"]),
            "big_vul_idx": x.get("big_vul_idx", None),  # sometimes not present
            "idx": x.get("idx", None),
        } for x in data]

    hfds = DatasetDict({
        "train_paired": HFDataset.from_list(convert(PrimeVul("train", "paired").data)),
        "valid_paired": HFDataset.from_list(convert(PrimeVul("valid", "paired").data)),
        "test_paired": HFDataset.from_list(convert(PrimeVul("test", "paired").data)),
        "train_unpaired": HFDataset.from_list(convert(PrimeVul("train", "unpaired").data)),
        "valid_unpaired": HFDataset.from_list(convert(PrimeVul("valid", "unpaired").data)),
        "test_unpaired": HFDataset.from_list(convert(PrimeVul("test", "unpaired").data)),
    })

    hfds.push_to_hub("ASSERT-KTH/Primevul")
