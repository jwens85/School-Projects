from kaggle.api.kaggle_api_extended import KaggleApi

print("Script is running...")

api = KaggleApi()
api.authenticate()

dataset = "hamzamanssor/car-damage-assessment"
destination = "./data"

api.dataset_download_files(dataset, path=destination, unzip=True)
print(f"Dataset downloaded and unzipped to {destination}")
