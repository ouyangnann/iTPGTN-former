# Dataset Download
Please download the dataset from: 
https://github.com/liyaguang/DCRNN?tab=readme-ov-file#data-preparation
https://github.com/Davidham3/STSGCN

# Dataset Directory Structure

Once the datasets are downloaded, the project folder structure should look like the following:
```plaintext
├── METR-LA/                  
│   └── metr-la.h5
├── PEMS-BAY/                
│   └── pems-bay.h5
├── PEMS03/
│   ├── distance.csv           
│   └── pems03.npz
├── PEMS04/
│   ├── distance.csv           
│   └── pems04.npz
├── PEMS07/  
│   ├── distance.csv   
│   └── pems07.npz
├── PEMS08/  
│   ├── distance.csv   
│   └── pems08.npz
```

# Run Data Preparation
After downloading the datasets and ensuring the above file structure, you need to run the data preparation code to format the data for use in your model.