dataset = {
            "imgpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
            "csvpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
            "metacsvpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv",
            "mode" : "PER_IMAGE",
            "pathologies" : [
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Lung Opacity",
                "Lung Lesion",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
                "Pleural Other",
                "Fracture",
                "Support Devices"],
            "views": ["PA"],
            "seed": 0,
            "train" : {
                "processed_csv": "/home/jiachen.li/medical_images/preprocessed/mimic_train.csv",
                "transforms": [
                    # ("ToPILImage", {}),
                    ("RandomAffine", {
                        "degrees": (-5, 5),
                        "shear": (0.9, 1.1)
                    }),
                    ("RandomResizedCrop", {
                        "size": (256, 256),
                        "scale": (0.5, 0.75),
                        "ratio": (0.95, 1.05),
                        "interpolation": 1
                    }),
                    ("ToTensor", {}),
                    ("Normalize", {
                        "mean": (0.485, 0.456, 0.406),
                        "std": (0.229, 0.224, 0.225)
                    })
                ],
                "mode": "PER_IMAGE",
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "input_channel": "RGB",
                "num_parallel_calls": 1

            },
            "val" : {
                "processed_csv": "/home/jiachen.li/medical_images/preprocessed/mimic_val.csv",
                "transforms":[
                    # ("ToPILImage", {}),
                    ("Resize", {
                        "size": 256,
                        "interpolation": 1
                    }),
                    ("CenterCrop", {
                        "size": 256
                    }),
                    ("ToTensor", {}),
                    ("Normalize", {
                        "mean": (0.485, 0.456, 0.406),
                        "std": (0.229, 0.224, 0.225)
                    })
                ],
                "mode": "PER_IMAGE",
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "input_channel": "RGB",
                "num_parallel_calls": 1,

            },
            "test": {
                "processed_csv": "/home/jiachen.li/medical_images/preprocessed/mimic_test.csv",
                "transforms": [
                    # ("ToPILImage", {}),
                    ("Resize", {
                        "size": 256,
                        "interpolation": 1
                    }),
                    ("CenterCrop", {
                        "size": 256
                    }),
                    ("ToTensor", {}),
                    ("Normalize", {
                        "mean": (0.485, 0.456, 0.406),
                        "std": (0.229, 0.224, 0.225)
                    })
                ],
                "mode": "PER_IMAGE",
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "input_channel": "RGB",
                "num_parallel_calls": 1,
            }
        }
