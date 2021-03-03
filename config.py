from texar.torch.data import Vocab


pathologies = ['cardiac monitor', 'lymphatic diseases', 'pulmonary disease', 'osteophytes', 'foreign body',
               'dish', 'aorta, thoracic', 'atherosclerosis', 'histoplasmosis', 'hypoventilation',
               'catheterization, central venous', 'pleural effusions', 'pleural effusion', 'callus',
               'sternotomy', 'lymph nodes', 'tortuous aorta', 'stent', 'interstitial pulmonary edema',
               'cholecystectomies', 'neoplasm', 'central venous catheter', 'pneumothorax',
               'metastatic disease', 'vena cava, superior', 'cholecystectomy', 'scoliosis',
               'subcutaneous emphysema', 'thoracolumbar scoliosis', 'spinal osteophytosis',
               'pulmonary fibroses', 'rib fractures', 'sarcoidosis', 'eventration', 'fibrosis', 'spine',
               'obstructive lung disease', 'pneumonitis', 'osteopenia', 'air trapping', 'demineralization',
               'mass lesion', 'pulmonary hypertension', 'pleural diseases', 'pleural thickening',
               'calcifications of the aorta', 'calcinosis', 'cystic fibrosis', 'empyema', 'catheter',
               'lymph', 'pericardial effusion', 'lung cancer', 'rib fracture', 'granulomatous disease',
               'chronic obstructive pulmonary disease', 'rib', 'clip', 'aortic ectasia', 'shoulder',
               'scarring', 'scleroses', 'adenopathy', 'emphysemas', 'pneumonectomy', 'infection',
               'aspiration', 'bilateral pleural effusion', 'bulla', 'lumbar vertebrae', 'lung neoplasms',
               'lymphadenopathy', 'hyperexpansion', 'ectasia', 'bronchiectasis', 'nodule', 'pneumonia',
               'right-sided pleural effusion', 'osteoarthritis', 'thoracic spondylosis', 'picc',
               'cervical fusion', 'tracheostomies', 'fusion', 'thoracic vertebrae', 'catheters',
               'emphysema', 'trachea', 'surgery', 'cervical spine fusion', 'hypertension, pulmonary',
               'pneumoperitoneum', 'scar', 'atheroscleroses', 'aortic calcifications', 'volume overload',
               'right upper lobe pneumonia', 'apical granuloma', 'diaphragms', 'copd', 'kyphoses',
               'spinal fractures', 'fracture', 'clavicle', 'focal atelectasis', 'collapse',
               'thoracotomies', 'congestive heart failure', 'calcified lymph nodes', 'edema',
               'degenerative disc diseases', 'cervical vertebrae', 'diaphragm', 'humerus', 'heart failure',
               'normal', 'coronary artery bypass', 'pulmonary atelectasis', 'lung diseases, interstitial',
               'pulmonary disease, chronic obstructive', 'opacity', 'deformity', 'chronic disease',
               'pleura', 'aorta', 'tuberculoses', 'hiatal hernia', 'scolioses', 'pleural fluid',
               'malignancy', 'kyphosis', 'bronchiectases', 'congestion', 'discoid atelectasis', 'nipple',
               'bronchitis', 'pulmonary artery', 'cardiomegaly', 'thoracic aorta', 'arthritic changes',
               'pulmonary edema', 'vascular calcification', 'sclerotic', 'central venous catheters',
               'catheterization', 'hydropneumothorax', 'aortic valve', 'hyperinflation', 'prostheses',
               'pacemaker, artificial', 'bypass grafts', 'pulmonary fibrosis', 'multiple myeloma',
               'postoperative period', 'cabg', 'right lower lobe pneumonia', 'granuloma',
               'degenerative change', 'atelectasis', 'inflammation', 'effusion', 'cicatrix',
               'tracheostomy', 'aortic diseases', 'sarcoidoses', 'granulomas', 'interstitial lung disease',
               'infiltrates', 'displaced fractures', 'chronic lung disease', 'picc line',
               'intubation, gastrointestinal', 'lung diseases', 'multiple pulmonary nodules',
               'intervertebral disc degeneration', 'pulmonary emphysema', 'spine curvature', 'fibroses',
               'chronic granulomatous disease', 'degenerative disease', 'atelectases', 'ribs',
               'pulmonary arterial hypertension', 'edemas', 'pectus excavatum', 'lung granuloma',
               'plate-like atelectasis', 'enlarged heart', 'hilar calcification', 'heart valve prosthesis',
               'tuberculosis', 'old injury', 'patchy atelectasis', 'histoplasmoses', 'exostoses',
               'mastectomies', 'right atrium', 'large hiatal hernia', 'hernia, hiatal', 'aortic aneurysm',
               'lobectomy', 'spinal fusion', 'spondylosis', 'ascending aorta', 'granulomatous infection',
               'fractures, bone', 'calcified granuloma', 'degenerative joint disease',
               'intubation, intratracheal', 'others']

transforms = [
    ("Resize", {
        "size": 256,
        "interpolation": 1
    }),
    ("CenterCrop", {
        "size": 224
    }),
    ("ToTensor", {}),
    ("Normalize", {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    })
]

HIDDEN_SIZE = 512
vocab_path = "./texar_vocab.txt"
vocab = Vocab(vocab_path)

dataset = {
            "img_root" : "/data/jiachen.li/iu_xray/images/images_normalized",
            "text_root" : "/home/jiachen.li/text_root",
            "vocab_path" : vocab_path,
            "csvpath" : "/data/jiachen.li/iu_xray/indiana_reports.csv",
            "metacsvpath" : None,
            "pathologies" : pathologies,
            "seed": 0,
            "mlc_lr": 1e-5,
            "lstm_lr": 5e-4,
            "train" : {
                "datasource":{
                    "img_root": "/home/jiachen.li/iu_xray_images",
                    "label_path": "./mlc_data/train_data.txt",
                    "text_root": "/home/jiachen.li/text_root",
                    "vocab_path": vocab_path,
                    "transforms": transforms,
                },
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "num_parallel_calls": 1
            },
            "val" : {
                "datasource":{
                    "img_root": "/home/jiachen.li/iu_xray_images",
                    "label_path": "./mlc_data/val_data.txt",
                    "text_root": "/home/jiachen.li/text_root",
                    "vocab_path": vocab_path,
                    "transforms": transforms,
                },
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "num_parallel_calls": 1,

            },
            "test": {
                "datasource":{
                    "img_root": "/home/jiachen.li/iu_xray_images",
                    "label_path": "./mlc_data/val_data.txt",
                    "text_root": "/home/jiachen.li/text_root",
                    "vocab_path": vocab_path,
                    "transforms": transforms,
                },
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "num_parallel_calls": 1,
            },
            "mlc_trainer":{
                "num_tags": len(pathologies),
                "train_encoder": False,
            },
            "model": {
                "mlc": {
                    "num_tags": len(pathologies),
                    "fc_in_features": 1024,
                },
                "tag_generator": {
                    'hidden_size': HIDDEN_SIZE,
                    "num_tags": len(pathologies),
                    "top_k_for_semantic": 10,
                },
                "sentence_lstm": {
                    "hidden_size": HIDDEN_SIZE,
                    "visual_dim": 1024,
                },
                "word_lstm":{
                    "hidden_size": HIDDEN_SIZE,
                    "vocab_size": vocab.size,
                    "max_decoding_length": 60,
                    "BOS": vocab.bos_token_id,
                    "EOS": vocab.eos_token_id,
                },
                "lambda_stop": 1.,
                "lambda_word": 1.,
                "lambda_attn": 1.,
                "max_sent_num": 14,
                "visual_weights": "exp_default_mlc/1614486844.6354058.pt",
                "pathologies": pathologies,
                "vocab_path": vocab_path,
                "train_visual": False,
            }
        }
