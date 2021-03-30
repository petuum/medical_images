pathologies = [
    'lung', 'normal', 'right', 'left', 'mild', 'base', 'bilateral',
    'thoracic vertebrae', 'opacity', 'degenerative', 'cardiomegaly',
    'calcinosis', 'pulmonary atelectasis', 'aorta', 'multiple',
    'hypoinflation', 'calcified granuloma', 'lower lobe', 'tortuous',
    'upper lobe', 'hilum', 'hyperdistention', 'cicatrix', 'spine',
    'small', 'interstitial', 'pleural effusion', 'markings',
    'diaphragm', 'aorta, thoracic', 'prominent', 'apex', 'middle lobe',
    'lymph nodes', 'airspace disease', 'atherosclerosis', 'density',
    'catheters, indwelling', 'deformity', 'scoliosis', 'nodule', 'round',
    'lingula', 'mediastinum', 'ribs', 'granulomatous disease',
    'surgical instruments', 'implanted medical device', 'patchy',
    'costophrenic angle', 'scattered', 'no indexing', 'severe', 'enlarged',
    'chronic', 'fractures, bone', 'bronchovascular', 'elevated',
    'technical quality of image unsatisfactory ', 'moderate', 'thorax',
    'blunted', 'pleura', 'streaky', 'posterior', 'pulmonary congestion',
    'osteophyte', 'infiltrate', 'lumbar vertebrae', 'spondylosis',
    'emphysema', 'focal', 'cardiac shadow', 'thickening', 'anterior',
    'large', 'abdomen', 'borderline', 'granuloma', 'flattened'
]

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
# Require to be specified
vocab_path = "/path/to/texar_vocab.txt"
img_root = "/path/to/iu_xray_images"
text_root_train = "/path/to/text_root_split/train"
text_root_val = "/path/to/text_root_split/val"
text_root_test = "/path/to/text_root_split/test"
visual_weights = None # or "/path/to/visual_weights"

dataset = {
            "mlc_lr": 1e-5,
            "lstm_lr": 5e-4,
            "train" : {
                "datasource":{
                    "img_root": img_root,
                    "text_root": text_root_train,
                    "vocab_path": vocab_path,
                    "transforms": transforms,
                    "pathologies": pathologies,
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
                    "img_root": img_root,
                    "text_root": text_root_val,
                    "vocab_path": vocab_path,
                    "transforms": transforms,
                    "pathologies": pathologies,
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
                    "img_root": img_root,
                    "text_root": text_root_test,
                    "vocab_path": vocab_path,
                    "transforms": transforms,
                    "pathologies": pathologies,
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
                "train_encoder": True,
            },
            # Configuration for the NLP modules and
            # the whole pipeline
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
                    "max_decoding_length": 60,
                },
                "lambda_stop": 1.,
                "lambda_word": 1.,
                "lambda_attn": 1.,
                "max_sent_num": 14,
                "visual_weights": visual_weights,
                "pathologies": pathologies,
                "vocab_path": vocab_path,
                "train_visual": True,
            }
        }
