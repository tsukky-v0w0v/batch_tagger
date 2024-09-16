import pandas as pd
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as rt

from models import models
from utils import setup_logger
logger=setup_logger("tagger")


MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


class Predictor: 
    def __init__(self, args):
        # モデル設定
        self.name = args.model
        self.repo_id = models[self.name]["repo_id"]

        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.cpu:
            self.providers = ["CPUExecutionProvider"]

        # 閾値設定
        self.general_threshold = args.general_threshold
        self.character_threshold = args.character_threshold
        if args.use_recommended_threshold:
            self.general_threshold = models[self.name]["threshold"]
        logger.info(f"general threshold: {self.general_threshold}")
        logger.info(f"character threshold: {self.character_threshold}")

        # タグ設定
        self.use_rating = args.use_rating == True
        self.use_character = args.use_character == True
        self.use_general = args.use_general == True

        self.additional_tag = []
        if args.additional_tag:
            if isinstance(args.additional_tag, str):
                self.additional_tag = [tag.strip() for tag in args.additional_tag.split(",")]
            if isinstance(args.additional_tag, list):
                self.additional_tag = [tag.strip() for tags in args.additional_tag for tag in tags.split(",")]
        logger.info(f"additional_tag: {self.additional_tag}")
        
        self.exclude_tag = []
        if args.exclude_tag:
            if isinstance(args.exclude_tag, str):
                self.exclude_tag = [tag.strip() for tag in args.exclude_tag.split(",")]
            if isinstance(args.exclude_tag, list):
                self.exclude_tag = [tag.strip() for tags in args.exclude_tag for tag in tags.split(",")]
        logger.info(f"exclude_tag: {self.exclude_tag}")
        
        self.all_sort = args.all_sort == True
        self.weighted_captions = args.weighted_captions == True

        self.load()
    

    def download(self):
        csv_path = hf_hub_download(self.repo_id, LABEL_FILENAME)
        model_path = hf_hub_download(self.repo_id, MODEL_FILENAME)
        return csv_path, model_path
    

    def load_labels(self, dataframe):
        name_series = dataframe["name"]
        name_series = name_series.map(lambda x: x.replace("_", " ") if x not in kaomojis else x)
        tags_names = name_series.tolist()

        rating_indexes = list(np.where(dataframe["category"] == 9)[0])
        general_indexes = list(np.where(dataframe["category"] == 0)[0])
        character_indexes = list(np.where(dataframe["category"] == 4)[0])
        return tags_names, rating_indexes, general_indexes, character_indexes


    def load(self):
        logger.info(f"Loading {self.name} model")
        
        csv_path, model_path = self.download()
        self.model = rt.InferenceSession(model_path, providers=self.providers)
        
        tags_df = pd.read_csv(csv_path)
        sep_tags = self.load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]
        
        logger.info(f"Loaded {self.name} model")
        
    
    def prepare_image(self, images: list):
        _, target_size, _, _ = self.model.get_inputs()[0].shape

        # convert images to fit the model
        processed_images = []
        for image_path in images:
            # alpha to white
            image = Image.open(image_path)
            image = image.convert("RGBA")
            canvas = Image.new("RGBA", image.size, "WHITE")
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")

            # to square
            image_shape = image.size
            max_dim = max(image_shape)
            pad_left = (max_dim - image_shape[0]) // 2
            pad_top = (max_dim - image_shape[1]) // 2
            padded_image = Image.new("RGB", (max_dim, max_dim), "WHITE")
            padded_image.paste(image, (pad_left, pad_top))

            # resize
            if max_dim != target_size:
                padded_image = padded_image.resize(
                    (target_size, target_size), 
                    Image.BICUBIC,
                )
            
            image_array = np.asarray(padded_image, dtype=np.float32)
            image_array = image_array[:, :, ::-1] # PIL RGB to OpenCV GBR

            processed_images.append(image_array)
        
        return np.stack(processed_images, axis=0)


    def predict(self, images: list):
        # 画像をバッチとして準備
        images = self.prepare_image(images)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: images})[0]

        return preds
    

    def postprocess_tags(self, preds):
        tags_list = []

        def filter_and_sort(labels, threshold, exclude_tag):
            filtered_labels = [x for x in labels if (x[1] > threshold) and (x[0] not in exclude_tag)]
            sorted_labels = sorted(filtered_labels, key=lambda x: x[1], reverse=True)
            return sorted_labels

        
        for pred in preds:
            labels = list(zip(self.tag_names, pred.astype(float)))
            
            # rating
            rating_labels = [labels[i] for i in self.rating_indexes]
            rating_dict = dict(rating_labels)
            max_rating_key = max(rating_dict, key=rating_dict.get)
            rating_dict = {max_rating_key: rating_dict[max_rating_key]}

            # character
            character_labels = [labels[i] for i in self.character_indexes]
            character_dict = dict(filter_and_sort(character_labels, self.character_threshold, self.exclude_tag))
            
            # general
            general_labels = [labels[i] for i in self.general_indexes]
            general_dict = dict(filter_and_sort(general_labels, self.general_threshold, self.exclude_tag))
            
            tags_dict = {}
            if self.use_rating:
                tags_dict |= rating_dict
            if self.use_character:
                tags_dict |= character_dict
            if self.use_general:
                tags_dict |= general_dict
            
            # all sort
            if self.all_sort:
                tags_dict = dict(sorted(tags_dict.items(), key=lambda item: item[1], reverse=True))

            # additional tag
            add_dict = {}
            if self.additional_tag:
                for tag in self.additional_tag:
                    add_dict[tag] = 1.0
            tags_dict = add_dict | {k:v for k, v in tags_dict.items() if k not in add_dict}
            
            # weighted captions
            tags = []
            for tag in list(tags_dict):
                new_tag = tag
                if self.weighted_captions:
                    new_tag = f"({new_tag}:{tags_dict[tag]})"
                tags.append(new_tag)
            tags_list.append(tags)
        
        return tags_list
    