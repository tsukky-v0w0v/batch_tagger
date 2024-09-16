from pathlib import Path
import argparse
import toml
from predictor import Predictor
from models import models
from utils import setup_logger
logger = setup_logger("tagger")

def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", type=str, required=True)

    parser.add_argument("--use_rating", action="store_true")
    parser.add_argument("--use_character", action="store_true")
    parser.add_argument("--use_general", action="store_true")

    parser.add_argument("--character_threshold", type=float, default=0.85)
    parser.add_argument("--general_threshold", type=float, default=0.35)
    parser.add_argument("--use_recommended_threshold", action="store_true")

    parser.add_argument("--ext", type=str, default=".txt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--weighted_captions", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--additional_tag", action="append")
    parser.add_argument("--exclude_tag", action="append")
    parser.add_argument("--all_sort", action="store_true")

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model", type=str, default="wd-v1-4-swinv2-tagger-v2", choices=models.keys())
    parser.add_argument("--batch_size", type=int, default=10)

    parser.add_argument("--config_file", type=str)
    return parser


def load_toml(file_path):
    if file_path:
        return toml.load(open(file_path))
    else:
        return {}


def image_files_list(args):
    image_extensions = [".png", ".jpg", "jpeg", "webp"]
    target_path = Path(args.target)
    if target_path.is_file():
        if target_path.suffix.lower() in image_extensions:
            return [target_path]
        else:
            return []
    
    if target_path.is_dir():
        if args.recursive:
            files = target_path.rglob("*")
        else:
            files = target_path.glob("")
        image_files = [file for file in files if file.suffix.lower() in image_extensions and file.is_file()]
        return image_files


def predict_images(predictor: Predictor, images, args):
    # predict
    preds = predictor.predict(images)
    tags_list = predictor.postprocess_tags(preds)

    # save to file
    for image_path, tags in zip(images, tags_list):
        caption_path = image_path.parent / f"{image_path.stem}{args.ext}"
        
        tags_string = ", ".join(tags)
        with open(caption_path, mode="w") as file:
            file.write(tags_string)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.config_file:
        config = load_toml(args.config_file)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    predictor = Predictor(args)
    
    batch = []
    image_files = image_files_list(args)
    for image_file in image_files:
        caption_path = image_file.parent / f"{image_file.stem}{args.ext}"
        
        if caption_path.is_file() and not args.overwrite:
            # 既にキャプションがある場合はスキップ
            logger.info(f"skip: {image_file}")
            continue

        batch.append(image_file)
        logger.info(f"processing: {image_file}")

        # バッチサイズに達したらpredict実行
        if len(batch) == args.batch_size:
            predict_images(predictor, batch, args)
            batch.clear()
        
    if batch:
        predict_images(predictor, batch, args)

    logger.info("completed")
    
        

    
