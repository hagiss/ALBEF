import json
import ruamel_yaml as yaml
import argparse
import torch.backends.cudnn as cudnn
import random
from pathlib import Path
from torchvision import transforms
import fire
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from transformers import AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup, DebertaV2ForMaskedLM, \
    DataCollatorForWholeWordMask
from torchvision.datasets import ImageFolder
from args import parse_dict
import logging
import os, psutil
from PIL import ImageFile
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from PIL import Image, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
process = psutil.Process(os.getpid())

logger = logging.getLogger(__name__)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class COCODetectionDataset(Dataset):
    def __init__(self, root, transform, prompt="", definition=False):
        super().__init__()
        self.transform = transform
        self.definition = definition

        # load image_path, annotations
        image_path = os.path.join(root, "val2017")
        annotation_path = os.path.join(root, "annotations-2/instances_val2017.json")

        # key is image_key, value is (path, [])
        self.datas = {}

        # for path in listdir(image_path):
        #     key = int(path.split(".")[0])
        #     self.datas[key] = (os.path.join(image_path, path), [])

        # load captions
        with open(annotation_path) as json_file:
            json_data = json.load(json_file)
            annotations = json_data['annotations']
            categories = json_data['categories']

        ########### for definition ############
        categories = [{'definition': 'a human being regarded as an individual', 'supercategory': 'person', 'id': 1, 'name': 'person'},
                      {'definition': 'a vehicle composed of two wheels held in a frame one behind the other, propelled by pedals and steered with handlebars attached to the front wheel', 'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                      {'definition': 'a four-wheeled road vehicle that is powered by an engine and is able to carry a small number of people', 'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                      {'definition': 'a two-wheeled vehicle that is powered by a motor and has no pedals', 'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
                      {'definition': 'a powered flying vehicle with fixed wings and a weight greater than that of the air it displaces', 'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
                      {'definition': 'a large motor vehicle carrying passengers by road, typically one serving the public on a fixed route and for a fare', 'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
                      {'definition': 'a series of railroad cars moved as a unit by a locomotive or by integral motors', 'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
                      {'definition': 'a large, heavy motor vehicle used for transporting goods, materials, or troops.', 'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
                      {'definition': 'a small vessel propelled on water by oars, sails, or an engine', 'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
                      {'definition': 'a set of automatically operated colored lights, typically red, amber, and green, for controlling traffic at road junctions and crosswalks', 'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
                      {'definition': 'a connection point by which firefighters can tap into a water supply', 'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
                      {'definition': 'a sign telling drivers to stop and wait until they can continue safely', 'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
                      {'definition': 'a coin-operated device which registers the purchase of parking time for a motor vehicle', 'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
                      {'definition': 'a long seat for several people, typically made of wood or stone', 'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
                      {'definition': 'a warm-blooded egg-laying vertebrate distinguished by the possession of feathers, wings, and a beak and (typically) by being able to fly', 'supercategory': 'animal', 'id': 16, 'name': 'bird'},
                      {'definition': 'a small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws. It is widely kept as a pet or for catching mice, and many breeds have been developed', 'supercategory': 'animal', 'id': 17, 'name': 'cat'},
                      {'definition': 'a domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, nonretractable claws, and a barking, howling, or whining voice', 'supercategory': 'animal', 'id': 18, 'name': 'dog'},
                      {'definition': 'a large plant-eating domesticated mammal with solid hoofs and a flowing mane and tail, used for riding, racing, and to carry and pull loads', 'supercategory': 'animal', 'id': 19, 'name': 'horse'},
                      {'definition': 'a domesticated ruminant animal with a thick woolly coat and (typically only in the male) curving horns. It is kept in flocks for its wool or meat, and is proverbial for its tendency to follow others in the flock', 'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
                      {'definition': 'a fully grown female animal of a domesticated breed of ox, kept to produce milk or beef', 'supercategory': 'animal', 'id': 21, 'name': 'cow'},
                      {'definition': 'a heavy plant-eating mammal with a prehensile trunk, long curved ivory tusks, and large ears, native to Africa and southern Asia. It is the largest living land animal', 'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
                      {'definition': 'a large, heavy mammal that walks on the soles of its feet, having thick fur and a very short tail', 'supercategory': 'animal', 'id': 23, 'name': 'bear'},
                      {'definition': 'an African wild horse with black-and-white stripes and an erect mane', 'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
                      {'definition': 'a large African mammal with a very long neck and forelegs, having a coat patterned with brown patches separated by lighter lines. It is the tallest living animal', 'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
                      {'definition': "a bag with shoulder straps that allow it to be carried on one's back", 'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
                      {'definition': 'a device consisting of a circular canopy of cloth on a folding metal frame supported by a central rod, used as protection against rain or sometimes sun', 'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
                      {'definition': "a woman's purse", 'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
                      {'definition': "a strip of material worn around the collar and tied in a knot at the front with the ends hanging down, typically forming part of a man's business or formal outfit", 'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
                      {'definition': "a case with a handle and a hinged lid, used for carrying clothes and other personal possessions", 'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
                      {'definition': "a concave plastic disk designed for skimming through the air as an outdoor game or amusement", 'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
                      {'definition': "each of a pair of long narrow pieces of hard flexible material, typically pointed and turned up at the front, fastened under the feet for gliding over snow", 'supercategory': 'sports', 'id': 35, 'name': 'skis'},
                      {'definition': "a board resembling a short, broad ski, used for sliding downhill on snow", 'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
                      {'definition': "an exercise ball is a ball constructed of soft elastic", 'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
                      {'definition': "a toy consisting of a light frame with thin material stretched over it, flown in the wind at the end of a long string", 'supercategory': 'sports', 'id': 38, 'name': 'kite'},
                      {'definition': "a smooth wooden or metal club used in the sport of baseball to hit the ball after it is thrown by the pitcher", 'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
                      {'definition': "a large glove worn by baseball players of the defending team, which assists players in catching and fielding balls hit by a batter or thrown by a teammate", 'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
                      {'definition': "a short narrow board with two small wheels fixed to the bottom of either end, on which (as a recreation or sport) a person can ride in a standing or crouching position", 'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
                      {'definition': "a long, narrow streamlined board used in surfing", 'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
                      {'definition': "a sports implement used for striking a ball or shuttlecock in games such as squash, tennis, racquetball, badminton and padel", 'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
                      {'definition': "a container, typically made of glass or plastic and with a narrow neck, used for storing drinks or other liquids", 'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
                      {'definition': "a glass with a stem and foot, used for drinking wine", 'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
                      {'definition': "a small bowl-shaped container for drinking from, typically having a handle", 'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
                      {'definition': "an implement with two or more prongs used for lifting food to the mouth or holding it when cutting", 'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
                      {'definition': "an instrument composed of a blade fixed into a handle, used for cutting or as a weapon", 'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
                      {'definition': "an implement consisting of a small, shallow oval or round bowl on a long handle, used for eating, stirring, and serving food", 'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
                      {'definition': "a round, deep dish or basin used for food or liquid", 'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
                      {'definition': "a long curved fruit which grows in clusters and has soft pulpy flesh and yellow skin when ripe", 'supercategory': 'food', 'id': 52, 'name': 'banana'},
                      {'definition': "the round fruit of a tree of the rose family, which typically has thin red or green skin and crisp flesh. Many varieties have been developed as dessert or cooking fruit or for making cider", 'supercategory': 'food', 'id': 53, 'name': 'apple'},
                      {'definition': "an item of food consisting of two pieces of bread with meat, cheese, or other filling between them, eaten as a light meal", 'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
                      {'definition': "a round juicy citrus fruit with a tough bright reddish-yellow rind", 'supercategory': 'food', 'id': 55, 'name': 'orange'},
                      {'definition': "a cultivated variety of cabbage bearing heads of green or purplish flower buds that are eaten as a vegetable", 'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
                      {'definition': "a tapering orange-colored root eaten as a vegetable", 'supercategory': 'food', 'id': 57, 'name': 'carrot'},
                      {'definition': "a frankfurter, especially one served hot in a long, soft roll and topped with various condiments", 'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
                      {'definition': "a dish of Italian origin consisting of a flat, round base of dough baked with a topping of tomato sauce and cheese, typically with added meat or vegetables", 'supercategory': 'food', 'id': 59, 'name': 'pizza'},
                      {'definition': "a small fried cake of sweetened dough, typically in the shape of a ball or ring", 'supercategory': 'food', 'id': 60, 'name': 'donut'},
                      {'definition': "an item of soft, sweet food made from a mixture of flour, shortening, eggs, sugar, and other ingredients, baked and often decorated", 'supercategory': 'food', 'id': 61, 'name': 'cake'},
                      {'definition': "a separate seat for one person, typically with a back and four legs", 'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
                      {'definition': "a long upholstered piece of furniture for several people to sit on", 'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
                      {'definition': "an ornamental plant that is grown indoors", 'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
                      {'definition': "a piece of furniture for sleep or rest, typically a framework with a mattress and coverings", 'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
                      {'definition': "a table on which meals are served in a dining room", 'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
                      {'definition': "a fixed receptacle into which a person may urinate or defecate, typically consisting of a large bowl connected to a system for flushing away the waste into a sewer or septic tank", 'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
                      {'definition': "a device that receives television signals and reproduces them on a screen", 'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
                      {'definition': "a computer that is portable and suitable for use while traveling", 'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
                      {'definition': "a small handheld device that is dragged across a flat surface to move the cursor on a computer screen, typically having buttons that are pressed to control functions", 'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
                      {'definition': "a remote control device", 'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
                      {'definition': "a panel of keys that operate a computer or typewriter", 'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
                      {'definition': "a phone with access to a cellular radio system so it can be used over a wide area, without a physical connection to a network", 'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
                      {'definition': "an oven that uses microwaves to cook or heat food", 'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
                      {'definition': "an enclosed compartment, as in a kitchen range, for cooking and heating food", 'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
                      {'definition': "an electrical device for making toast", 'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
                      {'definition': "a bowl-shaped plumbing fixture for washing hands, dishwashing, and other purposes", 'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
                      {'definition': "an appliance or compartment which is artificially kept cool and used to store food and drink", 'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
                      {'definition': "a written or printed work consisting of pages glued or sewn together along one side and bound in covers", 'supercategory': 'indoor', 'id': 84, 'name': 'book'},
                      {'definition': "a mechanical or electrical device for measuring time, indicating hours, minutes, and sometimes seconds, typically by hands on a round dial or by displayed figures", 'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
                      {'definition': "a decorative container, typically made of glass or china and used as an ornament or for displaying cut flowers", 'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
                      {'definition': "an instrument used for cutting cloth, paper, and other thin material, consisting of two blades laid one on top of the other", 'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
                      {'definition': "a soft toy bear", 'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
                      {'definition': "an electrical device for drying a person's hair by blowing warm air over it", 'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
                      {'definition': "a small brush with a long handle, used for cleaning the teeth", 'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]
        ################################

        id2cat = dict()
        id2def = dict()
        for cat in categories:
            id2cat[cat['id']] = cat['name']
            id2def[cat['id']] = cat['definition']

        err_cnt = 0
        # curMax = 0
        # idx = 0
        self.idx2key = dict()
        for i, ann in enumerate(annotations):
            key = ann['image_id']
            category = ann['category_id']
            unique_key = str(key) + "_" + str(category)
            category_name = id2cat[category]
            category_name = prompt + category_name
            definition_name = id2def[category]
            path = str(key).zfill(12) + ".jpg"
            self.datas[unique_key] = [os.path.join(image_path, path), None, None]
            try:
                self.datas[unique_key][1] = category_name
                self.datas[unique_key][2] = definition_name
            except:
                err_cnt += 1
            # curMax = max(curMax, len(self.datas[key][1]))
            self.idx2key[i] = unique_key

        # self.datas = list(self.datas.values())
        # print(self.datas[:10])

        print(f"TOTAL COCO: {len(list(self.datas.values()))}")
        print(f"Total errors in COCO: {err_cnt}")

    def __getitem__(self, item):
        unique_key = self.idx2key[item]
        image_path, category, definition = self.datas[unique_key]
        # try:
        #     category = random.choice(category)
        # except:
        #     category = "None"

        image = pil_loader(image_path)
        image = self.transform(image)
        # if len(texts) != 5:
        #     print(len(texts))

        # ret = {"image": image, "text1": texts[0],}
        # return ret
        if self.definition:
            return image, category, unique_key, definition
        return image, category, unique_key, category

    def __len__(self):
        return len(list(self.datas.values()))


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class CollateFunction(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.wwm = DataCollatorForWholeWordMask(self.tokenizer, mlm_probability=0.15)

    def whole_word_masking(self, tokens):
        # print(tokens.shape)
        masked_tokens = self.wwm(tokens)
        masked_tokens['input_ids'] = torch.tensor(masked_tokens['input_ids'])
        masked_tokens['labels'] = torch.tensor(masked_tokens['labels'])
        masked_tokens['unmasked_ids'] = torch.tensor(tokens)
        return masked_tokens

    # batch : [batch, str]
    def collate_fn(self, batch):
        # print(batch)
        text = self.tokenizer(batch, padding=True, truncation=True, max_length=78, add_special_tokens=False)#, return_tensors='pt')
        # print(text)
        masked1 = self.whole_word_masking(text['input_ids'])
        masked2 = self.whole_word_masking(text['input_ids'])
        attention_mask = torch.tensor(text['attention_mask'], dtype=torch.float32)

        tokens = {
            'input_ids': torch.tensor(text['input_ids']),
            'attention_mask': attention_mask,
            'masked_ids1': masked1['input_ids'],
            'masked_ids2': masked2['input_ids'],
            'labels1': masked1['labels'],
            'labels2': masked2['labels'],
            "unmasked_ids": masked1['unmasked_ids']
        }

        return tokens

    # batch : {image, text:[batch, str]}
    def collate_fn_laion(self, batch):
        image, text = batch
        # image, text = zip(*batch)
        # text = self.collate_fn(list(text))
        text = self.collate_fn(text)
        # return torch.stack(image), text
        return image, text

    def collate_fn_mm(self, text):
        text = self.collate_fn(list(text))
        return text

def detect(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Model construction")
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        # pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        # state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        # m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
        # state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)

        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    ####### TODO #######
    definition = False
    ####################

    # TODO check definition
    val_coco_dataset = COCODetectionDataset(config['val_root'], val_transform, definition=definition)
    print("coco", len(val_coco_dataset))
    val_coco_sampler = torch.utils.data.distributed.DistributedSampler(val_coco_dataset)
    val_coco_loader = torch.utils.data.DataLoader(
        val_coco_dataset, batch_size=128, sampler=val_coco_sampler,
        num_workers=4, drop_last=False, shuffle=False)

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    output_path = f"/workspace/localization/albef/category_normalized"
    os.makedirs(output_path, exist_ok=True)
    validateCOCO(model_without_ddp, val_coco_loader, tokenizer, output_path=output_path)

sfm = torch.nn.Softmax(dim=1)

def normalize(x):
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def getAttMap(img, attn_map):
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def _save_result(img, attn_map, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map))
    for ax in axes:
        ax.axis("off")
    fig.savefig(path)

def validateCOCO(model, val_loader, tokenizer, epoch=0, output_path="/workspace"):
    model.eval()

    cnt = 0
    with torch.no_grad():
        # TODO check definition
        for images, category, unique_key, definition in val_loader:
            cnt += 1
            batch_size = images.size(0)
            assert batch_size == len(category), "batch size should match"
            if cnt < 2:
                print(category)
            texts = tokenizer(category, padding=True, truncation=True, max_length=78, return_tensors='pt')
            images = images.cuda(non_blocking=True)

            image_embeddings, text_embeddings = model.forward_image_text(images, texts)

            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.unsqueeze(1) # (B, 1, dim)

            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            # mlp_image_embeddings = mlp_image_embeddings / mlp_image_embeddings.norm(dim=-1, keepdim=True)

            sim_grid = torch.bmm(image_embeddings, text_embeddings.permute(0,2,1)).squeeze() # (B, seq_len)
            # mlp_sim_grid = torch.bmm(mlp_image_embeddings, mlp_text_embeddings.permute(0,2,1)).squeeze() # (B, seq_len)

            sim_grid = sfm(sim_grid)
            # mlp_sim_grid = sfm(mlp_sim_grid)

            sim_grid = sim_grid.view(batch_size, 14, 14).unsqueeze(1)
            # sim_grid[:, :, 0] = sim_grid.min() # TODO: normalization heuristic
            # mlp_sim_grid = mlp_sim_grid.view(batch_size, 14, 14).unsqueeze(1)

            for i in range(batch_size):
                sim_grid[i,:,0] = sim_grid[i].min()

            sim_grid = F.interpolate(sim_grid, (224,224), mode='bicubic', align_corners=False).squeeze().detach().cpu().numpy()
            # mlp_sim_grid = F.interpolate(mlp_sim_grid, (224,224), mode='bicubic', align_corners=False).squeeze().detach().cpu().numpy()

            for i in range(batch_size):
                image_np = images[i].permute(1,2,0).detach().cpu().numpy().astype(np.float32)
                # assert image_np.max() <= 1, "Range Max"
                # assert image_np.min() >= 0, "Range Min"
                image_np = normalize(image_np)
                category_name = category[i].replace(" ", "_")
                _save_result(image_np, sim_grid[i], f"{output_path}/{unique_key[i]}_{category_name}.png")
                # _save_result(image_np, mlp_sim_grid[i], f"{output_path}/{category_name}_{cnt}_{i}_mlp.png")

            print(f"{cnt * batch_size} saved!")

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    detect(args, config)