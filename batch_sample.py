import os
import shutil

import cv2
import time
import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F

import torch

from dataset.collate_fn import CollateFN
from dataset.font_dataset import FontDataset

_ = torch.manual_seed(123)
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_content_encoder,
                 build_style_encoder)
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style)


def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False,
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    # parser.add_argument("--content_image_path", type=str, default=None)
    # parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None,
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    # args.style_image_path = style_image_path
    # args.content_image_path = content_image_path

    return args


def image_process(args, content_image=None, style_image=None):
    if not args.demo:
        # Read content image and style image
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            content_image_pil = None
        style_image = Image.open(args.style_image_path).convert('RGB')
    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil = None

    ## Dataset transform
    content_inference_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, \
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    content_image = content_inference_transforms(content_image)[None, :]
    style_image = style_inference_transforms(style_image)[None, :]

    return content_image, style_image, content_image_pil


def load_fontdiffuer_pipeline(args):
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe


def sampling(args, pipe, content_image=None, style_image=None):
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)

    content_image, style_image, content_image_pil = image_process(args=args,
                                                                  content_image=content_image,
                                                                  style_image=style_image)
    if content_image == None:
        print(f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn)
        end = time.time()

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                              image=images[0],
                                              content_image_pil=content_image_pil,
                                              content_image_path=None,
                                              style_image_path=args.style_image_path,
                                              resolution=args.resolution)
            else:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                              image=images[0],
                                              content_image_pil=None,
                                              content_image_path=args.content_image_path,
                                              style_image_path=args.style_image_path,
                                              resolution=args.resolution)
            print(f"Finish the sampling process, costing time {end - start}s")
        return images[0]


def load_controlnet_pipeline(args,
                             config_path="lllyasviel/sd-controlnet-canny",
                             ckpt_path="runwayml/stable-diffusion-v1-5"):
    from diffusers import ControlNetModel, AutoencoderKL
    # load controlnet model and pipeline
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    controlnet = ControlNetModel.from_pretrained(config_path,
                                                 torch_dtype=torch.float16,
                                                 cache_dir=f"{args.ckpt_dir}/controlnet")
    print(f"Loaded ControlNet Model Successfully!")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(ckpt_path,
                                                             controlnet=controlnet,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=f"{args.ckpt_dir}/controlnet_pipeline")
    # faster
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    print(f"Loaded ControlNet Pipeline Successfully!")

    return pipe


def controlnet(text_prompt,
               pil_image,
               pipe):
    image = np.array(pil_image)
    # get canny image
    image = cv2.Canny(image=image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(text_prompt,
                 num_inference_steps=50,
                 generator=generator,
                 image=canny_image,
                 output_type='pil').images[0]
    return image


def load_instructpix2pix_pipeline(args,
                                  ckpt_path="timbrooks/instruct-pix2pix"):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path,
                                                                  torch_dtype=torch.float16)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe


def instructpix2pix(pil_image, text_prompt, pipe):
    image = pil_image.resize((512, 512))
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(prompt=text_prompt, image=image, generator=generator,
                 num_inference_steps=20, image_guidance_scale=1.1).images[0]

    return image


def sample(args, pipe):
    # load fontdiffuser pipeline
    # pipe = load_fontdiffuer_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe)
    return out_image


def random_permutation(arr):
    permutation = arr.copy()
    random.shuffle(permutation)
    while any(x == y for x, y in zip(arr, permutation)):
        random.shuffle(permutation)
    return permutation


# 批量sample
def batch_sample(args):
    test_set = ["sfuc", "ufuc", "ufsc"]
    pipe = load_fontdiffuer_pipeline(args=args)

    for test in test_set:

        content_path = f"{os.path.join(args.data_path, test)}/ContentImage"
        target_path = f"{os.path.join(args.data_path, test)}/TargetImage"
        real_images_path = f"{os.path.join(args.save_image_dir, test)}/real_images"
        fake_images_path = f"{os.path.join(args.save_image_dir, test)}/fake_images"

        os.makedirs(real_images_path, exist_ok=True)
        os.makedirs(fake_images_path, exist_ok=True)

        for style in os.listdir(target_path):
            os.makedirs(os.path.join(real_images_path, style), exist_ok=True)
            os.makedirs(os.path.join(fake_images_path, style), exist_ok=True)

            content_file_list = os.listdir(content_path)
            style_file_list = os.listdir(os.path.join(target_path, style))
            random_style_file_list = random_permutation(style_file_list)
            max_num = 100
            count = 0

            for content_file, style_file in zip(content_file_list, random_style_file_list):
                if count < max_num:
                    image_name = style_file[:-5] + content_file

                    # copy real image
                    shutil.copy(os.path.join(target_path, style, image_name),
                                os.path.join(real_images_path, style, image_name))

                    # sample fake image
                    args.content_image_path = os.path.join(content_path, content_file)
                    args.style_image_path = os.path.join(target_path, style, style_file)

                    fake_image = sample(args, pipe).resize((256, 256))

                    # save
                    fake_image.save(os.path.join(fake_images_path, style, image_name))
                    count += 1


def calculate_fid(real_images_path, fake_images_path):
    fid = FrechetInceptionDistance(feature=64, normalize=True).to(torch.device("cuda", 0))
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    styles = os.listdir(real_images_path)
    if set(styles) != set(os.listdir(fake_images_path)):
        print("生成图片和真实图片不一致，，进程退出")
        exit(1)

    result = 0

    for style in styles:

        if set(os.listdir(os.path.join(real_images_path, style))) != set(
                os.listdir(os.path.join(fake_images_path, style))):
            print("生成图片和真实图片不一致，，进程退出")
            exit(1)

        for image in os.listdir(os.path.join(real_images_path, style)):
            # 读取 图像
            real_image = Image.open(os.path.join(real_images_path, style, image)).convert('RGB')
            fake_image = Image.open(os.path.join(fake_images_path, style, image)).resize((256, 256))

            # 应用数据变换并将图像转换为 PyTorch 张量
            real_image = transform(real_image).to(torch.device("cuda", 0))
            fake_image = transform(fake_image).to(torch.device("cuda", 0))

            fid.update(torch.stack([real_image]), real=True)
            fid.update(torch.stack([fake_image]), real=False)

        result += fid.compute()
        fid.reset()

    return result / len(styles)


def calculate_ssim(real_images_path, fake_images_path):
    ssim = StructuralSimilarityIndexMeasure().to(torch.device("cuda", 0))
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    styles = os.listdir(real_images_path)
    if set(styles) != set(os.listdir(fake_images_path)):
        print("生成图片的字体和真实图片不一致的字体，，进程退出")
        exit(1)

    result = 0

    for style in styles:

        if set(os.listdir(os.path.join(real_images_path, style))) != set(
                os.listdir(os.path.join(fake_images_path, style))):
            print("生成图片和真实图片不一致，，进程退出")
            exit(1)

        for image in os.listdir(os.path.join(real_images_path, style)):
            # 读取 图像
            real_image = Image.open(os.path.join(real_images_path, style, image)).convert("RGB")
            fake_image = Image.open(os.path.join(fake_images_path, style, image)).resize((256, 256))

            # 应用数据变换并将图像转换为 PyTorch 张量
            target = transform(real_image).to(torch.device("cuda", 0))
            predicted = transform(fake_image).to(torch.device("cuda", 0))

            ssim.update(torch.stack([predicted]), torch.stack([target]))

        result += ssim.compute()
        ssim.reset()

    return result / len(styles)


def calculate_lpips(real_images_path, fake_images_path):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(torch.device("cuda", 0))
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    styles = os.listdir(real_images_path)
    if set(styles) != set(os.listdir(fake_images_path)):
        print("生成图片的字体和真实图片的字体不一致，，进程退出")
        exit(1)

    result = 0

    for style in styles:

        if set(os.listdir(os.path.join(real_images_path, style))) != set(
                os.listdir(os.path.join(fake_images_path, style))):
            print("生成图片和真实图片不一致，，进程退出")
            exit(1)

        for image in os.listdir(os.path.join(real_images_path, style)):
            # 读取 图像
            real_image = Image.open(os.path.join(real_images_path, style, image)).convert("RGB")
            fake_image = Image.open(os.path.join(fake_images_path, style, image)).resize((256, 256))

            # 应用数据变换并将图像转换为 PyTorch 张量
            target = transform(real_image).to(torch.device("cuda", 0))
            predicted = transform(fake_image).to(torch.device("cuda", 0))

            lpips.update(torch.stack([predicted]), torch.stack([target]))

        result = result + lpips.compute()
        lpips.reset()

    return result / len(styles)


def calculate_psnr(real_images_path, fake_images_path):
    psnr = PeakSignalNoiseRatio().to(torch.device("cuda", 0))
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    styles = os.listdir(real_images_path)
    if set(styles) != set(os.listdir(fake_images_path)):
        print("生成图片的字体和真实图片的字体不一致，，进程退出")
        exit(1)

    result = 0

    for style in styles:

        if set(os.listdir(os.path.join(real_images_path, style))) != set(
                os.listdir(os.path.join(fake_images_path, style))):
            print("生成图片和真实图片不一致，，进程退出")
            exit(1)

        for image in os.listdir(os.path.join(real_images_path, style)):
            # 读取 图像
            real_image = Image.open(os.path.join(real_images_path, style, image)).convert("RGB")
            fake_image = Image.open(os.path.join(fake_images_path, style, image)).resize((256, 256))

            # 应用数据变换并将图像转换为 PyTorch 张量
            target = transform(real_image).to(torch.device("cuda", 0))
            predicted = transform(fake_image).to(torch.device("cuda", 0))

            psnr.update(torch.stack([predicted]), torch.stack([target]))

        result = result + psnr.compute()
        psnr.reset()

    return result / len(styles)


# def calculate_l1(real_images_path, fake_images_path):
#     styles = os.listdir(real_images_path)
#
#     if set(styles) != set(os.listdir(fake_images_path)):
#         print("生成图片的字体和真实图片不一致的字体，，进程退出")
#         exit(1)
#
#     result = 0
#     num = 0
#
#     for style in styles:
#
#         style_result = 0
#         num += len(os.listdir(os.path.join(real_images_path, style)))
#
#         if set(os.listdir(os.path.join(real_images_path, style))) != set(
#                 os.listdir(os.path.join(fake_images_path, style))):
#             print("生成图片和真实图片不一致，，进程退出")
#             exit(1)
#
#         for image in os.listdir(os.path.join(real_images_path, style)):
#             # 读取 图像
#             real_image = Image.open(os.path.join(real_images_path, style, image))
#             fake_image = Image.open(os.path.join(fake_images_path, style, image)).convert('L').resize((256, 256))
#
#             # 定义数据变换
#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
#
#             # 应用数据变换并将图像转换为 PyTorch 张量
#             target = transform(real_image).to(torch.device("cuda", 0))
#             predicted = transform(fake_image).to(torch.device("cuda", 0))
#
#             style_result = style_result + F.l1_loss(predicted, target).to(torch.device("cuda", 0))
#
#         result = result + style_result
#
#     return result / num


def test(data_path):
    test_set = ["sfuc", "ufuc", "ufsc"]

    for test in test_set:
        real_images_path = os.path.join(data_path, test, "real_images")
        fake_images_path = os.path.join(data_path, test, "fake_images")

        # l1_result = calculate_l1(real_images_path, fake_images_path)
        # print(f"{test}_l1_result", l1_result)

        psnr_result = calculate_psnr(real_images_path, fake_images_path)
        print(f"{test}_psnr_result", psnr_result)

        ssim_result = calculate_ssim(real_images_path, fake_images_path)
        print(f"{test}_ssim_result", ssim_result)

        lpips_result = calculate_lpips(real_images_path, fake_images_path)
        print(f"{test}_lpips_result", lpips_result)

        fid_result = calculate_fid(real_images_path, fake_images_path)
        print(f"{test}_fid_result", fid_result)


def showtime(start_time, end_time):
    # 计算持续时间
    duration = end_time - start_time

    # 提取出持续时间中的小时、分钟和秒数
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    if hours > 0:
        print(f"Spent time: {hours} hours, {minutes} minutes, {seconds} seconds")

    elif minutes > 0:
        print(f"Spent time: {minutes} minutes, {seconds} seconds")

    else:
        print(f"Spent time: {seconds} seconds")


if __name__ == "__main__":
    start_time = time.time()

    args = arg_parse()

    batch_sample(args)

    test(args.save_image_dir)

    end_time = time.time()

    showtime(start_time, end_time)

    pass
