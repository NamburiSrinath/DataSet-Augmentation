from huggingface_hub import notebook_login
from datetime import datetime
import os
import random
import numpy as np
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

if __name__ == "__main__":
    def dummy(images, **kwargs):
        return images, False

    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    lms = LMSDiscreteScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear"
    )

    acc_tok = "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"
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        scheduler=lms,
        use_auth_token=acc_tok,
    ).to("cuda")

    
    pipe.safety_checker = dummy

    CLASS_NAMES = ['dog', 'frog', 'horse', 'ship', 'truck']

    prompts = []

    fr = open('prompts_2.txt','r')
    for fl in fr:
        prompts += fl.strip().split(',')

    print(prompts)
    n_predictions = 6000

    # if not os.path.exists("CIFAR_10/synthetic"):
    #     os.mkdir("CIFAR_10/synthetic")

    # if not os.path.exists("CIFAR_10/synthetic/images"):
    #     os.mkdir("CIFAR_10/synthetic/images")

    # for tmp in CLASS_NAMES:
    #     if not os.path.exists("CIFAR_10/synthetic/images/" + tmp):
    #         os.mkdir("CIFAR_10/synthetic/images/" + tmp)

    if not os.path.exists("./images_gen"):
        os.mkdir("./images_gen")
    for tmp in CLASS_NAMES:
        if not os.path.exists("./images_gen/" + tmp):
            os.mkdir("./images_gen/"+ tmp)

    for i in range(n_predictions):
        for prompt_indx, prompt in enumerate(prompts):
            print(prompt)

            with autocast("cuda"):
                image = pipe(prompt, height=512, width=512)["sample"][0]  

            now = datetime.now()
            time = now.strftime("%Y%m%d_%H%M%S")

            img_name = CLASS_NAMES[prompt_indx] + "_" + time + "_" + str(i) + ".png"

            # print("***" + "generated_images/images/" + prompt + "/" + img_name +  "***")

            # image.save("generated_images_prompting/images/" + CLASS_NAMES[i] + "/" + img_name)
            image.save("images_gen/" + CLASS_NAMES[prompt_indx] + "/" + img_name)

        if i % 10 == 0:
            print(str(i) + " completed")