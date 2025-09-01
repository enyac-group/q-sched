import torch
import os
import scipy
import numpy as np
import pickle as pkl
import torch
import random
import os
import re
import math
import json
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import datasets
from tqdm import tqdm
import json
from email_notifs.send_email import send_email

def collect_clargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", default="test")
    parser.add_argument("--quant_type", default="fp16", choices=["fp16", "fp32", "w8a8", "w4a8", "w4a4", "w4a5", "w4a6", "qdiff", "smq", "int8_ptqd", "int4_ptqd", "svdquant_int4", "mixdq_int8", "mixdq_int4"])
    parser.add_argument("--ptqd_stats_file", type=str, default="")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--max_model_coeff", type=float, default=1.0)
    parser.add_argument("--max_sample_coeff", type=float, default=1.0)
    parser.add_argument("--start_model_coeff", type=float, default=1.0)
    parser.add_argument("--start_sample_coeff", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="PCM-2Step")
    parser.add_argument("--mode", type=str, default="sd15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coco_dir", type=os.path.expanduser, default="~/quantization-diffusions/coco", help="COCO2014 Directory Path")
    parser.add_argument("--calib_json", type=str, default="/mnt/Data/dataset_flickr30k.json", help="Calibration Prompt Json File")
    parser.add_argument("--quantized_modules", default=["unet"], nargs='+', help="modules in diffusion model to quantize. Options are unet, text_encoder1, text_encoder2, vae, ..")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--guidance", default=False, type=float, help="guidance scale parameter (if overriding model default)")
    parser.add_argument("--dataset", type=str, default="mjhq", choices=["karpathy", "svdquant", "mjhq"], help="Dataset to use for evaluation")
    parser.add_argument("--save_prompts", action="store_true", default=True, help="Save prompts to a text file in the output folder")
    return parser

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)

checkpoints = {
    "PCM-2Step": ["pcm_{}_smallcfg_2step_converted.safetensors", 2, 0.0],
    "PCM-4Step": ["pcm_{}_smallcfg_4step_converted.safetensors", 4, 0.0],
    "PCM-8Step": ["pcm_{}_smallcfg_8step_converted.safetensors", 8, 0.0],
    "PCM-16Step": ["pcm_{}_smallcfg_16step_converted.safetensors", 16, 0.0],
    "PCM-Normal_CFG_4-Step": ["pcm_{}_normalcfg_4step_converted.safetensors", 4, 7.5],
    "PCM-Normal_CFG_8-Step": ["pcm_{}_normalcfg_8step_converted.safetensors", 8, 7.5],
    "PCM-Normal_CFG_16-Step": ["pcm_{}_normalcfg_16step_converted.safetensors", 16, 7.5],
    "LCM-1Step": ["",1,8.0],
    "LCM-2Step": ["",2,8.0],
    "LCM-4Step": ["",4,8.0],
    "LCM-8Step": ["",8,8.0],
    "original": ["", 50, None],
    "SDXLTurbo-1Step": ["", 1, 0.0],  # SDXL Turbo uses 1 step and no guidance scale by default
    "SDXLTurbo-4Step": ["", 4, 0.0],
    "FLUX.1":["",4,0]
}

def quantize_pipe(pipe, args):
    if args.quant_type == "fp16":
        return pipe
    for module in args.quantized_modules:
        if "mixdq" in args.quant_type:
            print("MixDQ quantization")
            from diffusers import DiffusionPipeline
            assert "SDXLTurbo" in args.checkpoint
                
            if "int8" in args.quant_type:
                print("MixDQ with INT8")
                w_bit = 8
            elif "int4" in args.quant_type:
                print("MixDQ with INT4")
                w_bit=4
            else:
                raise NameError("quant_type does not specify int4 or int8")

            pipe.quantize_unet(
                w_bit=w_bit,
                a_bit=8,
                bos=True
            )
            pipe.set_cuda_graph(run_pipeline=True)
            pipe = pipe.to("cuda")
        elif "svdquant" in args.quant_type:
            assert "FLUX" in args.checkpoint
            from diffusers import FluxPipeline
            from flux.transformer_flux import NunchakuFluxTransformer2dModel
            from flux.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
            del pipe
            transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-schnell")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell", transformer=transformer,  torch_dtype=torch.bfloat16
            ).to("cuda")
            pipe.enable_model_cpu_offload()
            pipe.scheduler = FlowMatchEulerDiscreteScheduler(
                base_image_seq_len=256,
                base_shift=0.5,
                max_image_seq_len=4096,
                max_shift=1.15,
                num_train_timesteps=1000,
                shift=1.0,
                use_dynamic_shifting=False
            )
        elif args.quant_type == "smq":
            print("smoothquant quantization")
            import torchao
            from torchao.prototype.smoothquant import (
                insert_smooth_quant_observer_,
                SmoothQuantObservedLinear,
                smooth_quant,
                save_smooth_quant_recipe,
                load_smooth_quant_recipe
            )
            from torchao.quantization import quantize_
            
            insert_smooth_quant_observer_(pipe.unet, alpha=0.5, quant_mode="dynamic")
            
            is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
            print("Beginning Calibration")

            i = 0
            for idx in range(len(test_dataset)):
                print(i)
                sample = test_dataset[idx]
                captions = [sample['prompt']] if isinstance(sample['prompt'], str) else sample['prompt']
                imgs = pipe(prompt=captions, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance, eta=args.eta, output_type="pil")
                if i >= 10:
                    break
                i += 1

            torchao.quantization.quantize_(pipe.unet, smooth_quant(), is_observed_linear)
            save_smooth_quant_recipe(pipe.unet, "./smooth_quant_recipe.json")
        elif args.quant_type == "w4a8":
            print("w4a8 quantization")
            import torchao
            from torchao.quantization import (
                int8_dynamic_activation_int4_weight,
                quantize_
            )
            quantize_(pipe.components[module], int8_dynamic_activation_int4_weight())
        elif args.quant_type == "qdiff":
            print("INT4 Q-Diffusion quantization")
            from pcm.quantize_qdiff import quantize_qdiff
            pipe = quantize_qdiff(pipe)
        elif "ptqd" in args.quant_type:
            import torchao
            from torchao.quantization import (
                int8_dynamic_activation_int8_weight,
                int8_dynamic_activation_int4_weight,
                quantize_
            )
            
            args.ptqd_stats_file = args.save_folder+"/ptqd_dicts.pkl" if args.ptqd_stats_file == "" else args.ptqd_stats_file
            print("PTQD stats file: ", args.ptqd_stats_file)
            if not os.path.exists(args.ptqd_stats_file):
                args.ptqd_stats_file = args.save_folder+"/ptqd_dicts.pkl"
                noise_folder = args.save_folder+"/noise"
                qnoise_folder = args.save_folder+"/qnoise"
                num_calib_samples = 1024

                # load calibration data
                print("Loading Flickr Prompts..")
                flickr_captions = json.load(open(args.calib_json))
                calib_prompts = [c['sentences'][0]['raw'] for c in flickr_captions['images']]
                calib_prompts = calib_prompts[0:num_calib_samples]
                batchsize = 8
                
                print("Generating PTQD Stats")
                set_random_seed(args.seed)
                num_batches = math.ceil(num_calib_samples / batchsize)
                for batch_id in tqdm(range(0, num_batches)):
                    pipe(prompt=calib_prompts[batch_id:batch_id+batchsize], num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance, output_type="pil", \
                        collecting_ptqd=True, folder=noise_folder, eta=args.eta)
                    pipe.count += 1
                if args.quant_type == "int8_ptqd":
                    print("INT8 PTQD quantization")
                    quantize_(pipe.components[module], int8_dynamic_activation_int8_weight())
                elif args.quant_type == "int4_ptqd":
                    print("INT4 PTQD quantization")
                    quantize_(pipe.components[module], int8_dynamic_activation_int4_weight())
                
                pipe.count = 0
                # generate quantized samples
                set_random_seed(args.seed)
                for batch_id in tqdm(range(0, num_batches)):
                    pipe(prompt=calib_prompts[batch_id:batch_id+batchsize], num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance, output_type="pil", \
                    collecting_ptqd=True, folder=qnoise_folder, eta=args.eta)
                    pipe.count += 1
                print("Collecting & Computing PTQD Stats")
                slopes = {}
                intercepts = {}
                r_values = {}
                frames = []
                biases = {}
                stds = {}

                for t in range(0, args.num_inference_steps):
                    residual_count = 0
                    img_count = 0
                    error = None
                    data = None
                    print(str(t))
                    timestep = str(t)
                    prefix_fp = noise_folder +"/"+timestep+"/imgs"
                    prefix_q = qnoise_folder+"/"+timestep+"/imgs"
                    i = 0
                    fname = prefix_fp+str(i)+".pt"
                    q_fname = prefix_q+str(i)+".pt"
                    bs = 0
                    while os.path.exists(fname) and os.path.exists(q_fname):
                        noise_batch = torch.load(open(fname, "rb"))
                        qnoise_batch = torch.load(open(q_fname, "rb"))
                        residual_count = num_calib_samples - img_count
                        if i == 0:
                            error = (noise_batch - qnoise_batch).cpu()[0:residual_count,:,:,:]
                            data = noise_batch.cpu()[0:residual_count,:,:,:]
                            q_data = qnoise_batch.cpu()[0:residual_count,:,:,:]
                        else:
                            error = torch.cat((error, (noise_batch[0:residual_count,:,:,:] - qnoise_batch[0:residual_count,:,:,:]).cpu()), dim=0)
                            data = torch.cat((data, noise_batch[0:residual_count,:,:,:].cpu()), dim=0)
                            q_data = torch.cat((q_data, qnoise_batch[0:residual_count,:,:,:].cpu()), dim=0)
                        img_count += noise_batch.shape[0]
                        
                        i += 1
                        fname = prefix_fp + str(i)+".pt"
                        q_fname = prefix_q + str(i)+".pt"
                        if img_count >= num_calib_samples:
                            break
                    
                    channelwise_bias = torch.mean(error, dim=(0,2,3))
                    biases[t] = channelwise_bias
                    
                    k, intercept, r_value, p_value, std_err = scipy.stats.linregress(data.flatten(), error.flatten())
                    slopes[t] = k
                    intercepts[t] = intercept
                    r_values[t] = r_value

                    residual_error = data + error - (1+k)*data
                    stddev = torch.std(residual_error)
                    stds[t] = stddev

                f = open(args.ptqd_stats_file, "wb")
                pkl.dump([stds, biases, slopes, intercepts, r_values], f)
                f.close() 

            if args.quant_type == "int8_ptqd":
                print("INT8 PTQD quantization")
                quantize_(pipe.components[module], int8_dynamic_activation_int8_weight())
            elif args.quant_type == "int4_ptqd":
                print("INT4 PTQD quantization")
                quantize_(pipe.components[module], int8_dynamic_activation_int4_weight())

            pipe.ptqd_init(args.ptqd_stats_file)
            # Get first sample from dataset for testing
            test_sample = test_dataset[0]
            test_prompt = test_sample['prompt'] if isinstance(test_sample['prompt'], str) else test_sample['prompt']
            imgs = pipe(prompt=test_prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance, output_type="pil", \
                folder=args.save_folder, eta=args.eta).images
            imgs[0].save(args.save_folder+"/pqtd_test.png")
        elif args.quant_type == "fp32":
            print("Full Precision (FP32)")
            pipe = load_pipe(args.checkpoint, args.mode, dtype=torch.float32)
        elif args.quant_type == "w3a32":
            from torchao.quantization.observer import MappingType
            from torchao.dtypes import to_affine_quantized_intx
            
            quant_min = 0
            quant_max = 2
            for n, m in pipe.components[module].named_modules():
                if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                    m.weight = torch.nn.Parameter(to_affine_quantized_intx(m.weight, MappingType.SYMMETRIC, target_dtype=torch.uint4, block_size=(1,64), quant_min=quant_min, quant_max=quant_max))
        elif args.quant_type == "w4a4":
            print("W4A4 quantization")
            import torchao
            from torchao.quantization import (
                quantize_
            )
            from w4a4_config import int4_dynamic_activation_int4_weight

            for m in pipe.components[module].modules():
                if (isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d)):
                    quantize_(m, int4_dynamic_activation_int4_weight())
        elif args.quant_type == "w4a6":
            print("W4A6 quantization")
            import torchao
            from torchao.quantization import (
                quantize_
            )
            from w4a4_config import int6_dynamic_activation_int4_weight
            for m in pipe.components[module].modules():
                if (isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d)):
                    quantize_(m, int6_dynamic_activation_int4_weight())
        elif args.quant_type == "w4a5":
            print("W4A5 quantization")
            import torchao
            from torchao.quantization import (
                quantize_
            )
            from w4a4_config import int5_dynamic_activation_int4_weight
            for m in pipe.components[module].modules():
                if (isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d)):
                    quantize_(m, int5_dynamic_activation_int4_weight())
        else:
            print("int8 quantization")
            import torchao
            from torchao.quantization import (
                int8_dynamic_activation_int8_weight,
                quantize_
            )
            quantize_(pipe.components[module], int8_dynamic_activation_int8_weight())
    return pipe


def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in module.items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(8).to(dtype)

    return kohya_ss_state_dict


def load_pipe(args, dtype=torch.float16):
    ckpt = args.checkpoint
    
    if  "LCM" in ckpt:
        from lcm.lcm_pipeline import LatentConsistencyModelPipeline, LCMScheduler
        pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        pipe.scheduler = LCMScheduler(beta_start=0.00085, 
                        beta_end=0.0120, 
                        beta_schedule="scaled_linear", 
                        prediction_type="epsilon")
    elif "SDXLTurbo" in ckpt:
        from diffusers import DiffusionPipeline
        from sdxl_turbo.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=dtype,
            custom_pipeline="nics-efc/MixDQ" if "mixdq" in args.quant_type else None,
            variant="fp16"
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1,
            timestep_spacing="trailing",
            trained_betas=None
        )
    elif "FLUX" in ckpt:
        from diffusers import FluxPipeline
        from flux.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",  torch_dtype=torch.bfloat16
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(
            base_image_seq_len=256,
            base_shift=0.5,
            max_image_seq_len=4096,
            max_shift=1.15,
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=False
        )
    elif "PCM" in ckpt:
        if args.mode == "sd15":
            from pcm.sd15_pipeline import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                variant="fp16"
            )
        else:
            from pcm.sdxl_pipeline import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                variant="fp16",
            )
        if not ckpt == "original":
            checkpoint = checkpoints[ckpt][0].format(args.mode)
            pipe.load_lora_weights(
                "wangfuyun/PCM_Weights", weight_name=checkpoint, subfolder=args.mode
            )
        from pcm.tcd_scheduler import TCDScheduler
        pipe.scheduler = TCDScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
        )
    else:
        raise NotImplementedError("Pipe is not supported for mode: "+args.mode)
    pipe = pipe.to("cuda")
    # pcm_lora_weight = load_file(lora_fname)
    # pcm_lora_weight_convert = get_module_kohya_state_dict(pcm_lora_weight, "lora_unet", torch.float16)
    # pipe.load_lora_weights(
    #    pcm_lora_weight_convert
    # )
    return pipe

if __name__ == "__main__":
    
    parser = collect_clargs()
    args = parser.parse_args()
    args.num_inference_steps = checkpoints[args.checkpoint][1]
    args.guidance = args.guidance or checkpoints[args.checkpoint][2]
    print("Guidance Scale: "+str(args.guidance))
    pipe = load_pipe(args)
    pipe = quantize_pipe(pipe, args)
    args.scheduler = str(pipe.scheduler)

    save_img_dir=args.save_folder+"/dataset"
    os.makedirs(save_img_dir, exist_ok=True if args.resume else False)
    
    # Create a file to store prompts
    prompts_file = os.path.join(args.save_folder, "prompts.txt")
    if not args.resume or not os.path.exists(prompts_file):
        with open(prompts_file, 'w') as f:
            f.write("image_id,prompt\n")  # Write header
    
    json_file = args.save_folder+"/config.json"
    with open(json_file, 'wt') as f:
        json.dump(vars(args), f, indent=4)
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.deterministic=True 
    random_state = torch.random.get_rng_state()
    
    

    # Load dataset based on selection
    if args.dataset == "svdquant":
        # Load SVDQuant dataset and extract only text data
        dataset = datasets.load_dataset("DCI.py", max_dataset_size=5000, return_gt=False, split='train')
        # Create a simple dataset that only contains text
        test_dataset = datasets.Dataset.from_dict({
            'prompt': [item['prompt'] for item in dataset]
        })
        print(f"Loaded {len(test_dataset)} samples from SVDQuant dataset")
    elif args.dataset == "mjhq":
        # Load MJHQ dataset and extract only text data
        dataset = datasets.load_dataset("MJHQ.py", max_dataset_size=5000, split='train')
        # Create a simple dataset that only contains text
        test_dataset = datasets.Dataset.from_dict({
            'prompt': [item['prompt'] for item in dataset]
        })
        print(f"Loaded {len(test_dataset)} samples from MJHQ dataset")
    else:  # karpathy dataset
        from all_datasets import KarpathyDataset
        test_dataset = KarpathyDataset(args.coco_dir, split='restval')
        print(f"Loaded {len(test_dataset)} samples from Karpathy dataset")

    transform = transforms.ToTensor()
    set_random_seed(args.seed)
    pipe.count = 0
    pipe.scheduler.model_coeff = torch.linspace(args.max_model_coeff, args.start_model_coeff, args.num_inference_steps)
    pipe.scheduler.sample_coeff = torch.linspace(args.max_sample_coeff, args.start_sample_coeff, args.num_inference_steps)
    print("model coeffs: ", str(pipe.scheduler.model_coeff))
    print("sample coeffs: ", str(pipe.scheduler.sample_coeff))

    
    max_pipe_count = 0
    if args.resume:
        pattern = re.compile(r"^(\d+)_(\d+)\.png$")
        for fname in os.listdir(save_img_dir):
            file_components = fname.split("_")
            if len(file_components) >= 3:
                i = int(file_components[-2])
            else:
                i = int(file_components[0])
            if i > max_pipe_count:
                max_pipe_count = i
    print("max pipe count: ", str(max_pipe_count))

    # Unified iteration over both datasets
    def custom_collate_fn(batch):
        # Simply return the first item since we're using batch_size=1
        return batch[0]

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    for sample in tqdm(dataloader):
        if args.resume and pipe.count < max_pipe_count:
            pipe.count += 1
            continue
        try:
            # Handle both SVDQuant and Karpathy dataset formats
            if args.dataset == "svdquant":
                captions = [sample['prompt']] if isinstance(sample['prompt'], str) else sample['prompt']
            elif args.dataset == "mjhq":
                captions = [sample['prompt']] if isinstance(sample['prompt'], str) else sample['prompt']
            else:  # karpathy dataset
                captions = [sample['prompt'][0]] if isinstance(sample['prompt'][0], str) else sample['prompt'][0]
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
        print(captions)
        imgs = pipe(prompt=captions, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance, output_type="pil").images

        # Save generated images
        for i in range(0, len(imgs)):
            img = imgs[i]
            img_filename = f"{args.quant_type}_{pipe.count}_{i}.png"
            img.save(os.path.join(save_img_dir, img_filename))
            
            # Save prompt to prompts file
            if args.save_prompts:
                with open(prompts_file, 'a') as f:
                    f.write(f"{pipe.count}_{i},{captions[0]}\n")
        
        pipe.count += 1
    
    # Run FID score calculation
    if args.dataset == "svdquant":
        # For SVDQuant, we'll need to create a reference statistics file first
        print("Generating reference statistics for SVDQuant dataset...")
        os.system("python fid_score.py "+save_img_dir+" metadata/svdquant.npz --save-stats")
    elif args.dataset == "mjhq":
        # For MJHQ, we'll need to create a reference statistics file first
        print("Generating reference statistics for MJHQ dataset...")
        os.system("python fid_score.py "+save_img_dir+" metadata/mjhq.npz --save-stats")
    else:
        # For Karpathy dataset, use existing reference statistics
        out = os.system("python fid_score.py "+save_img_dir+" metadata/karpathy30k.npz --save-stats")
    
    send_email(json_file)
