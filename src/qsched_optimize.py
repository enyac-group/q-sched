import random
import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import torchvision
from torchmetrics.multimodal.clip_score import CLIPScore
import piq
from torchmetrics.multimodal import CLIPImageQualityAssessment
import pickle as pkl
from eval_dataset import checkpoints, load_pipe, set_random_seed, collect_clargs, quantize_pipe
import json
import hpsv2
import sys
import yaml

def plot_epoch_results(imgs, captions, m_coeffs, s_coeffs, args, epoch=None, score=None, save_folder=None, show_clipiqa=False):
    """
    Plot the generated images with their CLIP-IQA scores.
    
    Args:
        imgs (list): List of generated PIL images
        captions (list): List of captions used for generation
        m_coeffs: Either a single float or tuple of (start_m, end_m)
        s_coeffs: Either a single float or tuple of (start_s, end_s)
        args: Command line arguments
        epoch (int, optional): Current epoch number
        score (float, optional): Current optimization score
        save_folder (str, optional): Folder to save the plot
        show_clipiqa (bool, optional): Whether to show CLIP-IQA scores in titles
    """
    num_imgs_per_row = 5
    num_imgs_per_col = max(1, len(captions) // num_imgs_per_row)

    # Create the plot
    fig, ax = plt.subplots(num_imgs_per_col, num_imgs_per_row, figsize=(3*num_imgs_per_row, 3*num_imgs_per_col))
    if num_imgs_per_col == 1:
        ax = ax.reshape(1, -1)  # Reshape to 2D array for consistent indexing
    
    # Compute CLIP-IQA scores for the images if needed
    if show_clipiqa:
        tensorized_images = torch.stack([torchvision.transforms.functional.pil_to_tensor(img) for img in imgs]).to("cuda")
        clipiqa_metric = CLIPImageQualityAssessment(prompts=('quality',)).to("cuda")
        clipiqa = clipiqa_metric(tensorized_images)
    
    # Display images with their scores
    for j in range(num_imgs_per_col):
        for i in range(num_imgs_per_row):
            idx = i + j * num_imgs_per_row
            if idx < len(imgs):
                ax[j, i].imshow(imgs[idx])
                if show_clipiqa:
                    ax[j, i].set_title(f"CLIP-IQA: {round(clipiqa[idx].item(), 3)}")
                ax[j, i].set_axis_off()
            else:
                ax[j, i].set_visible(False)  # Hide empty subplots

    # Set the title based on whether it's an epoch plot or final results
    if isinstance(m_coeffs, (int, float)):
        m_str = f"{m_coeffs:.3f}"
    else:
        start_m, end_m = m_coeffs
        m_str = f"{start_m:.3f}→{end_m:.3f}"
        
    if isinstance(s_coeffs, (int, float)):
        s_str = f"{s_coeffs:.3f}"
    else:
        start_s, end_s = s_coeffs
        s_str = f"{start_s:.3f}→{end_s:.3f}"

    # Build title based on available information
    title_parts = []
    if epoch is not None:
        title_parts.append(f"Epoch {epoch+1}")
    elif args.baseline:
        title_parts.append("Baseline")
    else:
        title_parts.append("Best Results")
    
    title_parts.append(f"Model: {m_str}, Sample: {s_str}")
    
    if score is not None:
        title_parts.append(f"Score: {round(score, 4)}")
    
    plt.suptitle("\n".join(title_parts))
    
    # Set filename based on available information
    if epoch is not None:
        fname = f"epoch_{epoch+1}_m_{m_str}_s_{s_str}_results.png"
    elif args.baseline:
        fname = f"baseline_{args.checkpoint}_{args.quant_type}_eta{args.eta}.png"
    else:
        fname = f"best_results_{args.ranking_criteria}_k{args.k}.png"
    
    plt.tight_layout()
    
    # Save the plot
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        fname = os.path.join(save_folder, fname)
        plt.savefig(fname, dpi=300)
    plt.close()


# Add the Q-Refine directory to the Python path
q_refine_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q-Refine")
if q_refine_dir not in sys.path:
    sys.path.append(q_refine_dir)

# Import the AQ-Map and PQ-Map modules only when needed
aqmap_imported = False
pqmap_imported = False
evaluate_aqmap = None
evaluate_pqmap = None
preprocessor = None

def import_aqmap():
    global aqmap_imported, evaluate_aqmap
    if not aqmap_imported:
        from metrics.aqmap import evaluate_aqmap
        aqmap_imported = True
        print("AQ-Map module imported successfully")

def import_pqmap():
    global pqmap_imported, evaluate_pqmap, preprocessor
    if not pqmap_imported:
        from metrics.pqmap import perceptual_quality_map, preprocessor
        evaluate_pqmap = perceptual_quality_map  # Assign the imported function to the global variable
        pqmap_imported = True
        print("PQ-Map module imported successfully")

ranking_dict = {
    "clip_score": 0,
    "clipiqa": 1,
    "new": 2,
    "brisque": 3,
    "hpv2": 4,
    "aqmap": 5,
    "pqmap": 6,
    "kclip2": 7
}

if __name__ == "__main__":
    parser = collect_clargs()
    parser.add_argument("--stages", nargs="+", default=['generate', 'score', 'rank'])
    parser.add_argument("--batchsize", default=5, type=int)
    parser.add_argument("--k", default=2, type=float)
    parser.add_argument("--ranking_criteria", default="new", type=str)
    parser.add_argument("--min_model_coeff", default=0.8, type=float)
    parser.add_argument("--min_sample_coeff", default=0.8, type=float)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--calib_size", default=20, type=int,)
    parser.add_argument("--key", default="None", type=str)
    parser.add_argument("--num_points", default=10, type=int,
                        help="Number of points to sample between min and max coefficients")
    parser.add_argument("--use_single_scale", default=True, action="store_true",
                      help="Use single scale coefficients instead of start/end pairs")
    parser.set_defaults(max_model_coeff=1.1, max_sample_coeff=1.1, save_folder=None)

    args = parser.parse_args()
    set_random_seed(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.deterministic=True 

    num_imgs_per_row = 5
    # Load DCI prompts from yaml file
    with open('dci_prompts.yaml', 'r') as f:
        dci_prompts = yaml.safe_load(f)
    
    # Convert prompts dict to list and take first 20 prompts (5 rows * 4 columns)
    captions = list(dci_prompts.values())[:args.calib_size]
    num_imgs_per_col = max(1, len(captions) // num_imgs_per_row)
    guidance_scale = checkpoints[args.checkpoint][2]
    args.num_inference_steps = checkpoints[args.checkpoint][1]
    if args.save_folder is None:
        args.save_folder="schedule_ablations/"+args.mode+"_"+args.checkpoint+"_"+args.quant_type+"_eta"+str(args.eta)
    
    os.makedirs(args.save_folder, exist_ok=True)
    
    # load existing imgs_dict
    try:
        f = open(args.save_folder + "/dict.pkl", "rb")
        imgs_dict = pkl.load(f)
        f.close()
    except:
        imgs_dict = {}

    # load existing scores_dict
    try:
        f = open(args.save_folder+"/scores_dict.pkl", "rb")
        scores_dict = pkl.load(f)
        f.close()
    except:
        scores_dict = {}
    
    print("Stages: "+str(args.stages))
    if 'generate' in args.stages:
        pipe = load_pipe(args)
        pipe = quantize_pipe(pipe, args)
        
        # Add memory optimization
        torch.cuda.empty_cache()  # Clear CUDA cache before starting
        
        # Reduce batch size if needed to avoid OOM
        effective_batchsize = min(args.batchsize, 5)  # Limit batch size to 5
        print(f"Using effective batch size of {effective_batchsize} to avoid OOM")
        epoch = 0
        for sample_idx in torch.linspace(args.min_sample_coeff,args.max_model_coeff,args.num_points):
            for m_idx in torch.linspace(args.min_model_coeff, args.max_model_coeff,args.num_points):
                pipe.scheduler.model_coeff = torch.linspace(m_idx,1, args.num_inference_steps)
                pipe.scheduler.sample_coeff = torch.linspace(sample_idx,1, args.num_inference_steps)
                key = "m"+str(round(m_idx.item(),3))+"_s"+str(round(sample_idx.item(),3))
                if key in imgs_dict.keys() and len(imgs_dict[key]) == args.calib_size:
                    print("Skipping m"+str(round(m_idx.item(),3))+"_s"+str(round(sample_idx.item(),3))+" as it already has "+str(args.calib_size)+" images")
                    continue
                imgs = []
                for batch in range(0, len(captions), effective_batchsize):
                    torch.manual_seed(args.seed)
                    # Clear memory between batches
                    torch.cuda.empty_cache()
                    
                    if args.checkpoint == "FLUX.1":
                        imgs_batch = pipe(prompt=captions[batch:batch+effective_batchsize], 
                                         num_inference_steps=args.num_inference_steps, 
                                         guidance_scale=guidance_scale).images
                    else:
                        imgs_batch = pipe(prompt=captions[batch:batch+effective_batchsize], 
                                         num_inference_steps=args.num_inference_steps, 
                                         guidance_scale=guidance_scale, 
                                         eta=args.eta).images
                    imgs.extend(imgs_batch)
                    
                    # Save intermediate results to avoid losing progress
                    if len(imgs) % 20 == 0:
                        imgs_dict[key] = imgs
                        f = open(args.save_folder+"/dict.pkl", "wb")
                        pkl.dump(imgs_dict,f)
                        f.close()
                        print(f"Saved intermediate results for m{round(m_idx.item(),3)}_s{round(sample_idx.item(),3)}")
                
                plot_epoch_results(
                    imgs[-20:],  # Plot last 20 images
                    captions[batch:batch+len(imgs_batch)],
                    (m_idx.item(), 1.0),  # Current model coefficients
                    (sample_idx.item(), 1.0),  # Current sample coefficients
                    args,
                    epoch=epoch,
                    score=None,
                    save_folder=args.save_folder,
                    show_clipiqa=True
                )
                epoch += 1
                print(f"Plotted batch of images at m{round(m_idx.item(),3)}_s{round(sample_idx.item(),3)}")
                
                imgs_dict["m"+str(round(m_idx.item(),3))+"_s"+str(round(sample_idx.item(),3))] = imgs
                f = open(args.save_folder+"/dict.pkl", "wb")
                pkl.dump(imgs_dict,f)
                f.close()
        del pipe
    if "eval" in args.stages:
        captions = list(dci_prompts.values())[-1024:]
        pipe = load_pipe(args)
        pipe = quantize_pipe(pipe, args)
        
        # Add memory optimization
        torch.cuda.empty_cache()  # Clear CUDA cache before starting
        
        # Reduce batch size if needed to avoid OOM
        effective_batchsize = min(args.batchsize, 5)  # Limit batch size to 5
        print(f"Using effective batch size of {effective_batchsize} to avoid OOM")

        ids = args.key.split('_')
        m_idx = ids[0]  # e.g., "m0.867"
        sample_idx = ids[1]  # e.g., "s1.0"
        m_idx = float(m_idx[1:])  # Remove 'm' and convert to float
        sample_idx = float(sample_idx[1:])  # Remove 's'
        pipe.scheduler.model_coeff = torch.linspace(m_idx,1, args.num_inference_steps)
        pipe.scheduler.sample_coeff = torch.linspace(sample_idx,1, args.num_inference_steps)
        imgs = []
        for batch in range(0, len(captions), effective_batchsize):
            torch.manual_seed(args.seed)
            # Clear memory between batches
            torch.cuda.empty_cache()
            
            if args.checkpoint == "FLUX.1":
                imgs_batch = pipe(prompt=captions[batch:batch+effective_batchsize], 
                                    num_inference_steps=args.num_inference_steps, 
                                    guidance_scale=guidance_scale).images
            else:
                imgs_batch = pipe(prompt=captions[batch:batch+effective_batchsize], 
                                    num_inference_steps=args.num_inference_steps, 
                                    guidance_scale=guidance_scale, 
                                    eta=args.eta).images
            imgs.extend(imgs_batch)

        del pipe
    if 'score' in args.stages or "eval" in args.stages:
        if "eval" in args.stages:
            imgs_dict = {args.key: imgs}
        
        for index, (key, value) in enumerate(reversed(imgs_dict.items())):
            imgs = value
            tensorized_images = [torchvision.transforms.functional.pil_to_tensor(img) for img in imgs]
            tensorized_images = torch.stack(tensorized_images).to("cuda")
            brisque_score = piq.brisque(tensorized_images, data_range=255, reduction='none')
            clipiqa = []
            clip_score = 0
            hpv2_results = []
            aqmap_results = []
            pqmap_results = []

            clipiqa_prompt="quality" # 'complexity','noisiness', 'quality'
            clipiqa_metric = CLIPImageQualityAssessment(prompts=(clipiqa_prompt,)).to("cuda")
            num_batches = math.ceil(len(imgs) / args.batchsize)
            for batch in range(0, num_batches):
                clipiqa_score = clipiqa_metric(tensorized_images[batch:batch+args.batchsize])
                clipiqa.append(clipiqa_score)
            del clipiqa_metric  # Free memory
            
            
            # Check if scores already exist in the dictionary
            try:
                hpv2_results = scores_dict[key][ranking_dict["hpv2"]]
                print(f"Found existing hpsv2 scores for {key}")
            except (KeyError, IndexError):
                print(f"Computing new hpsv2 scores for {key}")
                for i, (image, caption) in enumerate(zip(imgs, captions)):
                    hpv2_score = hpsv2.score(image, caption, hps_version="v2.1")
                    hpv2_results.append(hpv2_score)
            
            # Check if AQ-Map scores already exist
            try:
                aqmap_results = scores_dict[key][ranking_dict["aqmap"]]
                print(f"Found existing AQ-Map scores for {key}")
            except (KeyError, IndexError):
                print(f"Computing new AQ-Map scores for {key}")
                # Import AQ-Map module only when needed
                import_aqmap()
                
                aqmap_results = []
                for i, (image, caption) in enumerate(zip(imgs, captions)):
                    # Save the image temporarily to evaluate with AQ-Map
                    temp_img_path = os.path.join(args.save_folder, f"temp_img_{i}.jpg")
                    image.save(temp_img_path)
                    
                    # Use the imported module instead of subprocess
                    try:
                        # Call the function directly
                        score_value = evaluate_aqmap(temp_img_path, caption)
                        aqmap_results.append(score_value)
                    except Exception as e:
                        print(f"Error running AQ-Map for image {i}: {e}")
                        aqmap_results.append(0.0)  # Default score if evaluation fails
                    
                    # Clean up the temporary image
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)

            
            
            # Check if PQ-Map scores already exist
            try:
                pqmap_results = scores_dict[key][ranking_dict["pqmap"]]
                print(f"Found existing PQ-Map scores for {key}")
            except (KeyError, IndexError):
                print(f"Computing new PQ-Map scores for {key}")
                # Import PQ-Map module only when needed
                import_pqmap()
                
                pqmap_results = []
                for i, image in enumerate(imgs):
                    # Save the image temporarily to evaluate with PQ-Map
                    temp_img_path = os.path.join(args.save_folder, f"temp_img_{i}.jpg")
                    image.save(temp_img_path)
                    
                    try:
                        # Call the function directly
                        _, score_value = evaluate_pqmap(image, preprocessor, multi=False, draw=False)
                        pqmap_results.append(score_value)
                    except Exception as e:
                        print(f"Error running PQ-Map for image {i}: {e}")
                        pqmap_results.append(0.0)  # Default score if evaluation fails
                    
                    # Clean up the temporary image
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
            
            print("Computing CLIP Score for key:", key)
            num_batches = math.ceil(len(imgs) / args.batchsize)
            clipscore_metric = CLIPScore().to("cuda")
            try:
                clip_score = scores_dict[key][0]
            except:
                for batch in range(0, num_batches):
                    clipscore_metric.update(tensorized_images[batch:batch+args.batchsize], captions[batch:batch+args.batchsize])
                clip_score = clipscore_metric.compute().item()
            del clipscore_metric  # Free memory
            
            if "eval" in args.stages:
                from metrics.pickscore import calc_probs
                pickscore = calc_probs(captions, imgs)
                
                clipiqa = torch.stack(clipiqa).mean().item()
                hpv2_results = float(np.mean(hpv2_results))
                pickscore = float(np.mean(pickscore))

                processed_scores = []
                for score in aqmap_results:
                    if isinstance(score, np.ndarray):
                        if score.size == 1:
                            processed_scores.append(score.item())
                        else:
                            # If array has multiple elements, take the mean
                            processed_scores.append(np.mean(score))
                    else:
                        processed_scores.append(float(score))
                aqmap_results = np.array(processed_scores)
                aqmap_results = np.mean(aqmap_results).astype(np.float32)
                print('All Scores for key:', key)
                print("CLIP Score:", clip_score)
                print("CLIP-IQA Scores:",clipiqa)
                print("HPV2 Scores:",hpv2_results)
                print("AQ-Map Scores:", aqmap_results)
                print("JAQ loss:", hpv2_results + args.k * aqmap_results)
                print("PickScore:", pickscore)
                print("PQ-Map Scores:", pqmap_results)
                json.dump({
                    "clip_score": clip_score,
                    "clipiqa": clipiqa,
                    "pickscore": pickscore,
                    "hpv2": hpv2_results,
                    "aqmap": aqmap_results,
                    "pqmap": pqmap_results
                }, open(args.save_folder+"/scores_"+key+".json", "w"), indent=4)
                exit()

            brisque_score = 0
            scores_dict[key] = [clip_score, clipiqa, None, brisque_score, hpv2_results, aqmap_results, pqmap_results]

            f = open(args.save_folder+"/scores_dict.pkl", "wb")
            pkl.dump(scores_dict,f)
            f.close()
            
    if "rank" in args.stages:
        
        rank_id = int(ranking_dict[args.ranking_criteria])
        clipiqa_prompt="quality"
        rank_dict = {}
        if args.ranking_criteria == 'new':
            print("computing new ranking score")
            for index, (key, value) in enumerate(scores_dict.items()):
                try:
                    new_score = torch.log(value[0])+args.k*(torch.mean(value[1][clipiqa_prompt])) # 0.8
                except:
                    new_score = torch.log(value[0])+args.k*(torch.mean(value[1]))

                rank_dict[key] = new_score
        if args.ranking_criteria == 'kclip2':
            print("computing kclip2 ranking score")
            for index, (key, value) in enumerate(scores_dict.items()):
                try:
                    # Get AQ-Map scores and ensure they're in a consistent format
                    aqmap_scores = value[ranking_dict["aqmap"]]
                    if isinstance(aqmap_scores, (list, np.ndarray)):
                        # Convert to numpy array and handle any nested arrays
                        processed_aqmap = []
                        for score in aqmap_scores:
                            if isinstance(score, np.ndarray):
                                if score.size == 1:
                                    processed_aqmap.append(score.item())
                                else:
                                    processed_aqmap.append(np.mean(score))
                            else:
                                processed_aqmap.append(float(score))
                        aqmap_mean = np.mean(processed_aqmap)
                    else:
                        aqmap_mean = float(aqmap_scores)

                    # Get HPV2 scores (keep original handling)
                    hpv2_mean = np.mean(value[ranking_dict["hpv2"]])

                    new_score = aqmap_mean + args.k * hpv2_mean
                except Exception as e:
                    print(f"Error computing kclip2 score for {key}: {e}")
                    new_score = 0.0  # Default score if computation fails

                rank_dict[key] = new_score
        
        if args.ranking_criteria == 'aqmap':
            print("computing AQ-Map ranking score")
            for index, (key, value) in enumerate(scores_dict.items()):
                # Handle AQ-Map scores which are numpy arrays
                aqmap_scores = value[5]  # This should be a list of numpy arrays
                if isinstance(aqmap_scores, (list, np.ndarray)):
                    # If it's a list of numpy arrays, handle each array appropriately
                    processed_scores = []
                    for score in aqmap_scores:
                        if isinstance(score, np.ndarray):
                            if score.size == 1:
                                processed_scores.append(score.item())
                            else:
                                # If array has multiple elements, take the mean
                                processed_scores.append(np.mean(score))
                        else:
                            processed_scores.append(float(score))
                    aqmap_scores = np.array(processed_scores)
                    aqmap_score = torch.log(value[0]) + args.k * np.mean(aqmap_scores)
                else:
                    # If it's a single score, convert to float
                    aqmap_score = torch.log(value[0]) + args.k * float(aqmap_scores)
                rank_dict[key] = aqmap_score
        
        if args.ranking_criteria == 'pqmap':
            print("computing PQ-Map ranking score")
            for index, (key, value) in enumerate(scores_dict.items()):
                # Handle PQ-Map scores
                pqmap_scores = value[6]  # This should be a list of scores
                if isinstance(pqmap_scores, (list, np.ndarray)):
                    # If it's a list of scores, convert to numpy array and compute mean
                    pqmap_scores = np.array([float(score) for score in pqmap_scores])
                    pqmap_score = torch.log(value[0]) + args.k * np.mean(pqmap_scores)
                else:
                    # If it's a single score, convert to float
                    pqmap_score = torch.log(value[0]) + args.k * float(pqmap_scores)
                rank_dict[key] = pqmap_score

        if args.ranking_criteria == 'clipiqa':
            key, value = max(scores_dict.items(), key=lambda item: torch.mean(item[1][rank_id][clipiqa_prompt]))
        elif args.ranking_criteria in ["hpv2"]:
            key, value = max(scores_dict.items(), key=lambda item: np.mean(item[1][rank_id]) )
        elif args.ranking_criteria in ["brisque"]:
            key, value = min(scores_dict.items(), key=lambda item: torch.mean(item[1][rank_id]) )
        elif args.ranking_criteria in ["new", "kclip2", "aqmap", "pqmap"]:
            key, value = max(scores_dict.items(), key=lambda item: item[1])
            print(key, value)
        else:
            key, value = max(scores_dict.items(), key=lambda item: item[1][rank_id])
        
        # Save best ranking info to config file
        # Parse m and s coefficients from the key (format: "m{model_coeff}_s{sample_coeff}")
        m_coeff = float(key.split('_')[0][1:])  # remove 'm' and convert to float
        s_coeff = float(key.split('_')[1][1:])  # remove 's' and convert to float
        
        # Get the original scores for this key
        original_scores = scores_dict[key]
        
        best_rank_info = {
            "model_coeff": m_coeff,
            "sample_coeff": s_coeff,
            "clip_score": float(original_scores[0].item()),
            "ranking_criteria": args.ranking_criteria,
            "k_value": args.k,
            "clipiqa_scores": [float(score.item()) for score in original_scores[1][clipiqa_prompt]],
            "checkpoint": args.checkpoint,
            "quant_type": args.quant_type,
            "eta": args.eta
        }
        
        config_path = os.path.join(args.save_folder, f"best_rank_config_{args.ranking_criteria}.json")
        with open(config_path, "w") as f:
            json.dump(best_rank_info, f, indent=4)
            
        fig, ax = plt.subplots(num_imgs_per_col,num_imgs_per_row, figsize=(3*num_imgs_per_row,3*num_imgs_per_col))
        if num_imgs_per_col == 1:
            ax = ax.reshape(1, -1)  # Reshape to 2D array for consistent indexing
        imgs = imgs_dict[key]
        clip_score = original_scores[0]
        try:
            clipiqa = original_scores[1][clipiqa_prompt]
        except:
            clipiqa = original_scores[1]
        for j in range(0, num_imgs_per_col):
            for i in range(0, num_imgs_per_row):
                idx =i+j*num_imgs_per_row
                ax[j, i].imshow(imgs[idx])
                ax[j, i].set_title("CLIP-IQA: "+str(round(clipiqa[idx].item(),3)))
                ax[j, i].set_axis_off()
        plt.suptitle(key+"\nClip Score: "+str(round(clip_score,2)))
        plt.tight_layout()
        fname=args.save_folder+"/best_"+args.ranking_criteria+"_k"+str(args.k)+"_m"+str(m_coeff)+"_s"+str(s_coeff)+".png"
        print(fname)
        plt.savefig(fname, dpi=300)
        plt.close()
