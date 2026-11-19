"""Real IXI validation: inspect depth-plane fidelity of frozen CNN and 20-step hybrid.

This diagnostic draws its anatomical image content solely from acquired IXI MRI.
The CGH optical field is numerically simulated (not optical hardware data).
Select validation subjects explicitly; held-out test data are forbidden.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mri_pipeline import build_target_planes, load_volume
from subject_manifest import read_manifest
from train_mri_subjects import make_model, DEPTHS, WAVELENGTH_M, PITCH_M
from compare_mri_baselines import propagate_planes, target_scaled
from refine_mri_validation import optimize_checkpoints


def run(manifest, checkpoint, out, subject_ids, steps=20, lr=0.05, axis=2):
    if steps < 1 or lr <= 0 or axis not in (0,1,2):
        raise ValueError("Invalid steps, learning rate, or slice axis")
    records = {r["subject_id"]:r for r in read_manifest(manifest)
               if r["split"]=="validation"}
    if any(s not in records for s in subject_ids):
        raise ValueError("All subject IDs must exist in validation (not test)")
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if ckpt["model_name"]!="phase_cnn":
        raise ValueError("Expected frozen PhaseCNN checkpoint")
    if tuple(float(v) for v in ckpt["depths_m"])!=tuple(DEPTHS):
        raise ValueError("Depth parameters mismatch")
    for key, expected in (("wavelength_m",WAVELENGTH_M),("pixel_pitch_m",PITCH_M)):
        if not np.isclose(float(ckpt[key]),expected,rtol=1e-8,atol=0):
            raise ValueError("Checkpoint optical sampling mismatch")
    model=make_model("phase_cnn").cpu().eval()
    model.load_state_dict(ckpt["model"],strict=True)
    size=int(ckpt["size"])
    output=Path(out)
    output.mkdir(parents=True,exist_ok=True)
    table=[]
    for subject in subject_ids:
        vol, spacing=load_volume(records[subject]["path"])
        planes,indices=build_target_planes(vol,axis=axis,size=size)
        raw=torch.from_numpy(planes)
        target=target_scaled(planes)
        with torch.no_grad():
            phase=model(raw[None])[0].detach()
            cnn=propagate_planes(phase).detach()
        # Re-optimize and retain the final phase, rather than merely reporting
        # the numerical MSE from optimize_checkpoints.
        variable=torch.nn.Parameter(phase.clone())
        optimizer=torch.optim.Adam([variable],lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss=(propagate_planes(variable)-target).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            refined=propagate_planes(variable).detach()
        target_np=target.numpy()
        cnn_np=cnn.numpy()
        refined_np=refined.numpy()
        mse_cnn=((cnn_np-target_np)**2).mean(axis=(1,2))
        mse_hybrid=((refined_np-target_np)**2).mean(axis=(1,2))
        # Signal correlations are descriptive spatial pattern metrics, not
        # optical cross-talk or clinical fidelity measurements.
        correlations=[]
        for k in range(3):
            a=target_np[k].ravel()
            b=refined_np[k].ravel()
            correlations.append(float(np.corrcoef(a,b)[0,1])
                                if a.std()>1e-9 and b.std()>1e-9 else None)
            table.append({"subject_id":subject,"plane":k+1,
                          "z_m":DEPTHS[k],"cnn_mse":float(mse_cnn[k]),
                          "hybrid_mse":float(mse_hybrid[k]),
                          "hybrid_target_pearson":correlations[-1]})
        fig,axs=plt.subplots(4,3,figsize=(12,12),layout="constrained")
        for k in range(3):
            peak=max(float(np.percentile(target_np[k],99)),float(np.percentile(refined_np[k],99)),1e-8)
            axs[0,k].imshow(planes[k],cmap="gray",vmin=0,vmax=1)
            axs[0,k].set_title(f"Acquired MRI slice {indices[k]}")
            axs[1,k].imshow(target_np[k],cmap="gray",vmin=0,vmax=peak)
            axs[1,k].set_title(f"Target intensity z={DEPTHS[k]*1000:g} mm")
            axs[2,k].imshow(cnn_np[k],cmap="gray",vmin=0,vmax=peak)
            axs[2,k].set_title(f"PhaseCNN | MSE {mse_cnn[k]:.4f}")
            axs[3,k].imshow(refined_np[k],cmap="gray",vmin=0,vmax=peak)
            corr=correlations[-1]
            axs[3,k].set_title(f"Hybrid {steps} steps | MSE {mse_hybrid[k]:.4f} | r={corr:.2f}" if corr is not None else f"Hybrid {steps} steps | MSE {mse_hybrid[k]:.4f}")
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{subject}: real acquired MRI input, simulated CGH (not optical measurement)")
        fig.savefig(output/f"{subject}_depth_comparison.png",dpi=200)
        plt.close(fig)
        print(f"{subject}: CNN={float(mse_cnn.mean()):.5f}, hybrid={float(mse_hybrid.mean()):.5f}",flush=True)
    with (output/"per_plane_fidelity.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    report={"status":"real acquired IXI MRI validation targets, simulated optical propagation",
            "subjects":subject_ids,"steps":steps,"lr":lr,"test_set_accessed":False,
            "note":"Pearson spatial correlation is not a cross-talk measurement; only basic per-plane fidelity is evaluated."}
    (output/"run_info.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--subjects",nargs="+",default=["IXI-373","IXI-337","IXI-002"])
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--lr",type=float,default=0.05)
    p.add_argument("--axis",type=int,default=2)
    args=p.parse_args()
    run(args.manifest,args.checkpoint,args.out,args.subjects,args.steps,args.lr,args.axis)
