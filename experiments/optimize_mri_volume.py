"""Optimize a single MRI-derived intensity target as a simulated phase-only CGH.

This is NOT a trained generalizing neural model, tissue optical transfer model
or physical measurement. Requires scipy, matplotlib, torch and local package.
python experiments/optimize_mri_volume.py --input brain.nii.gz --out results/brain_mri
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from holography.torch_optics import propagate
from mri_pipeline import load_volume,build_target_planes

def optimize(target,depths_m,steps=100,lr=.05,wavelength_m=532e-9,pitch_m=3.74e-6):
    torch.manual_seed(20261030)
    objective=torch.as_tensor(target)
    objective=objective/objective.mean(dim=(-2,-1),keepdim=True).clamp_min(.001)
    phase=torch.nn.Parameter(torch.zeros_like(objective[0]).uniform_(-.1,.1))
    optimizer=torch.optim.Adam([phase],lr=lr)
    history=[]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        source=torch.polar(torch.ones_like(phase),phase).to(torch.complex64)
        pred=torch.stack([propagate(source,wavelength_m,pitch_m,z).abs().square() for z in depths_m])
        loss=(pred-objective).square().mean()
        loss.backward();optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        field=torch.polar(torch.ones_like(phase),phase).to(torch.complex64)
        rec=torch.stack([propagate(field,wavelength_m,pitch_m,z).abs().square() for z in depths_m])
    return phase.detach().numpy(),objective.numpy(),rec.numpy(),history

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--axis',type=int,default=2)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--steps',type=int,default=100)
    args=p.parse_args()
    volume,spacing=load_volume(args.input)
    targets,indices=build_target_planes(volume,args.axis,size=args.size)
    depths=(.004,.006,.008)
    phase,target,rec,history=optimize(targets,depths,steps=args.steps)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'arrays.npz',target=target,reconstruction=rec,phase=phase,loss=history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(11,5.6),layout='constrained')
    axes[0,0].imshow(phase,cmap='twilight',vmin=-np.pi,vmax=np.pi);axes[0,0].set_title('Optimized SLM phase (rad)')
    axes[1,0].plot(history);axes[1,0].set_title('Optimization loss');axes[1,0].set_xlabel('Iteration')
    for i,z in enumerate(depths):
        axes[0,i+1].imshow(target[i],cmap='gray');axes[0,i+1].set_title(f'Target z={z*1000:g} mm')
        axes[1,i+1].imshow(rec[i],cmap='gray');axes[1,i+1].set_title(f'Simulated z={z*1000:g} mm')
    for ax in axes.ravel():
        if ax is not axes[1,0]: ax.set_axis_off()
    fig.suptitle('SIMULATED MRI-derived intensity hologram (NOT optical measurement)')
    fig.savefig(args.out/'reconstruction.png',dpi=220)
    fig.savefig(args.out/'reconstruction.pdf')
    plt.close(fig)
    manifest={'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'volume_shape':volume.shape,'voxel_spacing_mm':spacing,'selected_axis':args.axis,
              'selected_indices':indices,'depths_m':depths,'wavelength_m':532e-9,
              'slm_pixel_pitch_m':3.74e-6,'size':args.size,'steps':args.steps,
              'initial_loss':history[0],'final_loss':history[-1],
              'status':'simulated MRI-derived intensity targets, uncalibrated phase fit',
              'neural_network_trained':False,'physical_measurement':False}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
