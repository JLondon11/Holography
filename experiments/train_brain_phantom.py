"""Brain-shaped SYNTHETIC phantom CGH experiment. No patient MRI or optical measurements.
This is a demonstration of computational capability, not publication-grade validation.
python experiments/train_brain_phantom.py --steps 35 --out results/brain_phantom
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from holography.models import PhaseCNN
from holography.torch_optics import propagate

DEPTHS = (.004,.006,.008)
def phantom(n, seed):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[-1:1:complex(n),-1:1:complex(n)]
    layers=[]
    for depth in (-.18,0,.18):
        outer=((xx/(.71-.11*abs(depth)))**2+((yy+.02)/(.80-.1*abs(depth)))**2)<1
        mid=((xx/.60)**2+((yy+.03)/.68)**2)<1
        inner=((xx/.49)**2+((yy+.03)/.54)**2)<1
        vent=(((xx-.095)/.065)**2+((yy+.03)/.18)**2<1)|(((xx+.095)/.065)**2+((yy+.03)/.18)**2<1)
        angle=np.arctan2(yy+.02,xx)
        rad=np.sqrt((xx/.71)**2+((yy+.02)/.8)**2)
        folds=(np.cos(21*angle+15*rad+depth*8)>.52)&(rad>.51)&(rad<.91)
        image=np.zeros((n,n),dtype=np.float32)
        image[outer]=.20;image[mid]=.55;image[inner]=.72
        image[folds&outer]=.31;image[vent]=.07
        image+=outer*rng.normal(0,.018,(n,n)).astype(np.float32)
        layers.append(np.clip(image,0,1))
    return np.stack(layers)

def reconstruct(phase):
    field=torch.polar(torch.ones_like(phase),phase).to(torch.complex64)
    return torch.stack([propagate(field,532e-9,3.74e-6,z).abs().square() for z in DEPTHS],1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=35)
    parser.add_argument("--size",type=int,default=96)
    parser.add_argument("--out",type=Path,default=Path("results/brain_phantom"))
    args=parser.parse_args()
    torch.manual_seed(20261030)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    targets=torch.from_numpy(np.stack([phantom(args.size,i) for i in (19,23,29,31)]))
    model=PhaseCNN(3);opt=torch.optim.Adam(model.parameters(),lr=.003)
    losses=[]
    for _ in range(args.steps):
        model.train();opt.zero_grad(set_to_none=True)
        loss=(reconstruct(model(targets[:3]))-targets[:3]).square().mean()
        loss.backward();opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        phase=model(targets[3:])[0]
        recon=reconstruct(phase[None])[0]
    mse=float((recon-targets[3]).square().mean())
    args.out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),args.out/"phase_cnn.pt")
    np.savez_compressed(args.out/"simulation.npz",target=targets[3].numpy(),
                        reconstructed=recon.numpy(),slm_phase=phase.numpy(),
                        loss=np.asarray(losses))
    manifest={"status":"synthetic demonstration NOT measured MRI","size":args.size,
        "steps":args.steps,"seed":20261030,"train_seeds":[19,23,29],
        "test_seed":31,"wavelength_m":532e-9,"pitch_m":3.74e-6,
        "depths_m":DEPTHS,"training_loss_initial":losses[0],
        "training_loss_final":losses[-1],"heldout_intensity_mse":mse,
        "note":"Numerical model is not calibrated to a physical SLM."}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
