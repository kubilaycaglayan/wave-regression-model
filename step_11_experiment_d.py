"""Paired-seed frozen vs partial EfficientNet-B0 fine-tuning experiment."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import step_4_a_wave_dataset  # load transforms and conditional TorchVision fallback first
import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.models.efficientnet import MBConv
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from step_4_a_wave_dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, WaveDataset, build_evaluation_transform, build_training_transform

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "step-3-dataset-splits/snapshots/20260928T120720292607Z"
OUTPUT = ROOT / "step-11-experiment-d-efficientnet-partial-finetuning"
SEEDS = (42, 43, 44)
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 8, 100, 10
VERSION_PATTERN = __import__("re").compile(r"^v(?P<n>\d+)-(?P<t>[^/]+)$")

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def make_cuda_usable_if_needed() -> None:
    """Match the repository's fallback for installed cuDNN engine failures."""
    if not torch.cuda.is_available(): return
    try:
        probe=nn.Conv2d(3,4,kernel_size=3).cuda()
        probe(torch.zeros(1,3,16,16,device="cuda"))
    except RuntimeError:
        torch.backends.cudnn.enabled=False; torch.cuda.empty_cache()
        print("CUDA convolution probe failed; disabled cuDNN backend for this run",flush=True)

def snapshot_info(path: Path) -> dict[str,Any]:
    manifest_path=path/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    expected={"train":76,"validation":17,"test":16}; files={}
    for split,count in expected.items():
        p=path/f"{split}.csv"
        if not p.is_file() or manifest["files"][split]["count"] != count or sha(p)!=manifest["files"][split]["sha256"]:
            raise ValueError(f"Snapshot {split} file/count/hash mismatch: {p}")
        with p.open(newline="",encoding="utf8") as f: actual=sum(1 for _ in csv.DictReader(f))
        if actual != count: raise ValueError(f"Snapshot {split} has {actual} rows, expected {count}")
        files[split]={"path":str(p.resolve()),"sha256":sha(p),"count":actual}
    return {"name":path.name,"manifest":str(manifest_path.resolve()),"manifest_sha256":sha(manifest_path),"files":files}

class Data(pl.LightningDataModule):
    def __init__(self,snapshot:Path,image_dir:Path):
        super().__init__(); self.snapshot=snapshot; self.image_dir=image_dir
    def setup(self,stage=None):
        self.train=WaveDataset(self.snapshot/"train.csv",self.image_dir,build_training_transform())
        self.val=WaveDataset(self.snapshot/"validation.csv",self.image_dir,build_evaluation_transform())
    def train_dataloader(self): return DataLoader(self.train,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
    def val_dataloader(self): return DataLoader(self.val,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)

class Model(pl.LightningModule):
    def __init__(self,arm:str,learning_rate_head:float=1e-3,learning_rate_backbone:float=1e-4):
        super().__init__(); self.save_hyperparameters()
        self.backbone=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        dim=self.backbone.classifier[-1].in_features; self.backbone.classifier=nn.Identity()
        self.regression_head=nn.Sequential(nn.Linear(dim,1),nn.Sigmoid()); self.loss=nn.SmoothL1Loss()
        for p in self.backbone.parameters(): p.requires_grad=False
        if arm=="partial":
            # Discover the final MBConv and the following feature projection
            # from the instantiated torchvision model, then persist names.
            children=list(self.backbone.features.named_children())
            stages=[(i,name,module) for i,(name,module) in enumerate(children) if any(isinstance(child,MBConv) for child in module.modules())]
            if not stages: raise RuntimeError("EfficientNet-B0 has no MBConv feature stages")
            last_index,last_name,last_mbconv=stages[-1]
            later=children[last_index+1:]
            if not later: raise RuntimeError("Could not discover final feature projection after the last MBConv stage")
            projection_name,projection=later[-1]
            self.trainable_feature_module_names=[f"backbone.features.{last_name}",f"backbone.features.{projection_name}"]
            for module in (last_mbconv,projection):
                for p in module.parameters(): p.requires_grad=True
            # BatchNorm affine weights and biases stay frozen even within the
            # selected feature stages; their running statistics stay in eval.
            for module in self.backbone.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):
                    for p in module.parameters(): p.requires_grad=False
        self.arm=arm
        if arm=="frozen": self.trainable_feature_module_names=[]
    def train(self,mode=True):
        super().train(mode); self.backbone.eval(); self.regression_head.train(mode)
        if mode and self.arm=="partial":
            for qualified in self.trainable_feature_module_names:
                module=self.get_submodule(qualified)
                module.train(True)
        for name,module in self.backbone.named_modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
        return self
    def forward(self,x): return self.regression_head(self.backbone(x)).squeeze(-1)
    def training_step(self,batch,batch_idx):
        x,y=batch; pred=self(x); loss=self.loss(pred,y)
        self.log("train_loss",loss,on_step=False,on_epoch=True,batch_size=x.size(0))
        self.log("train_mae",(pred-y).abs().mean(),on_step=False,on_epoch=True,batch_size=x.size(0)); return loss
    def validation_step(self,batch,batch_idx):
        x,y=batch; pred=self(x); self.log("val_mae",(pred-y).abs().mean(),on_step=False,on_epoch=True,batch_size=x.size(0))
    def configure_optimizers(self):
        groups=[{"params":self.regression_head.parameters(),"lr":1e-3}]
        backbone=[p for p in self.backbone.parameters() if p.requires_grad]
        if backbone: groups.insert(0,{"params":backbone,"lr":1e-4})
        return torch.optim.AdamW(groups)

class EpochMetrics(Callback):
    """Keep epoch metrics to pair train MAE with the selected validation epoch."""
    def __init__(self): self.rows=[]
    def on_validation_end(self,trainer,pl_module):
        values=trainer.callback_metrics
        if "train_mae" in values and "val_mae" in values:
            self.rows.append({"epoch":int(trainer.current_epoch)+1,"train_mae":float(values["train_mae"].detach().cpu()),"validation_mae":float(values["val_mae"].detach().cpu())})

def module_names(model):
    return model.trainable_feature_module_names

def verify(model:Model,arm:str):
    trainable_backbone=[n for n,p in model.backbone.named_parameters() if p.requires_grad]
    bn=[(n,m.training,any(p.requires_grad for p in m.parameters())) for n,m in model.backbone.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    if any(training or trainable for _,training,trainable in bn): raise RuntimeError("BatchNorm must stay frozen and eval")
    if arm=="frozen" and trainable_backbone: raise RuntimeError(f"Frozen control has trainable backbone: {trainable_backbone}")
    if arm=="partial":
        bad=[n for n in trainable_backbone if not (n.startswith("features.7.") or n.startswith("features.8."))]
        if bad or not any(n.startswith("features.7.") for n in trainable_backbone) or not any(n.startswith("features.8.") for n in trainable_backbone):
            raise RuntimeError(f"Unexpected trainable backbone parameters: {trainable_backbone}")
    return {"arm":arm,"trainable_backbone_parameters":trainable_backbone,"trainable_backbone_modules":module_names(model),"trainable_head_parameters":[n for n,p in model.regression_head.named_parameters() if p.requires_grad],"batchnorm_modules":len(bn),"batchnorm_all_frozen_eval":True,"trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),"total_parameter_count":sum(p.numel() for p in model.parameters())}

def metrics(actual,pred):
    y=torch.tensor(actual,dtype=torch.float64); p=torch.tensor(pred,dtype=torch.float64); e=p-y; a=e.abs(); low=(y<=.39); high=(y>=.60); width=float(y.max()-y.min()); pwidth=float(p.max()-p.min())
    return {"mae":float(a.mean()),"rmse":float((e.square().mean()).sqrt()),"median_absolute_error":float(a.median()),"maximum_absolute_error":float(a.max()),"mean_signed_error":float(e.mean()),"prediction_minimum":float(p.min()),"prediction_maximum":float(p.max()),"prediction_range_width":pwidth,"actual_range_width":width,"prediction_actual_range_width_ratio":pwidth/width if width else None,"low_end_signed_bias":float(e[low].mean()) if low.any() else None,"low_end_count":int(low.sum()),"high_end_signed_bias":float(e[high].mean()) if high.any() else None,"high_end_count":int(high.sum())}

def evaluate(model,loader,device):
    model.to(device); model.eval(); ys=[]; ps=[]
    with torch.inference_mode():
        for x,y in loader: ys.extend(y.tolist()); ps.extend(model(x.to(device)).cpu().tolist())
    return ys,ps

def run(root:Path,arm:str,seed:int,snapshot:Path,image_dir:Path):
    run_id=f"{arm}-seed-{seed}"; dest=root/"runs"/arm/f"seed-{seed}"; metapath=dest/"metrics.json"
    if metapath.exists():
        old=json.loads(metapath.read_text())
        if old.get("status")=="complete": print(f"Skipping completed run {run_id}",flush=True); return old
    pending=dest.with_name(dest.name+f".pending-{int(time.time())}"); pending.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); make_cuda_usable_if_needed(); pl.seed_everything(seed,workers=True); data=Data(snapshot,image_dir); data.setup("fit"); model=Model(arm); model.train(); checks=verify(model,arm)
    config={"arm":arm,"seed":seed,"pretrained_weights":"EfficientNet_B0_Weights.DEFAULT (ImageNet)","loss":"SmoothL1Loss","optimizer":"AdamW","learning_rates":{"backbone":1e-4 if arm=="partial" else None,"regression_head":1e-3},"batch_size":BATCH_SIZE,"max_epochs":MAX_EPOCHS,"early_stopping":{"monitor":"val_mae","mode":"min","patience":PATIENCE},"deterministic":True,"augmentations":"RandomHorizontalFlip(p=0.5) + ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02)","image_size":list(IMAGE_SIZE),"imagenet_mean":list(IMAGENET_MEAN),"imagenet_std":list(IMAGENET_STD),"split_snapshot":snapshot_info(snapshot),"test_evaluated":False,"trainable_backbone_modules":checks["trainable_backbone_modules"]}
    dump(pending/"config.json",config); ckpt=ModelCheckpoint(dirpath=pending,filename=f"{run_id}-best-val-mae-{{epoch:02d}}-{{val_mae:.4f}}",monitor="val_mae",mode="min",save_top_k=1)
    early=EarlyStopping(monitor="val_mae",mode="min",patience=PATIENCE,verbose=True); history=EpochMetrics()
    trainer=pl.Trainer(accelerator="auto",devices=1,max_epochs=MAX_EPOCHS,callbacks=[ckpt,early,history],deterministic=True,logger=False,enable_progress_bar=True,log_every_n_steps=1)
    print(f"Starting {run_id}: {checks['trainable_parameter_count']:,} trainable parameters",flush=True); trainer.fit(model,datamodule=data)
    if not ckpt.best_model_path: raise RuntimeError(f"No best checkpoint for {run_id}")
    best=Model(arm); state=torch.load(ckpt.best_model_path,map_location="cpu",weights_only=False); best.load_state_dict(state["state_dict"]); best.train(); best_checks=verify(best,arm)
    actual,pred=evaluate(best,data.val_dataloader(),torch.device("cuda" if torch.cuda.is_available() else "cpu")); result_metrics=metrics(actual,pred)
    selected_epoch=min(history.rows,key=lambda row:row["validation_mae"])
    train_mae=selected_epoch["train_mae"]
    duration=time.perf_counter()-started; shutil_path=Path(ckpt.best_model_path); ckpt_sha=sha(shutil_path)
    final={"schema_version":1,"status":"complete","run_id":run_id,"config":config,"metrics":result_metrics,"training_mae_at_selected_epoch":train_mae,"train_validation_mae_gap":result_metrics["mae"]-train_mae,"selected_epoch":selected_epoch["epoch"],"epoch_metrics":history.rows,"epochs_trained":int(trainer.fit_loop.epoch_progress.current.completed),"training_duration_seconds":duration,"parameter_checks":best_checks,"checkpoint":{"path":str((dest/shutil_path.name).resolve()),"sha256":ckpt_sha},"validation_predictions":[{"filename":data.val.samples[i][0],"actual":actual[i],"predicted":pred[i],"absolute_error":abs(actual[i]-pred[i])} for i in range(len(actual))],"completed_at_utc":now(),"test_evaluated":False}
    if dest.exists(): raise FileExistsError(dest)
    pending.rename(dest); final["checkpoint"]["path"]=str((dest/shutil_path.name).resolve()); dump(dest/"metrics.json",final); print(f"Completed {run_id}: val MAE={result_metrics['mae']:.5f}, {duration:.1f}s",flush=True); return final

def aggregate(results,version:Path):
    by={arm:[r for r in results if r["config"]["arm"]==arm] for arm in ("frozen","partial")}; summary={}
    fields=("mae","rmse","train_validation_mae_gap","prediction_actual_range_width_ratio","low_end_signed_bias","high_end_signed_bias")
    for arm,runs in by.items():
        summary[arm]={}
        for key in fields:
            vals=[r["train_validation_mae_gap"] if key=="train_validation_mae_gap" else r["metrics"][key] for r in runs]
            summary[arm]["mean_"+key]=sum(vals)/len(vals); summary[arm]["sd_"+key]=float(torch.tensor(vals,dtype=torch.float64).std(unbiased=True))
    paired=[]
    for seed in SEEDS:
        f=next(r for r in by["frozen"] if r["config"]["seed"]==seed); p=next(r for r in by["partial"] if r["config"]["seed"]==seed)
        paired.append({"seed":seed,"mae_difference_partial_minus_frozen":p["metrics"]["mae"]-f["metrics"]["mae"]})
    images=[]
    ref=sorted(by["frozen"][0]["validation_predictions"],key=lambda x:x["filename"])
    for frow in ref:
        frozen_rows=[next(x for x in r["validation_predictions"] if x["filename"]==frow["filename"]) for r in by["frozen"]]
        partial_rows=[next(x for x in r["validation_predictions"] if x["filename"]==frow["filename"]) for r in by["partial"]]
        if any(x["actual"]!=frow["actual"] for x in frozen_rows+partial_rows): raise ValueError("Validation labels differ between runs")
        fe=sum(x["absolute_error"] for x in frozen_rows)/len(frozen_rows); pe=sum(x["absolute_error"] for x in partial_rows)/len(partial_rows)
        images.append({"filename":frow["filename"],"actual":frow["actual"],"frozen_mean_absolute_error":fe,"partial_mean_absolute_error":pe,"improvement":fe-pe})
    report={"schema_version":1,"status":"complete","aggregate":summary,"paired_seed_mae_differences":paired,"validation_per_image":images,"images_improved":sum(x["improvement"]>0 for x in images),"images_worsened":sum(x["improvement"]<0 for x in images),"largest_improvements":sorted(images,key=lambda x:x["improvement"],reverse=True)[:5],"largest_regressions":sorted(images,key=lambda x:x["improvement"])[:5],"test_evaluated":False}
    dump(version/"aggregate_metrics.json",report)
    lines=["# Experiment D results","",f"Version: `{version.name}`","","| Arm | Mean validation MAE | MAE SD | Mean RMSE | Mean train/validation gap | Mean range ratio | Low bias | High bias |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("frozen","partial"):
        s=summary[arm]; lines.append(f"| {arm} | {s['mean_mae']:.5f} | {s['sd_mae']:.5f} | {s['mean_rmse']:.5f} | {s['mean_train_validation_mae_gap']:.5f} | {s['mean_prediction_actual_range_width_ratio']:.4f} | {s['mean_low_end_signed_bias']:.4f} | {s['mean_high_end_signed_bias']:.4f} |")
    lines += ["","## Paired seed differences","","Partial minus frozen MAE; negative favors partial fine-tuning.",""]+[f"- Seed {x['seed']}: {x['mae_difference_partial_minus_frozen']:+.5f}" for x in paired]
    lines += ["",f"Validation images improved (seed-mean absolute error): {report['images_improved']}; worsened: {report['images_worsened']}.","","### Largest improvements","","| Image | Improvement |","|---|---:|"]+[f"| {x['filename']} | {x['improvement']:+.4f} |" for x in report["largest_improvements"]]+["","### Largest regressions","","| Image | Improvement |","|---|---:|"]+[f"| {x['filename']} | {x['improvement']:+.4f} |" for x in report["largest_regressions"]]+["","## Per-image comparison","","Each error is averaged across the three seeds in that arm.","","| Image | Actual | Frozen mean absolute error | Partial mean absolute error | Improvement |","|---|---:|---:|---:|---:|"]+[f"| {x['filename']} | {x['actual']:.3f} | {x['frozen_mean_absolute_error']:.4f} | {x['partial_mean_absolute_error']:.4f} | {x['improvement']:+.4f} |" for x in images]
    (version/"experiment_report.md").write_text("\n".join(lines)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--snapshot-dir",type=Path,default=SNAPSHOT); parser.add_argument("--image-dir",type=Path,default=ROOT/"step-2-final-water-data"); parser.add_argument("--arm",choices=("frozen","partial")); parser.add_argument("--seed",type=int,choices=SEEDS); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--version"); parser.add_argument("--force-new",action="store_true"); args=parser.parse_args()
    if bool(args.arm)!=bool(args.seed): parser.error("--arm and --seed must be supplied together")
    snap=args.snapshot_dir.resolve(); info=snapshot_info(snap); OUTPUT.mkdir(exist_ok=True)
    existing=sorted((p for p in OUTPUT.iterdir() if p.is_dir() and VERSION_PATTERN.fullmatch(p.name)),key=lambda p:int(VERSION_PATTERN.fullmatch(p.name).group("n")))
    if args.version:
        matches=[p for p in existing if p.name==args.version or p.name.startswith(args.version+"-")]
        if len(matches)!=1: raise ValueError(f"Version must uniquely match; found {[p.name for p in matches]}")
        version=matches[0]
    elif not args.force_new:
        incomplete=[p for p in reversed(existing) if (p/"experiment_manifest.json").is_file() and json.loads((p/"experiment_manifest.json").read_text()).get("status")!="complete"]
        version=incomplete[0] if incomplete else None
    else: version=None
    if version is None:
        n=max((int(VERSION_PATTERN.fullmatch(p.name).group("n")) for p in existing),default=0)+1; version=OUTPUT/f"v{n}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"; version.mkdir()
    configuration={"experiment":"D_partial_finetuning_efficientnet_b0","source_sha256":sha(Path(__file__).resolve()),"snapshot":info,"planned_runs":[f"{arm}-seed-{seed}" for arm in ("frozen","partial") for seed in SEEDS],"shared_training":{"pretrained_weights":"EfficientNet_B0_Weights.DEFAULT (ImageNet)","batch_size":BATCH_SIZE,"max_epochs":MAX_EPOCHS,"early_stopping":{"monitor":"val_mae","mode":"min","patience":PATIENCE},"loss":"SmoothL1Loss","optimizer":"AdamW","scheduler":None,"deterministic":True,"image_size":list(IMAGE_SIZE),"imagenet_mean":list(IMAGENET_MEAN),"imagenet_std":list(IMAGENET_STD),"augmentations":"RandomHorizontalFlip(p=0.5) + ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02)"},"control_lr_head":1e-3,"partial_lr_backbone":1e-4,"partial_lr_head":1e-3,"test_evaluated":False}
    manifest_path=version/"experiment_manifest.json"
    if manifest_path.is_file():
        prior=json.loads(manifest_path.read_text())
        if prior.get("configuration")!=configuration and not args.version:
            if args.force_new: pass
            else:
                n=max((int(VERSION_PATTERN.fullmatch(p.name).group("n")) for p in existing),default=0)+1
                version=OUTPUT/f"v{n}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                version.mkdir(); manifest_path=version/"experiment_manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest["configuration"]!=configuration: raise ValueError("Experiment version configuration differs; use --force-new")
        if manifest.get("status")=="complete" and not args.dry_run: raise RuntimeError(f"Completed version is immutable: {version}")
    else: manifest={"schema_version":1,"version_id":version.name,"created_at_utc":now(),"configuration":configuration,"status":"running","completed_runs":[]}
    dump(manifest_path,manifest)
    if args.dry_run:
        print(json.dumps({"version":str(version),"snapshot":info,"planned_runs":[f"{arm}-seed-{seed}" for arm in ("frozen","partial") for seed in SEEDS]},indent=2)); return
    selected=[(args.arm,args.seed)] if args.arm else [(arm,seed) for arm in ("frozen","partial") for seed in SEEDS]
    results=[]
    for arm,seed in selected: results.append(run(version,arm,seed,snap,args.image_dir.resolve()))
    # Aggregate once all six matching metrics exist; supports interruption/resume.
    complete=[]
    for arm in ("frozen","partial"):
        for seed in SEEDS:
            p=version/"runs"/arm/f"seed-{seed}"/"metrics.json"
            if p.is_file() and json.loads(p.read_text()).get("status")=="complete": complete.append(json.loads(p.read_text()))
    manifest["completed_runs"]=[r["run_id"] for r in complete]; manifest["updated_at_utc"]=now()
    if len(complete)==6:
        aggregate(complete,version); manifest["status"]="complete"; manifest["completed_at_utc"]=now(); manifest["artifacts"]={"aggregate":"aggregate_metrics.json","report":"experiment_report.md"}
    dump(manifest_path,manifest)
    print(f"Version {version.name}: {len(complete)}/6 runs complete")

if __name__=="__main__": main()
