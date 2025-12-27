#convert distilled student policy + teacher's critic to PPO format
import torch
import os

# Load BOTH checkpoints
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
project_path = os.path.join(project_root, "logs/g1_policies/stepping_stone/stepping_stone")
distill_run = "2025-11-29_22-00-22InitialDistill"
teacher_run = "2025-11-29_18-39-48walkingteacher"
distill_ckpt = torch.load(os.path.join(project_path, distill_run, "model_1000.pt"))
teacher_ckpt = torch.load(os.path.join(project_path, teacher_run, "model_1000.pt"))

print("="*60)
print("Converting distilled student + teacher's critic to PPO format")
print("="*60)

# Start with empty state dict
new_dict = {}

# 1. Student -> Actor
print("\n1. Converting student → actor...")
for k, v in distill_ckpt["model_state_dict"].items():
    if k == "std":
        new_dict[k] = v
        print(f"  ✓ Keeping: {k}")
    elif k.startswith("student."):
        new_key = k.replace("student.", "actor.")
        new_dict[new_key] = v
        print(f"  ✓ {k} → {new_key}")

# 2. Teacher's Critic -> Critic (from teacher PPO checkpoint)
print("\n2. Copying teacher's critic...")
for k, v in teacher_ckpt["model_state_dict"].items():
    if k.startswith("critic."):
        new_dict[k] = v
        print(f"  ✓ Copied: {k}")

print("\n3. Verifying dimensions...")
print(f"  Actor output: {new_dict['actor.6.weight'].shape[0]} dims (actions)")
print(f"  Critic output: {new_dict['critic.6.weight'].shape[0]} dims (value)")

# Create new checkpoint with ALL required fields
new_ckpt = {
    "model_state_dict": new_dict,
    "optimizer_state_dict": distill_ckpt["optimizer_state_dict"],  # Use distillation's optimizer state
    "iter": distill_ckpt["iter"],
}

# Copy any other fields from teacher checkpoint
for key in teacher_ckpt.keys():
    if key not in new_ckpt:
        new_ckpt[key] = teacher_ckpt[key]
        print(f"  ✓ Copied additional field: {key}")

# Save
output_path = os.path.join(project_path, distill_run, "model_ppo.pt")
torch.save(new_ckpt, output_path)

print("\n" + "="*60)
print("✓ SUCCESS!")
print(f"  Student policy (actor): from distillation")
print(f"  Value function (critic): from teacher PPO")
print(f"  Optimizer state: from teacher PPO")
print(f"  Saved to: {output_path}")
print("="*60)