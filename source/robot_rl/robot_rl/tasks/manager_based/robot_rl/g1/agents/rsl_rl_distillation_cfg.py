from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

@configclass
class G1DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120 #longer steps was not necessary for distillation
    max_iterations = 300
    save_interval = 50
    experiment_name = "g1"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=10,
        learning_rate=5.0e-4,#huber seems stable even with higher lr
        gradient_length=2,
        max_grad_norm=2.0,
        loss_type="huber",
    )
