import torch


def generate_diffusion(x, model, noise_scheduler, device: str, num_inference_steps: int = 100):
    x, model = x.to(device), model.to(device)
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    bs = x.shape[0]
    for i, t in enumerate(noise_scheduler.timesteps):
        model_input = noise_scheduler.scale_model_input(x, t)

        t_batch = torch.full(
            size=(bs,),
            fill_value=t.item(),
            dtype=torch.long
        ).cuda()

        with torch.no_grad():
            noise_pred = model(
                model_input,
                t_batch
            )

        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x


def generate_flow_matching(model, x_0, n_steps: int = 100, device: str = "cuda"):
    model.to(device)
    model.eval()
    x_t = x_0.to(device)

    bs = x_0.shape[0]
    
    eps = 1e-8
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)

    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            f_eval = model(x_t, t_prev)
            
        x_t = x_t + (t[i] - t[i - 1]) * f_eval

    return x_t