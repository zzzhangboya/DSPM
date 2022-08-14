import numpy as np
import pandas as pd 
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from stockdata import prepare_stock_data
from models.cond_refinenet_dilated import CondRefineNetDilated

__all__ = ['StockRunner']


class StockRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                            betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    # def logit_transform(self, image, lam=1e-6):
    #     image = lam + (1 - 2 * lam) * image
    #     return torch.log(image) - torch.log1p(-image)

    def train(self):
        # if self.config.data.random_flip is False:
        #     tran_transform = test_transform = transforms.Compose([
        #         transforms.Resize(self.config.data.image_size),
        #         transforms.ToTensor()
        #     ])
        # else:
        #     tran_transform = transforms.Compose([
        #         transforms.Resize(self.config.data.image_size),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.ToTensor()
        #     ])
        #     test_transform = transforms.Compose([
        #         transforms.Resize(self.config.data.image_size),
        #         transforms.ToTensor()
        #     ])

        dataset = pd.read_csv(self.args.train_data_dir)
        val_dataset = pd.read_csv(self.args.val_data_dir)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        val_iter = iter(val_loader)
        self.config.input_dim = self.config.data.stock_number * self.config.data.time_window * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                            self.config.model.num_classes))).float().to(self.config.device)

        for epoch in range(self.config.training.n_epochs):
            for i, X in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                # X = X / 256. * 255. + torch.rand_like(X) / 256.
                # if self.config.data.logit_transform:
                #     X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                            n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        val_X = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_X = next(val_iter)

                    val_X = val_X.to(self.config.device)
                    # test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    # if self.config.data.logit_transform:
                    #     test_X = self.logit_transform(test_X)

                    val_labels = torch.randint(0, len(sigmas), (val_X.shape[0],), device=val_X.device)

                    with torch.no_grad():
                        val_dsm_loss = anneal_dsm_score_estimation(score, val_X, val_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    tb_logger.add_scalar('val_dsm_loss', val_dsm_loss, global_step=step)

                if step % self.config.training.save_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    # def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
    #     images = []

    #     labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
    #     labels = labels.long()

    #     with torch.no_grad():
    #         for _ in range(n_steps):
    #             images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
    #             noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
    #             grad = scorenet(x_mod, labels)
    #             x_mod = x_mod + step_lr * grad + noise
    #             x_mod = x_mod
    #             print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

    #         return images

    # def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    #     images = []

    #     with torch.no_grad():
    #         for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
    #             labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
    #             labels = labels.long()
    #             step_size = step_lr * (sigma / sigmas[-1]) ** 2
    #             for s in range(n_steps_each):
    #                 images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
    #                 noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
    #                 grad = scorenet(x_mod, labels)
    #                 x_mod = x_mod + step_size * grad + noise
    #                 # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
    #                 #                                                          grad.abs().max()))

    #         return images


    # def test(self):
    #     states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
    #     score = CondRefineNetDilated(self.config).to(self.config.device)
    #     score = torch.nn.DataParallel(score)

    #     score.load_state_dict(states[0])

    #     if not os.path.exists(self.args.image_folder):
    #         os.makedirs(self.args.image_folder)

    #     sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
    #                                 self.config.model.num_classes))

    #     score.eval()
    #     grid_size = 5

    #     imgs = []
    #     if self.config.data.dataset == 'MNIST':
    #         samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)
    #         all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

    #         for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
    #             sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
    #                                 self.config.data.image_size)

    #             if self.config.data.logit_transform:
    #                 sample = torch.sigmoid(sample)

    #             image_grid = make_grid(sample, nrow=grid_size)
    #             if i % 10 == 0:
    #                 im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    #                 imgs.append(im)

    #             save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
    #             torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))


    #     else:
    #         samples = torch.rand(grid_size ** 2, 3, 32, 32, device=self.config.device)

    #         all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

    #         for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
    #             sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
    #                                 self.config.data.image_size)

    #             if self.config.data.logit_transform:
    #                 sample = torch.sigmoid(sample)

    #             image_grid = make_grid(sample, nrow=grid_size)
    #             if i % 10 == 0:
    #                 im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    #                 imgs.append(im)

    #             save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)), nrow=10)
    #             torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

    #     imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)

    def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_stock, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):

        refer_stock = refer_stock.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_stock = refer_stock.contiguous().view(-1, self.config.data.channels, self.config.data.stock_number, self.config.data.time_window)
        x_mod = x_mod.view(-1, self.config.data.channels, self.config.data.stock_number, self.config.data.time_window)
        # half_refer_image = refer_image[..., :16]
        past_return = refer_stock[..., :self.config.data.past_time_step]
        future_return = refer_stock[..., self.config.data.past_time_step:]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                # corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                corrupted_past_return = past_return + torch.randn_like(past_return) * sigma
                x_mod[:, :, :, :self.config.data.past_time_step] = corrupted_past_return
                for s in range(n_steps_each):
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :, :, :self.config.data.past_time_step] = corrupted_past_return
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return x_mod

    def test_inpainting(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))
        score.eval()

        predictions = pd.DataFrame()
        
        # transform = transforms.Compose([
        #     transforms.Resize(self.config.data.image_size),
        #     transforms.ToTensor()
        # ])

        test_dataset = pd.read_csv(self.args.test_data_dir)
        dataloader = DataLoader(test_dataset, batch_size=self.config.inpainting.batch_size, shuffle=True, num_workers=4)
        for i, ref_stock in enumerate(dataloader):

            samples = torch.rand(self.config.inpainting.batch_size, self.config.inpainting.repeat_number, self.config.data.channels, self.config.data.stock_number,
                            self.config.data.time_window).to(self.config.device)

            prediction = self.anneal_Langevin_dynamics_inpainting(samples, ref_stock, score, sigmas, 100, 0.00002)
            predictions.append(prediction)

        return predictions
